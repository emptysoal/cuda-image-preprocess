#include "public.h"
#include "calibrator.h"
#include "mask2color.h"
#include "preprocess.h"

using namespace nvinfer1;


const char *      inputName = "data";
const char *      outputName = "mask";
const int         classesNum = 32;
const int         inputHeight = 448;
const int         inputWidth = 448;
const int         outputSize = inputHeight * inputWidth;
const std::string wtsFile = "./para.wts";
const std::string trtFile = "./model.plan";
const std::string dataPath = "../../../../Camvid_segment_dataset";
const std::string valDataPath = dataPath + "/images/val";
const std::string testDataPath = dataPath + "/images/test";  // 用于推理
static Logger     gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool        bFP16Mode = false;
// for INT8 mode
const bool        bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = valDataPath;  // 用于 int8 量化


// preprocess same as pytorch training
void imagePreProcess(cv::Mat& img, float* inputData)
{
    /*
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp_image = ((resized_img/255. - mean) / std).astype(np.float32)
    */

    // resize
    cv::Mat resizeImg;
    cv::resize(img, resizeImg, cv::Size(inputWidth, inputHeight));

    // transpose((2, 0, 1)) and bgr to rgb and normalize
    uchar* uc_pixel = resizeImg.data;
    for (int i = 0; i < inputHeight * inputWidth; i++)
    {
        inputData[i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229;  // R-0.485
        inputData[i + inputHeight * inputWidth] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
        inputData[i + 2 * inputHeight * inputWidth] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
        uc_pixel += 3;
    }
}

void imagePostProcess(std::string file_name, int* outputs, int originHeight, int originWidth)
{
    cv::Mat mask(inputHeight, inputWidth, CV_8UC1);
    uchar* uc_pixel = mask.data;
    for (int i = 0; i < inputHeight * inputWidth; i++)
    {
        uc_pixel[i] = (uchar)outputs[i];
    }
    // resize
    cv::Mat resizeMask;
    cv::resize(mask, resizeMask, cv::Size(originWidth, originHeight), 0, 0, cv::INTER_NEAREST);
    // blur
    cv::Mat blurMask;
    cv::medianBlur(resizeMask, blurMask, 3);
    // mask to color
    cv::Mat colorImg = cv::Mat::zeros(originHeight, originWidth, CV_8UC3);
    toColor(blurMask, colorImg);
    // save segmentation result
    cv::imwrite("mask_" + file_name, colorImg);
}


void inference_one(IExecutionContext* context, float* inputData, int* outputData, std::vector<void *> vBufferD, std::vector<int> vTensorSize)
{
    CHECK(cudaMemcpy(vBufferD[0], (void *)inputData, vTensorSize[0], cudaMemcpyHostToDevice));

    context->executeV2(vBufferD.data());

    CHECK(cudaMemcpy((void *)outputData, vBufferD[1], vTensorSize[1], cudaMemcpyDeviceToHost));
}


IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


IActivationLayer* bottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, int dilation, std::string lname)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + ".conv1.weight"], emptywts);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    conv2->setDilationNd(DimsHW{dilation, dilation});
    conv2->setPaddingNd(DimsHW{dilation, dilation});
    conv2->setStrideNd(DimsHW{stride, stride});
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + ".conv3.weight"], emptywts);
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4)
    {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + ".downsample.0.weight"], emptywts);
        conv4->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + ".downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    else
    {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }

    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);

    return relu3;
}


std::vector<IActivationLayer *> build_backbone(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, 64, DimsHW{7, 7}, weightMap["backbone.conv1.weight"], emptywts);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, 1, "backbone.layer1.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, 1, "backbone.layer1.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, 1, "backbone.layer1.2");
    IActivationLayer* low_level_layer = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, 1, "backbone.layer2.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, 1, "backbone.layer2.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, 1, "backbone.layer2.2");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, 1, "backbone.layer2.3");

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, 1, "backbone.layer3.0");
    for (int i = 1; i < 6; i++)
    {
        x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, 1, "backbone.layer3." + std::to_string(i));
    }

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 1, 2, "backbone.layer4.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, 4, "backbone.layer4.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, 8, "backbone.layer4.2");

    return {x, low_level_layer};
}


IActivationLayer* sub_aspp(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int kernel, int padding, int dilation, std::string lname)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    
    IConvolutionLayer* conv = network->addConvolutionNd(input, 256, DimsHW{kernel, kernel}, weightMap[lname + ".atrous_conv.weight"], emptywts);
    conv->setDilationNd(DimsHW{dilation, dilation});
    conv->setPaddingNd(DimsHW{padding, padding});
    conv->setStrideNd(DimsHW{1, 1});

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-5);

    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);

    return relu;
}


IActivationLayer* global_avg_pool(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname)
{
    int h = input.getDimensions().d[2];
    int w = input.getDimensions().d[3];
    IPoolingLayer* pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{h, w});

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv = network->addConvolutionNd(*pool->getOutput(0), 256, DimsHW{1, 1}, weightMap[lname + ".1.weight"], emptywts);
    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".2", 1e-5);
    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);

    return relu;
}


IActivationLayer* build_aspp(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input)
{
    IActivationLayer* aspp1 = sub_aspp(network, weightMap, input, 1, 0, 1, "aspp.aspp1");
    IActivationLayer* aspp2 = sub_aspp(network, weightMap, input, 3, 6, 6, "aspp.aspp2");
    IActivationLayer* aspp3 = sub_aspp(network, weightMap, input, 3, 12, 12, "aspp.aspp3");
    IActivationLayer* aspp4 = sub_aspp(network, weightMap, input, 3, 18, 18, "aspp.aspp4");

    IActivationLayer* gap = global_avg_pool(network, weightMap, input, "aspp.global_avg_pool");
    IResizeLayer* gap_rsz = network->addResize(*gap->getOutput(0));
    Dims32 dim{4, {1, 256, aspp4->getOutput(0)->getDimensions().d[2], aspp4->getOutput(0)->getDimensions().d[3]}};
    gap_rsz->setOutputDimensions(dim);
    gap_rsz->setResizeMode(ResizeMode::kLINEAR);
    gap_rsz->setCoordinateTransformation(ResizeCoordinateTransformation::kALIGN_CORNERS);

    ITensor* inputTensors[] = { aspp1->getOutput(0), aspp2->getOutput(0), aspp3->getOutput(0), aspp4->getOutput(0), gap_rsz->getOutput(0) };
    IConcatenationLayer* concat = network->addConcatenation(inputTensors, 5);

    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv = network->addConvolutionNd(*concat->getOutput(0), 256, DimsHW{1, 1}, weightMap["aspp.conv1.weight"], emptywts);
    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), "aspp.bn1", 1e-5);
    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);

    return relu;
}


IConvolutionLayer* build_decoder(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input2, 48, DimsHW{1, 1}, weightMap["decoder.conv1.weight"], emptywts);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "decoder.bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IResizeLayer* rsz = network->addResize(input1);
    Dims32 dim{4, {1, 256, relu1->getOutput(0)->getDimensions().d[2], relu1->getOutput(0)->getDimensions().d[3]}};
    rsz->setOutputDimensions(dim);
    rsz->setResizeMode(ResizeMode::kLINEAR);
    rsz->setCoordinateTransformation(ResizeCoordinateTransformation::kALIGN_CORNERS);

    ITensor* inputTensors[] = { rsz->getOutput(0), relu1->getOutput(0) };
    IConcatenationLayer* concat = network->addConcatenation(inputTensors, 2);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*concat->getOutput(0), 256, DimsHW{3, 3}, weightMap["decoder.last_conv.0.weight"], emptywts);
    conv2->setPaddingNd(DimsHW{1, 1});
    conv2->setStrideNd(DimsHW{1, 1});
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "decoder.last_conv.1", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), 256, DimsHW{3, 3}, weightMap["decoder.last_conv.4.weight"], emptywts);
    conv3->setPaddingNd(DimsHW{1, 1});
    conv3->setStrideNd(DimsHW{1, 1});
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), "decoder.last_conv.5", 1e-5);
    IActivationLayer* relu3 = network->addActivation(*bn3->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv4 = network->addConvolutionNd(*relu3->getOutput(0), classesNum, DimsHW{1, 1}, weightMap["decoder.last_conv.8.weight"], weightMap["decoder.last_conv.8.bias"]);

    return conv4;
}


void buildNetwork(INetworkDefinition* network, IOptimizationProfile* profile, IBuilderConfig* config, std::map<std::string, Weights>& weightMap)
{
    ITensor* inputTensor = network->addInput(inputName, DataType::kFLOAT, Dims32 {4, {-1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 3, inputHeight, inputWidth}});
    config->addOptimizationProfile(profile);

    // build backbone
    std::vector<IActivationLayer *> out = build_backbone(network, weightMap, *inputTensor);
    IActivationLayer* backbone_out = out[0];
    IActivationLayer* low_level_layer = out[1];
    std::cout << "Succeeded building backbone!" << std::endl;

    // build aspp
    IActivationLayer* aspp = build_aspp(network, weightMap, *backbone_out->getOutput(0));
    std::cout << "Succeeded building aspp!" << std::endl;

    // build decoder
    IConvolutionLayer* decoder = build_decoder(network, weightMap, *aspp->getOutput(0), *low_level_layer->getOutput(0));
    std::cout << "Succeeded building decoder!" << std::endl;

    // resize to origin shape
    IResizeLayer* rsz = network->addResize(*decoder->getOutput(0));
    rsz->setOutputDimensions(Dims32 {4, {1, classesNum, inputHeight, inputWidth}});
    rsz->setResizeMode(ResizeMode::kLINEAR);
    rsz->setCoordinateTransformation(ResizeCoordinateTransformation::kALIGN_CORNERS);

    // add topk layer
    ITopKLayer* top1 = network->addTopK(*rsz->getOutput(0), TopKOperation::kMAX, 1, 1U << 1);
    top1->getOutput(1)->setName(outputName);
    network->markOutput(*top1->getOutput(1));
    std::cout << "Succeeded building total network!" << std::endl;
}


ICudaEngine* getEngine()
{
    ICudaEngine *engine = nullptr;

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30);
        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode)
        {
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode)
        {
            config->setFlag(BuilderFlag::kINT8);
            int batchSize = 8;
            pCalibrator = new Int8EntropyCalibrator2(batchSize, inputWidth, inputHeight, calibrationDataPath.c_str(), cacheFile.c_str());
            config->setInt8Calibrator(pCalibrator);
        }
        // load .wts
        std::map<std::string, Weights> weightMap = loadWeights(wtsFile);

        buildNetwork(network, profile, config, weightMap);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        // Release host memory
        for (auto& mem : weightMap)
        {
            free((void*) (mem.second.values));
        }

        IRuntime* runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) { std::cout << "Failed building engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr)
        {
            delete pCalibrator;
        }

        std::ofstream engineFile(trtFile, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    return engine;
}


int run()
{
    ICudaEngine* engine = getEngine();

    IExecutionContext* context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {4, {1, 3, inputHeight, inputWidth}});

    std::vector<int> vTensorSize(2, 0);  // bytes of input and output
    for (int i = 0; i < 2; i++)
    {
        Dims32 dim = context->getBindingDimensions(i);
        int size = 1;
        for (int j = 0; j < dim.nbDims; j++)
        {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    // prepare input data and output data ---------------------------
    float inputData[3 * inputHeight * inputWidth];
    int outputData[outputSize];  // using int. output is index
    //  prepare input and output space on device
    std::vector<void *> vBufferD (2, nullptr);
    for (int i = 0; i < 2; i++)
    {
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(testDataPath.c_str(), file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    // inference
    int total_cost = 0;
    int img_count = 0;
    for (int i = 0; i < file_names.size(); i++)
    {
        std::string testImagePath = testDataPath + "/" + file_names[i];
        cv::Mat img = cv::imread(testImagePath, cv::IMREAD_COLOR);
        int originHeight = img.rows;
        int originWidth = img.cols;

        auto start = std::chrono::system_clock::now();
        // imagePreProcess(img, inputData);  // put image data on inputData
        preprocess(img, inputData, inputHeight, inputWidth);  // put image data on inputData
        inference_one(context, inputData, outputData, vBufferD, vTensorSize);
        auto end = std::chrono::system_clock::now();

        imagePostProcess(file_names[i], outputData, originHeight, originWidth);  // for visualization, not necessary

        total_cost += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        img_count++;
    }

    int avg_cost = total_cost / img_count;
    std::cout << "Total image num is: " << img_count;
    std::cout << " inference total cost is: " << total_cost << "ms";
    std::cout << " average cost is: " << avg_cost << "ms" << std::endl;

    // free device memory
    for (int i = 0; i < 2; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    return 0;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    return 0;
}

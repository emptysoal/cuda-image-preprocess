#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


__global__ void norm(const uchar* srcData, float* tgtData, const int h, const int w)
{
    /*
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        (img / 255. - mean) / std
    */
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * w;
    int idx3 = idx * 3;

    if (ix < w && iy < h)
    {
        tgtData[idx3] = ((float)srcData[idx3] / 255.0 - 0.406) / 0.225;  // B pixel
        tgtData[idx3 + 1] = ((float)srcData[idx3 + 1] / 255.0 - 0.456) / 0.224;  // G pixel
        tgtData[idx3 + 2] = ((float)srcData[idx3 + 2] / 255.0 - 0.485) / 0.229;  // R pixel
    }
}


void normalize(const std::string& imagePath)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    int w = img.cols;
    int h = img.rows;
    printf("Image width is %d, height is %d\n", w, h);
    int wh = w * h;
    int elements = wh * 3;
    // target
    float outputData[elements];

    // target on device
    float* tgtDevData;
    cudaMalloc((void**)&tgtDevData, sizeof(float) * elements);
    // source on device
    uchar* srcDevData;
    cudaMalloc((void**)&srcDevData, sizeof(uchar) * elements);
    cudaMemcpy(srcDevData, img.data, sizeof(uchar) * elements, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    printf("Block(%d, %d),Grid(%d, %d).\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    norm<<<gridSize, blockSize>>>(srcDevData, tgtDevData, h, w);
    // cudaDeviceSynchronize();

    cudaMemcpy(outputData, tgtDevData, sizeof(float) * elements, cudaMemcpyDeviceToHost);

    // print part of pixel for comparing
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            std::cout << (int)img.data[(i * w + j) * 3] << ",";  // B src
            std::cout << outputData[(i * w + j) * 3] << " ";  // B tgt
            std::cout << (int)img.data[(i * w + j) * 3 + 1] << ","; // G src
            std::cout << outputData[(i * w + j) * 3 + 1] << " ";  // G tgt
            std::cout << (int)img.data[(i * w + j) * 3 + 2] << ",";  // R src
            std::cout << outputData[(i * w + j) * 3 + 2] << std::endl;  // R tgt
        }
    }

    cudaFree(tgtDevData);
    cudaFree(srcDevData);
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("This program need 1 argument\n");
        printf("Usage: ./normal [image path]\n");
        printf("Example: ./normal lena.jpg\n");
        return 1;
    }

    std::string imagePath(argv[1]);
    normalize(imagePath);

    return 0;
}
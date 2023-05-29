#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


__global__ void RGB(const uchar* srcData, uchar* tgtData, const int h, const int w)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * w;
    int idx3 = idx * 3;
    if (ix < w && iy < h)
    {
        tgtData[idx3] = srcData[idx3 + 2];
        tgtData[idx3 + 1] = srcData[idx3 + 1];
        tgtData[idx3 + 2] = srcData[idx3];
    }
}


void toRGB(const cv::Mat& srcImg, cv::Mat& dstImg)
{
    int w = srcImg.cols;
    int h = srcImg.rows;
    printf("Image width is %d, height is %d\n", w, h);
    int wh = w * h;
    int elements = wh * 3;

    // rgb img on device
    uchar* dstDevData;
    cudaMalloc((void**)&dstDevData, sizeof(uchar) * elements);
    // bgr img on device
    uchar* srcDevData;
    cudaMalloc((void**)&srcDevData, sizeof(uchar) * elements);
    double gtct_time = (double)cv::getTickCount();
    cudaMemcpy(srcDevData, srcImg.data, sizeof(uchar) * elements, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    printf("Block(%d, %d),Grid(%d, %d).\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    RGB<<<gridSize, blockSize>>>(srcDevData, dstDevData, h, w);

    cudaMemcpy(dstImg.data, dstDevData, sizeof(uchar) * elements, cudaMemcpyDeviceToHost);
    printf("=>need time:%.2f ms\n", ((double)cv::getTickCount() - gtct_time) / ((double)cv::getTickFrequency()) * 1000);

    cudaFree(srcDevData);
    cudaFree(dstDevData);
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("This program need 1 argument\n");
        printf("Usage: ./color2gray [image path]\n");
        printf("Example: ./color2gray lena.jpg\n");
        return 1;
    }
    // read source image
    std::string imagePath(argv[1]);
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    // target image
    cv::Mat outputImg(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    toRGB(img, outputImg);

    cv::imwrite("rgb.jpg", outputImg);

    return 0;
}

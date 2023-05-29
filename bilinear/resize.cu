/*
    Define cuda bilinear function by self
*/
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


__global__ void linear(const uchar* srcData, const int srcH, const int srcW, uchar* tgtData, const int tgtH, const int tgtW)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * tgtW;
    int idx3 = idx * 3;

    float scaleY = (float)tgtH / (float)srcH;
    float scaleX = (float)tgtW / (float)srcW;

    // (ix,iy)为目标图像坐标
    // (before_x,before_y)原图坐标
    float beforeX = float(ix + 0.5) / scaleX - 0.5;
    float beforeY = float(iy + 0.5) / scaleY - 0.5;
    // 原图像坐标四个相邻点
    // 获得变换前最近的四个顶点,取整
    int topY = static_cast<int>(beforeY);
    int bottomY = topY + 1;
    int leftX = static_cast<int>(beforeX);
    int rightX = leftX + 1;
    //计算变换前坐标的小数部分
    float u = beforeX - leftX;
    float v = beforeY - topY;

    if (ix < tgtW && iy < tgtH)
    {
        // 如果计算的原始图像的像素大于真实原始图像尺寸
        if (topY >= srcH - 1 && leftX >= srcW - 1)  //右下角
        {
            for (int k = 0; k < 3; k++)
            {
                tgtData[idx3 + k] = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k];
            }
        }
        else if (topY >= srcH - 1)  // 最后一行
        {
            for (int k = 0; k < 3; k++)
            {
                tgtData[idx3 + k]
                = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
                + (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k];
            }
        }
        else if (leftX >= srcW - 1)  // 最后一列
        {
            for (int k = 0; k < 3; k++)
            {
                tgtData[idx3 + k]
                = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
                + (1. - u) * (v) * srcData[(leftX + bottomY * srcW) * 3 + k];
            }
        }
        else  // 非最后一行或最后一列情况
        {
            for (int k = 0; k < 3; k++)
            {
                tgtData[idx3 + k]
                = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
                + (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k]
                + (1. - u) * (v) * srcData[(leftX + bottomY * srcW) * 3 + k]
                + u * v * srcData[(rightX + bottomY * srcW) * 3 + k];
            }
        }
    }
}


void interpolate(const cv::Mat& srcImg, cv::Mat& dstImg, const int dstHeight, const int dstWidth)
{
    int srcHeight = srcImg.rows;
    int srcWidth = srcImg.cols;
    printf("Source image width is %d, height is %d\n", srcWidth, srcHeight);
    printf("Target image width is %d, height is %d\n", dstWidth, dstHeight);
    int srcElements = srcHeight * srcWidth * 3;
    int dstElements = dstHeight * dstWidth * 3;

    // target image data on device
    uchar* dstDevData;
    cudaMalloc((void**)&dstDevData, sizeof(uchar) * dstElements);
    // source images data on device
    uchar* srcDevData;
    cudaMalloc((void**)&srcDevData, sizeof(uchar) * srcElements);
    double gtct_time = (double)cv::getTickCount();
    cudaMemcpy(srcDevData, srcImg.data, sizeof(uchar) * srcElements, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);
    printf("Block(%d, %d),Grid(%d, %d).\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    linear<<<gridSize, blockSize>>>(srcDevData, srcHeight, srcWidth, dstDevData, dstHeight, dstWidth);

    cudaMemcpy(dstImg.data, dstDevData, sizeof(uchar) * dstElements, cudaMemcpyDeviceToHost);
    printf("=>need time:%.2f ms\n", ((double)cv::getTickCount() - gtct_time) / ((double)cv::getTickFrequency()) * 1000);

    cudaFree(srcDevData);
    cudaFree(dstDevData);
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("This program need 1 argument\n");
        printf("Usage: ./resize [image path]\n");
        printf("Example: ./resize lena.jpg\n");
        return 1;
    }
    // read source image
    std::string imagePath(argv[1]);
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

    int outputHeight = 768;
    int outputWidth = 768;
    cv::Mat outputImg(outputHeight, outputWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    interpolate(img, outputImg, outputHeight, outputWidth);    

    cv::imwrite("resized.jpg", outputImg);

    return 0;
}

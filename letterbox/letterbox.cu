#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


__global__ void letter(const uchar* srcData, const int srcH, const int srcW, uchar* tgtData, 
    const int tgtH, const int tgtW, const int rszH, const int rszW, const int startY, const int startX)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * tgtW;
    int idx3 = idx * 3;

    if ( ix > tgtW || iy > tgtH ) return;  // thread out of target range
    // gray region on target image
    if ( iy < startY || iy > (startY + rszH - 1) ) {
        tgtData[idx3] = 128;
        tgtData[idx3 + 1] = 128;
        tgtData[idx3 + 2] = 128;
        return;
    }
    if ( ix < startX || ix > (startX + rszW - 1) ){
        tgtData[idx3] = 128;
        tgtData[idx3 + 1] = 128;
        tgtData[idx3 + 2] = 128;
        return;
    }

    float scaleY = (float)rszH / (float)srcH;
    float scaleX = (float)rszW / (float)srcW;

    // (ix,iy)为目标图像坐标
    // (before_x,before_y)原图坐标
    float beforeX = float(ix - startX + 0.5) / scaleX - 0.5;
    float beforeY = float(iy - startY + 0.5) / scaleY - 0.5;
    // 原图像坐标四个相邻点
    // 获得变换前最近的四个顶点,取整
    int topY = static_cast<int>(beforeY);
    int bottomY = topY + 1;
    int leftX = static_cast<int>(beforeX);
    int rightX = leftX + 1;
    //计算变换前坐标的小数部分
    float u = beforeX - leftX;
    float v = beforeY - topY;

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

void letterbox(const cv::Mat& srcImg, cv::Mat& dstImg, const int dstHeight, const int dstWidth)
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

    // calculate width and height after resize
    int w, h, x, y;
    float r_w = dstWidth / (srcWidth * 1.0);
    float r_h = dstHeight / (srcHeight * 1.0);
    if (r_h > r_w) {
        w = dstWidth;
        h = r_w * srcHeight;
        x = 0;
        y = (dstHeight - h) / 2;
    }
    else {
        w = r_h * srcWidth;
        h = dstHeight;
        x = (dstWidth - w) / 2;
        y = 0;
    }

    dim3 blockSize(32, 32);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);
    printf("Block(%d, %d),Grid(%d, %d).\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    letter<<<gridSize, blockSize>>>(srcDevData, srcHeight, srcWidth, dstDevData, dstHeight, dstWidth, h, w, y, x);

    cudaMemcpy(dstImg.data, dstDevData, sizeof(uchar) * dstElements, cudaMemcpyDeviceToHost);
    printf("=>need time:%.2f ms\n", ((double)cv::getTickCount() - gtct_time) / ((double)cv::getTickFrequency()) * 1000);

    cudaFree(srcDevData);
    cudaFree(dstDevData);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("This program need 1 argument\n");
        printf("Usage: ./letterbox [image path]\n");
        printf("Example: ./letterbox lena.jpg\n");
        return 1;
    }

    // read source image
    std::string imagePath(argv[1]);
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

    int outputHeight = 640;
    int outputWidth = 640;
    cv::Mat outputImg(outputHeight, outputWidth, CV_8UC3);

    letterbox(img, outputImg, outputHeight, outputWidth);

    cv::imwrite("letter.jpg", outputImg);

    return 0;
}

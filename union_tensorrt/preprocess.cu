#include "preprocess.h"


__global__ void resize(const uchar* srcData, const int srcH, const int srcW, uchar* tgtData, const int tgtH, const int tgtW)
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


__global__ void process(const uchar* srcData, float* tgtData, const int h, const int w)
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
        tgtData[idx] = ((float)srcData[idx3 + 2] / 255.0 - 0.485) / 0.229;  // R pixel
        tgtData[idx + h * w] = ((float)srcData[idx3 + 1] / 255.0 - 0.456) / 0.224;  // G pixel
        tgtData[idx + h * w * 2] = ((float)srcData[idx3] / 255.0 - 0.406) / 0.225;  // B pixel
    }
}


void preprocess(const cv::Mat& srcImg, float* dstData, const int dstHeight, const int dstWidth)
{
    int srcHeight = srcImg.rows;
    int srcWidth = srcImg.cols;
    int srcElements = srcHeight * srcWidth * 3;
    int dstElements = dstHeight * dstWidth * 3;

    // target data on device
    float* dstDevData;
    cudaMalloc((void**)&dstDevData, sizeof(float) * dstElements);
    // middle image data on device ( for bilinear resize )
    uchar* midDevData;
    cudaMalloc((void**)&midDevData, sizeof(uchar) * dstElements);
    // source images data on device
    uchar* srcDevData;
    cudaMalloc((void**)&srcDevData, sizeof(uchar) * srcElements);
    cudaMemcpy(srcDevData, srcImg.data, sizeof(uchar) * srcElements, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);

    // bilinear resize
    resize<<<gridSize, blockSize>>>(srcDevData, srcHeight, srcWidth, midDevData, dstHeight, dstWidth);
    cudaDeviceSynchronize();
    // hwc to chw / bgr to rgb / normalize
    process<<<gridSize, blockSize>>>(midDevData, dstDevData, dstHeight, dstWidth);

    cudaMemcpy(dstData, dstDevData, sizeof(float) * dstElements, cudaMemcpyDeviceToHost);

    cudaFree(srcDevData);
    cudaFree(midDevData);
    cudaFree(dstDevData);
}

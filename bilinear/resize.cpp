/*
    Define cpp bilinear function by self
*/
#include <iostream>
#include <opencv2/opencv.hpp>


// src_img 缩放前图像
// dst_img 缩放后图像
void interpolate(const cv::Mat& srcImg, cv::Mat& dstImg, const double scaleX, const double scaleY)
{
    int srcHeight = srcImg.rows;
    int srcWidth = srcImg.cols;
    int dstHeight = dstImg.rows;
    int dstWidth = dstImg.cols;

    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;

    for (int i = 0; i < dstHeight; i++)
    {
        for (int j = 0; j < dstWidth; j++)
        {
            // (j,i)为目标图像坐标
            // (before_x,before_y)原图坐标
            double beforeX = double(j + 0.5) / scaleX - 0.5;
            double beforeY = double(i + 0.5) / scaleY - 0.5;
            // 原图像坐标四个相邻点
            // 获得变换前最近的四个顶点,取整
            int topY = static_cast<int>(beforeY);
            int bottomY = topY + 1;
            int leftX = static_cast<int>(beforeX);
            int rightX = leftX + 1;

            //计算变换前坐标的小数部分
            double u = beforeX - leftX;
            double v = beforeY - topY;

            // 如果计算的原始图像的像素大于真实原始图像尺寸
            if (topY >= srcHeight - 1 && leftX >= srcWidth - 1)  //右下角
            {
                for (int k = 0; k < 3; k++)
                {
                    dstData[(j + i * dstWidth) * 3 + k] = (1. - u) * (1. - v) * srcData[(leftX + topY * srcWidth) * 3 + k];
                }
            }
            else if (topY >= srcHeight - 1)  // 最后一行
            {
                for (int k = 0; k < 3; k++)
                {
                    dstData[(j + i * dstWidth) * 3 + k]
                    = (1. - u) * (1. - v) * srcData[(leftX + topY * srcWidth) * 3 + k]
                    + (u) * (1. - v) * srcData[(rightX + topY * srcWidth) * 3 + k];
                }
            }
            else if (leftX >= srcWidth - 1)  // 最后一列
            {
                for (int k = 0; k < 3; k++)
                {
                    dstData[(j + i * dstWidth) * 3 + k]
                    = (1. - u) * (1. - v) * srcData[(leftX + topY * srcWidth) * 3 + k]
                    + (1. - u) * (v) * srcData[(leftX + bottomY * srcWidth) * 3 + k];
                }
            }
            else  // 非最后一行或最后一列情况
            {
                for (int k = 0; k < 3; k++)
                {
                    dstData[(j + i * dstWidth) * 3 + k]
                    = (1. - u) * (1. - v) * srcData[(leftX + topY * srcWidth) * 3 + k]
                    + (u) * (1. - v) * srcData[(rightX + topY * srcWidth) * 3 + k]
                    + (1. - u) * (v) * srcData[(leftX + bottomY * srcWidth) * 3 + k]
                    + u * v * srcData[(rightX + bottomY * srcWidth) * 3 + k];
                }
            }
        }
    }
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("This program need 1 argument\n");
        printf("Usage: ./resize [image path]\n");
        printf("Example: ./resize lena.jpg\n");
        return 1;
    }

    std::string imagePath(argv[1]);
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

    double sx = 3;
    double sy = 3;

    int tgtHeight = img.rows * sy;
    int tgtWidth = img.cols * sx;
    cv::Mat tgtImg(tgtHeight, tgtWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    double gtct_time = (double)cv::getTickCount();
    interpolate(img, tgtImg, sx, sy);
    printf("=>need time:%.2f ms\n", ((double)cv::getTickCount() - gtct_time) / ((double)cv::getTickFrequency()) * 1000);

    cv::imwrite("resized.jpg", tgtImg);

    return 0;
}
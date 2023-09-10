# Cuda编程加速图像预处理

## 项目简介

- 基于 `cuda` 和 `opencv` 环境

- **目标：**
  - 单独使用，以加速图像处理操作；
  - **结合 TensorRT 使用，进一步加快推理速度**

## 加速效果

- 这里对比 `Deeplabv3+ ` 使用  `cuda` 预处理前后的 tensorrt 推理速度
- 未使用cuda图像预处理的代码，可参考作者的另一个  [tensorrt](https://github.com/emptysoal/tensorrt-experiment)  的项目：

FP32精度:

| C++图像预处理 | CUDA图像预处理 |
| :-----------: | :------------: |
|     25 ms     |     19 ms      |

Int8量化后:

| C++图像预处理 | CUDA图像预处理 |
| :-----------: | :------------: |
|     10 ms     |    **3 ms**    |

## 文件说明

```bash
project dir
	├── bgr2rgb  # 实现BGR转RGB的cuda加速
    |   ├── Makefile
    |   └── bgr2rgb.cu
    ├── bilinear  # 实现双线性插值的cuda加速
    |   ├── Makefile
    |   └── resize.cu
    ├── hwc2chw  # 实现相当于transpose((2, 0, 1))的cuda加速
    |   ├── Makefile
    |   └── transpose.cu
    ├── normalize  # 实现归一化的cuda加速
    |   ├── Makefile
    |   └── normal.cu
    ├── preprocess  # 汇总以上的图像处理（不是简单的拼接），实现常用的图像预处理，之后输入到网络当中
    |   ├── Makefile
    |   └── preprocess.cu
    ├── union_tensorrt  # 将上述的图像预处理，结合TensorRT一起使用，对比推理加速效果
    |   ├── Makefile
    |   ├── preprocess.cu
    |   ├── preprocess.h
    |   └── trt_infer.cpp  # 用于模型推理
    └── lena.jpg  # 用于测试的图片
```

## 使用说明

### 图像加速单一操作：

- 对于目录：bgr2rgb、bilinear、hwc2chw、normalize，实现单一功能上的图像操作加速
- 使用测试：

```bash
cd <dir name>
make
./<bin file> <image path>

example:
cd bgr2rgb
make
./bgr2rgb ../lena.jpg
```

备注：如果 cuda 或 opencv 安装目录与 Makefile 中的不同，记得切换成自己的

### 常规图像预处理

- 在推理之前，图像通常需经过 Resize、BGR to RGB、HWC to CHW、Normalize
- 使用测试：

```bash
cd preprocess
make
./preprocess ../lena.jpg  # 即可对图像完成上述全部操作
```

### 结合 TensorRT 使用

使用方式：

1）根据作者另一个 [tensorrt](https://github.com/emptysoal/tensorrt-experiment) 的项目，构建好环境，下载分割数据集，并训练Deeplabv3+网络

2）进入到目录：`Deeplabv3+/TensorRT/C++/api_model/`

3）将本项目的`union_tensorrt`目录下的文件放入上述目录中（或替换原文件）

4）依次执行以下命令来使用TensorRT推理

```bash
python pth2wts.py
make
./trt_infer
```

5）得到以下结果，说明运行成功，同目录下会生成分割结果图像

```bash
Loading weights: ./para.wts
Succeeded building backbone!
Succeeded building aspp!
Succeeded building decoder!
Succeeded building total network!
Succeeded building serialized engine!
Succeeded building engine!
Succeeded saving .plan file!
Total image num is: 8 inference total cost is: 105ms average cost is: 19ms
```


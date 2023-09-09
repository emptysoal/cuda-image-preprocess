# Cuda programming speed up image preprocessing
[READMA中文版](https://github.com/emptysoal/cuda-image-preprocess/blob/main/README-chinese.md)

## Introduction

- Based on `cuda` and `opencv` 

- Target:
  - Can be used alone to speed up image processing operations;
  - Combined with the use of TensorRT, the inferencing speed is further accelerated.

## Speed

- Here we compare the tensorrt inference speed before and after `Deeplabv3+` preprocessing with `cuda`
- Not using cuda code of image preprocessing, refer to my another [tensorrt](https://github.com/emptysoal/tensorrt-experiment)  project

FP32:

| C++ image preproce | cuda image preprocess |
| :----------------: | :-------------------: |
|       25 ms        |         19 ms         |

Int8 quantization:

| C++ image preproce | cuda image preprocess |
| :----------------: | :-------------------: |
|       10 ms        |       **3 ms**        |

## File description

```bash
project dir
	├── bgr2rgb  # cuda code achieve BGR to RGB 
    |   ├── Makefile
    |   └── bgr2rgb.cu
    ├── bilinear  # cuda code achieve bilinear resize
    |   ├── Makefile
    |   └── resize.cu
    ├── hwc2chw  # cuda code achieve shape from HWC to CHW, such as np.transpose((2, 0, 1))
    |   ├── Makefile
    |   └── transpose.cu
    ├── normalize  # cuda code achieve image data normalization
    |   ├── Makefile
    |   └── normal.cu
    ├── preprocess  # unite the above(not simple stitching), achieve common image preprocessing
    |   ├── Makefile
    |   └── preprocess.cu
    ├── union_tensorrt  # An example for uniting TensorRT, speed up Deeplabv3+ inferencing
    |   ├── Makefile
    |   ├── preprocess.cu
    |   ├── preprocess.h
    |   └── trt_infer.cpp
    └── lena.jpg  # Pictures for testing
```

## Usages

### A single operation to speed up image processing

- For directories: bgr2rgb、bilinear、hwc2chw、normalize

```bash
cd <dir name>
make
./<bin file> <image path>

# For example:
cd bgr2rgb
make
./bgr2rgb ../lena.jpg
# Then you can see the result of the image lena.jpg after the exchange of R channel and B channel, and save it in the current directory 
```

Note: If the cuda or opencv installation directory is different from the one in the Makefile, remember to switch to your own 

### General image preprocessing

- Before model inference，images usually need to be Resize, BGR to RGB, HWC to CHW, and Normalize
- You can implement this process using the following steps:

```bash
cd preprocess
make
./preprocess ../lena.jpg
```

### Used in combination with TensorRT

Method：

1）According to my another [tensorrt](https://github.com/emptysoal/tensorrt-experiment) project, building environment, download datasets, and training Deeplabv3+ network 

2）Enter into directory: `Deeplabv3+/TensorRT/C++/api_model/`

3）Place the files which in this project `union_tensorrt` directory into the above directory (or replace the original file) 

4）Execute the following commands in sequence to use TensorRT inference

```bash
python pth2wts.py
make
./trt_infer
```

5）The following results indicate that the operation is successful, and the segmentation result image will be generated in the same directory 

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


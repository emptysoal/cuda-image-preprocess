CUDA_PATH       = /usr/local/cuda
OPENCV_INC_PATH = /usr/local/include/opencv4
OPENCV_LIB_PATH = /usr/local/lib
INCLUDE         = -I$(CUDA_PATH)/include -I$(OPENCV_INC_PATH)
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart
LDFLAG         += -L$(OPENCV_LIB_PATH) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

CC = nvcc

all: letterbox

letterbox: letterbox.cu
	$(CC) -std=c++11 letterbox.cu -o letterbox $(INCLUDE) $(LDFLAG)

clean:
	rm -rf ./letterbox

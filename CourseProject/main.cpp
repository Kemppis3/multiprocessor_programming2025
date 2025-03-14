
//TODO: 
//Fix gaussian filter function (creates colorful horizontal lines in output image)
//Add timing tracking for OpenCL implementation and compare speed.


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include "lodepng.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "ImageData.hpp"
#define MATRIXSIZE 100

double opTime;

static void display()
{

	cl_platform_id platform;
	cl_int status = clGetPlatformIDs(1, &platform, NULL);

	if (status != CL_SUCCESS)
	{
		printf("Error: Failed to get platform information!");
		return;
	}

	char buffer[1024];
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
	printf("Platform name: %s\n", buffer);

	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
	printf("Platform version: %s\n", buffer);
}

void matrix_add(float *matrix_1, float *matrix_2, float *result, int matrixsize)
{
	for (int i = 0; i < matrixsize * matrixsize; i++)
	{
		result[i] = matrix_1[i] + matrix_2[i];
	}
}

double getTime()
{
	LARGE_INTEGER frequency, counter;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&counter);
	return (double)counter.QuadPart / (double)frequency.QuadPart;
}


std::string loadKernelFromFile(std::string filename) {

	std::ifstream file(filename);

	if(!file.is_open()) {
		std::cerr << "Opening kernel file failed";
		return "";
	} else {	
		std::stringstream buffer;
		buffer << file.rdbuf();
		return buffer.str();
	}

}


int main() {

	ImageData inputImage("im0.png");
	unsigned width = inputImage.getWidth();
	unsigned height = inputImage.getHeight();

	unsigned char* inptr = inputImage.ImageDataToCharPointer();

	unsigned newWidth = width/4;
	unsigned newHeight = height/4;
	unsigned char* outputImage = new unsigned char[newWidth*newHeight*4];


    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create buffers
    cl_mem buffer_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*4, NULL, NULL);
	cl_mem buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, newWidth*newHeight*4, NULL, NULL);

    // Copy data to the device
    clEnqueueWriteBuffer(queue, buffer_in, CL_TRUE, 0, width*height*4, inptr, 0, NULL, NULL);

    // Load and compile the kernel
	std::string sourceString = loadKernelFromFile("kernel.cl");
    const char *source = sourceString.c_str();
	
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "ProcessImage", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_in);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_out);
    clSetKernelArg(kernel, 2, sizeof(unsigned), &width);
    clSetKernelArg(kernel, 3, sizeof(unsigned), &height);

    // Execute the kernel
	size_t global_work_size[2] = {newWidth, newHeight};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Read results back to the host
    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, newWidth*newHeight*4, outputImage, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(buffer_in);
	clReleaseMemObject(buffer_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);


	ImageData out(outputImage, newWidth, newHeight);
	out.WriteImageToFile("im0_bw.png");
    return 0;
}
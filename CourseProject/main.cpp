
//TODO: 
//Fix gaussian filter function (creates colorful horizontal lines in output image)
//Add timing tracking for OpenCL implementation and compare speed.


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include "lodepng.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "ImageData.hpp"
#define MATRIXSIZE 100

std::chrono::duration<double> opTime;

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

std::vector<unsigned char> CalcZNCC(std::vector<unsigned char> image1, std::vector<unsigned char> image2, unsigned int imageWidth, unsigned int imageHeight, int windowWidth, int windowHeight, unsigned int max_disp) {

	//Mean
	float mean1; 
	float mean2;

	//Standard deviation
	float std1; 
	float std2;

	unsigned int index1;
	unsigned int index2;

	//Current disparity ZNCC value and best ZNCC value
	float current_val;
	float best_val;

	//Best disparity
	float best_d;

	//Border calculation
	int validPixels;

	std::vector<unsigned char> output;
	output.resize(image1.size());

	for(int y = 0; y<imageHeight; y++) {

		for(int x = 0; x<imageWidth; x++) {

			best_d = max_disp;
			best_val = -1;

			for(int d = 0; d < max_disp; d++) {

				mean1 = 0.0;
				mean2 = 0.0;
				validPixels = 0;

				for (int ky = 0; ky < windowHeight; ky++) {
					for (int kx = 0; kx < windowWidth; kx++) {

						if (y+ky >= imageHeight || x+kx >= imageWidth || x+kx-d < 0) {
							continue;
						}

						index1 = ((y+ky)*imageWidth+(x+kx))*4;
						index2 = ((y+ky)*imageWidth+(x+kx-d))*4;

						mean1 += image1[index1];
						mean2 += image2[index2];
						validPixels++;

					}
				}

				if(validPixels > 0) {
					mean1 /= validPixels;
					mean2 /= validPixels;
				}

				std1 = 0.0;
				std2 = 0.0;
				current_val = 0.0;

				for(int ky = 0; ky < windowHeight; ky++) {
					for(int kx = 0; kx < windowWidth; kx++) {

						if (y+ky >= imageHeight || x+kx >= imageWidth || x+kx-d < 0) {
							continue;
						}

						index1 = ((y+ky)*imageWidth+(x+kx))*4;
						index2 = ((y+ky)*imageWidth+(x+kx-d))*4;

						std1 += (image1[index1] - mean1) * (image1[index1] - mean1);
						std2 += (image2[index2] - mean2) * (image2[index2] - mean2);
						current_val += (image1[index1] - mean1) * (image2[index2] - mean2);
						
					}
				}

				if(std1 > 0 && std2 > 0) {
					current_val /= std::sqrt(std1)*std::sqrt(std2);
				}
				

				if(current_val > best_val) {
					best_val = current_val;
					best_d = d;
				}
			}

			index1 = (y*imageWidth+x)*4;

			best_d = std::round((best_d/max_disp)*255.0);

			output[index1] = best_d;
			output[index1+1] = best_d;
			output[index1+2] = best_d;
			output[index1+3] = 255;
		}
	}

	return output;

}


std::vector<unsigned char> CrossCheck(std::vector<unsigned char> map1, std::vector<unsigned char> map2, unsigned int threshold) {

	std::vector<unsigned char> outputMap;
	outputMap.resize(map1.size());

	for(int i = 0; i < map1.size(); i++) {

		if(std::abs(map1[i] - map2[i]) > threshold) {
			outputMap[i] = 0;
		} else  {
			outputMap[i] = map1[i];
		}
	}
 
	return outputMap;
}


/* std::vector<unsigned char> OcculsionFill(std::vector<unsigned char> map) {

	for(int i = 0; i < map.size(); i++) {

		if(map[i] == 0){

		} 
	}

}
 */
int main() {

	ImageData image1("im0.png");
	ImageData image2("im1.png");

	image1.ResizeImage();
	image2.ResizeImage();

	image1.ImageToGrayscale();
	image2.ImageToGrayscale();

	std::vector<unsigned char> map1 = CalcZNCC(image1.getImage(), image2.getImage(), image1.getWidth(), image1.getHeight(), 9, 9, 260);
	//std::vector<unsigned char> map2 = CalcZNCC(image2.getImage(), image1.getImage(), image1.getWidth(), image1.getHeight(), 9, 9, 260);

	//std::vector<unsigned char> mapOut = CrossCheck(map1, map2, 8);

	ImageData outImage(map1, image1.getWidth(), image1.getHeight());
	outImage.WriteImageToFile("output.png");

	return 0;
}


/*
int main() {


	auto start = std::chrono::high_resolution_clock::now();

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

	auto end = std::chrono::high_resolution_clock::now();
	opTime = end-start;

	std::cout << "Execution time: " << opTime.count() << "s" << std::endl;

    return 0;
}
*/
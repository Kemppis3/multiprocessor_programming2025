
//Todo: 
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include "lodepng.h"
#include <iostream>
#include <vector>
#define MATRIXSIZE 100


static void display() {

	cl_platform_id platform;
	cl_int status = clGetPlatformIDs(1, &platform, NULL);

	if (status != CL_SUCCESS) {
		printf("Error: Failed to get platform information!");
		return;
	}

	char buffer[1024];
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
	printf("Platform name: %s\n", buffer);

	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
	printf("Platform version: %s\n", buffer);
}

void matrix_add(float *matrix_1, float *matrix_2, float *result, int matrixsize) {
	for (int i = 0; i < matrixsize * matrixsize; i++) {
		result[i] = matrix_1[i] + matrix_2[i];
	}
}

double getTime() {
	LARGE_INTEGER frequency, counter;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&counter);
	return (double)counter.QuadPart / (double)frequency.QuadPart;
}

unsigned int processingImage(const char* filename, const char* outputName) {

	std::vector<unsigned char> image;
	unsigned width, height;

	unsigned error = lodepng::decode(image, width, height, filename);

	if (error) {
		std::cout << "decoder error" << lodepng_error_text(error) << std::endl;
		return error;
	}

	std::cout << "Original Size: " << width << "x" << height << std::endl;
	unsigned newWidth = width / 4;
	unsigned newHeight = height / 4;
	std::vector<unsigned char> resizedImage(newWidth * newHeight * 4);

	for (unsigned y = 0; y < newHeight; ++y) {
		for (unsigned x = 0; x < newWidth; ++x) {
			unsigned originalX = x * 4;
			unsigned originalY = y * 4;
			unsigned originalIndex = (originalY * width + originalX) * 4;
			unsigned newIndex = (y * newWidth + x) * 4;

			resizedImage[newIndex] = image[originalIndex];
			resizedImage[newIndex + 1] = image[originalIndex + 1];
			resizedImage[newIndex + 2] = image[originalIndex + 2];
			resizedImage[newIndex + 3] = image[originalIndex + 3];
		}
	}

	error = lodepng::encode(outputName, resizedImage, newWidth, newHeight);
	if (error) {
		std::cout << "Error saving image" << lodepng_error_text(error) << std::endl;
		return error;
	}
	std::cout << "Resized image saved to " << outputName << " (" << newWidth << "x" << newHeight << ")" << std::endl;
	return 0;
}

unsigned int grayScale(const char* original, const char* resized) {

}

int main() {

	processingImage("test.png", "result.png");
	
	return 0;
}



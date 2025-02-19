
//Todo: Maybe define separate file for functions


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include "lodepng.h"
#include <iostream>
#include <vector>
#define MATRIXSIZE 100

struct ImageData {
    std::vector<unsigned char> image; 
    unsigned width;                   
    unsigned height;                 
};

double opTime;

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



ImageData ReadImage(const char* filename) {

	double startTime = getTime();

	std::vector<unsigned char> image;
	unsigned width, height;

	unsigned error = lodepng::decode(image, width, height, filename);
	
	if (error) {
		std::cout << "decoder error" << lodepng_error_text(error) << std::endl;
	} else {
		double endTime = getTime();
		std::cout << "Image read from file! Image read time = " << endTime-startTime << "s" << std::endl; 
		opTime += endTime-startTime;
		return {image, width, height};
	}
}

ImageData ResizeImage(ImageData image, unsigned int scalefactor) {

	double startTime = getTime();
	unsigned newWidth = image.width / scalefactor;
	unsigned newHeight = image.height / scalefactor;
	std::vector<unsigned char> resizedImage(newWidth * newHeight * 4);

	for (unsigned y = 0; y < newHeight; ++y) {
		for (unsigned x = 0; x < newWidth; ++x) {
			unsigned originalX = x * 4;
			unsigned originalY = y * 4;
			unsigned originalIndex = (originalY * image.width + originalX) * 4;
			unsigned newIndex = (y * newWidth + x) * 4;

			resizedImage[newIndex] = image.image[originalIndex];
			resizedImage[newIndex + 1] = image.image[originalIndex + 1];
			resizedImage[newIndex + 2] = image.image[originalIndex + 2];
			resizedImage[newIndex + 3] = image.image[originalIndex + 3];
		}
	}

	double endTime = getTime();
	std::cout << "Image rescaled to size: " << "(" << newWidth << "x" << newHeight << ") " << "Image rescaling time = " << endTime-startTime << "s" << std::endl;
	opTime += endTime-startTime;
	return {resizedImage, newWidth, newHeight};
}

int WriteImage(ImageData image, const char* outputFileName) {

	double startTime = getTime();
	unsigned error;
	error = lodepng::encode(outputFileName, image.image, image.width, image.height);

	if (error) {
		std::cout << "Error saving image" << lodepng_error_text(error) << std::endl;
		return error;
	}
	double endTime = getTime();
	std::cout << "Image saved to " << outputFileName << " (" << image.width << "x" << image.height << ") " << "Save time = " << endTime-startTime << "s"<< std::endl;
	opTime += endTime-startTime;
	return 0;
}


ImageData GrayScaleImage(ImageData image) {

	double startTime = getTime();
	std::vector<unsigned char> greyImage = image.image;

	for(unsigned i = 0; i < image.width*image.height; ++i) {
		
		unsigned char r = image.image[i * 4];
        unsigned char g = image.image[i * 4 + 1];
        unsigned char b = image.image[i * 4 + 2]; 

        unsigned char gray = static_cast<unsigned char>(0.2126 * r + 0.7152 * g + 0.0722 * b);

        greyImage[i * 4] = gray;
        greyImage[i * 4 + 1] = gray;
        greyImage[i * 4 + 2] = gray;

	}

	double endTime = getTime();
	std::cout << "Image greyscaled! Greyscaling time = " << endTime-startTime << "s" << std::endl;
	opTime += endTime-startTime;	
	return {greyImage, image.width, image.height};
}

ImageData ApplyFilter(ImageData image) {

	double startTime = getTime();
	std::vector<unsigned char>filteredImage = image.image;

	//Gaussian blur kernel
	const float gaussianKernel[5][5] = {
		{1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f,  1/256.0f},
		{4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f,  4/256.0f},
		{6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f,  6/256.0f},
		{4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f,  4/256.0f},
		{1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f,  1/256.0f}
	};

	
	for (unsigned y = 2; y < image.height - 2; y++) {
        for (unsigned x = 2; x < image.width - 2; x++) {
            
			float sumR = 0, sumG = 0, sumB = 0;

            // Apply 5x5 kernel
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    unsigned pixelIndex = ((y + ky) * image.width + (x + kx)) * 4;

                    sumR += filteredImage[pixelIndex] * gaussianKernel[ky + 2][kx + 2];
                    sumG += filteredImage[pixelIndex + 1] * gaussianKernel[ky + 2][kx + 2];
                    sumB += filteredImage[pixelIndex + 2] * gaussianKernel[ky + 2][kx + 2];
                }
            }

            unsigned pixelIndex = (y * image.width + x) * 4;
            filteredImage[pixelIndex] = static_cast<unsigned char>(sumR);
            filteredImage[pixelIndex + 1] = static_cast<unsigned char>(sumG);
        	filteredImage[pixelIndex + 2] = static_cast<unsigned char>(sumB);
        }
	}

	double endTime = getTime();
	std::cout << "Image filtered! Filtering time = " << endTime-startTime << "s" << std::endl;
	opTime += endTime-startTime;
	return {filteredImage, image.width, image.height};
}




int main() {

	ImageData image = ReadImage("im0.png");
	ImageData resized_image = ResizeImage(image, 4);
	ImageData grayscaled_image = GrayScaleImage(resized_image);
	ImageData filtered_image = ApplyFilter(grayscaled_image);
	WriteImage(resized_image, "image_resized.png");
	WriteImage(grayscaled_image, "image_grayscaled.png");
	WriteImage(filtered_image, "image_filtered.png");
	
	std::cout << "Total time = " << opTime << "s" << std::endl;
	
	return 0;
}



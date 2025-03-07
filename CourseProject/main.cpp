
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
#define MATRIXSIZE 100


using namespace std;

struct ImageData
{
	std::vector<unsigned char> image;
	unsigned width;
	unsigned height;
};

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

ImageData ReadImage(string filename)
{

	double startTime = getTime();

	std::vector<unsigned char> image;
	unsigned width, height;

	unsigned error = lodepng::decode(image, width, height, filename);

	if (error)
	{
		cout << "decoder error" << lodepng_error_text(error) << endl;
	}
	else
	{
		double endTime = getTime();
		cout << "Image read from file! Image read time = " << endTime - startTime << "s" << endl;
		opTime += endTime - startTime;
		return {image, width, height};
	}
}

ImageData ResizeImage(ImageData image, unsigned int scalefactor)
{

	double startTime = getTime();
	unsigned newWidth = image.width / scalefactor;
	unsigned newHeight = image.height / scalefactor;
	std::vector<unsigned char> resizedImage(newWidth * newHeight * 4);

	for (unsigned y = 0; y < newHeight; ++y)
	{
		for (unsigned x = 0; x < newWidth; ++x)
		{
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
	std::cout << "Image rescaled to size: " << "(" << newWidth << "x" << newHeight << ") " << "Image rescaling time = " << endTime - startTime << "s" << std::endl;
	opTime += endTime - startTime;
	return {resizedImage, newWidth, newHeight};
}

int WriteImage(ImageData image, const char *outputFileName)
{

	double startTime = getTime();
	unsigned error;
	error = lodepng::encode(outputFileName, image.image, image.width, image.height);

	if (error)
	{
		std::cout << "Error saving image" << lodepng_error_text(error) << std::endl;
		return error;
	}
	double endTime = getTime();
	std::cout << "Image saved to " << outputFileName << " (" << image.width << "x" << image.height << ") " << "Save time = " << endTime - startTime << "s" << std::endl;
	opTime += endTime - startTime;
	return 0;
}

ImageData GrayScaleImage(ImageData image)
{

	double startTime = getTime();
	std::vector<unsigned char> greyImage = image.image;

	for (unsigned i = 0; i < image.width * image.height; ++i)
	{

		unsigned char r = image.image[i * 4];
		unsigned char g = image.image[i * 4 + 1];
		unsigned char b = image.image[i * 4 + 2];

		unsigned char gray = static_cast<unsigned char>(0.2126 * r + 0.7152 * g + 0.0722 * b);

		greyImage[i * 4] = gray;
		greyImage[i * 4 + 1] = gray;
		greyImage[i * 4 + 2] = gray;
	}

	double endTime = getTime();
	std::cout << "Image greyscaled! Greyscaling time = " << endTime - startTime << "s" << std::endl;
	opTime += endTime - startTime;
	return {greyImage, image.width, image.height};
}

ImageData ApplyFilter(ImageData image)
{

	double startTime = getTime();
	std::vector<unsigned char> filteredImage = image.image;

	// Gaussian blur kernel
	const float gaussianKernel[5][5] = {
		{1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f},
		{4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
		{6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f},
		{4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
		{1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f}};

	for (unsigned y = 2; y < image.height - 2; y++)
	{
		for (unsigned x = 2; x < image.width - 2; x++)
		{

			float sumR = 0, sumG = 0, sumB = 0;

			// Apply 5x5 kernel
			for (int ky = -2; ky <= 2; ky++)
			{
				for (int kx = -2; kx <= 2; kx++)
				{
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
	std::cout << "Image filtered! Filtering time = " << endTime - startTime << "s" << std::endl;
	opTime += endTime - startTime;
	return {filteredImage, image.width, image.height};
}

string loadKernelFromFile(string filename) {

	ifstream file(filename);

	if(!file.is_open()) {
		cerr << "Opening kernel file failed";
	} else {	
		stringstream buffer;
		buffer << file.rdbuf();
		return buffer.str();
	}

}

unsigned char* ImageDataToCharPointer(ImageData image) {

	vector<unsigned char> imdata = image.image;
	size_t size = imdata.size();
	unsigned char* imptr = new unsigned char[size];
	
	memcpy(imptr, imdata.data(), size);

	return imptr;
}

ImageData CharPointerToImageData(unsigned char* imptr, unsigned width, unsigned height) {

	unsigned size = width*height*4;
	vector<unsigned char> imdata(imptr, imptr+size);

	return {imdata, width, height};

}



int main() {

	ImageData inputImage = ReadImage("im0.png");
	unsigned width = inputImage.width;
	unsigned height = inputImage.height;

	unsigned char* inptr = ImageDataToCharPointer(inputImage);

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
	string sourceString = loadKernelFromFile("kernel.cl");
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

	ImageData out = CharPointerToImageData(outputImage, newWidth, newHeight);
	WriteImage(out, "im0_bw.png");

    return 0;
}
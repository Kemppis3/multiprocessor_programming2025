//TODO: 
//Fix gaussian filter function (creates colorful horizontal lines in output image)
//Add timing tracking for OpenCL implementation and compare speed.


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <thread>
#include <mutex>
#include <algorithm>
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
#include <omp.h>
#define MATRIXSIZE 100

// Global variable for checking and storing operation execution times.
std::chrono::duration<double> opTime;

/**
* Function for checking OpenCL status for errors and displaying detected:
* OpenCL platform,
* Device,
* Memory type,
* Memory size,
* Computing units,
* Clock frequency,
* Buffer size,
* item dimensions,
* Item sizes
*/

static void display()
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;
    cl_device_local_mem_type memType;
    cl_ulong memSize, constBufSize;
    cl_uint computeUnits, clockFreq;
    size_t maxWorkGroupSize, maxWorkItemSizes[3];
    cl_uint dims;

    cl_int status = clGetPlatformIDs(1, &platform, NULL);

    if (status != CL_SUCCESS)
    {
        printf("Error: Failed to get platform information!");
        exit(1);
    }

    char buffer[1024];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
    printf("Platform name: %s\n", buffer);

    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
    printf("Platform version: %s\n", buffer);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get device id!");
        exit(1);
    }
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(memType), &memType, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(memSize), &memSize, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq), &clockFreq, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(constBufSize), &constBufSize, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dims), &dims, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * dims, maxWorkItemSizes, NULL);

    printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n", memType == CL_LOCAL ? "CL_LOCAL" : memType == CL_GLOBAL ? "CL_GLOBAL" : "Unknown");
    printf("CL_DEVICE_LOCAL_MEM_SIZE: %llu bytes\n", memSize);
    printf("CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", computeUnits);
    printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %u MHz\n", clockFreq);
    printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %llu bytes\n", constBufSize);
    printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %zu\n", maxWorkGroupSize);
    printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: (%zu, %zu, %zu)\n",
        maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);
}
/**
* @brief Performs addition of two matrices
* @note This function was Was used in the early c++ implementation
*/

void matrix_add(float *matrix_1, float *matrix_2, float *result, int matrixsize)
{
	for (int i = 0; i < matrixsize * matrixsize; i++)
	{
		result[i] = matrix_1[i] + matrix_2[i];
	}
}
/**
* @brief Loads contents of a OpenCL kernel file into a string.
* @note This function was used in the early implementation phase.
*/

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

/**
* @brief Calculates the Zero-Mean Normalized Cross-Correlation (ZNCC) disparity map between two given images.
 * @param image1: The first input image
 * @param image2: The second input image
 * @param imageWidth: The width of the input images
 * @param imageHeight: The height of the input images
 * @param windowWidth: The width of the ZNCC window
 * @param windowHeight: The height of the ZNCC window
 * @param max_disp: The maximum disparity to search for
 * @return vector<unsinged char> output: The disparity map
* @note Used in the early implementation
*/

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

				if(validPixels == 0) continue;

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

	std::cout << "Calculated ZNCC" << std::endl;

	return output;

}

/**
* @brief Multithreaded version of the Zero-Mean Normalized Cross-Correlation function
 * @param image1: The first input image
 * @param image2: The second input image
 * @param width: The width of the input images
 * @param height: The height of the input images
 * @param win_size: The size of ZNCC window
 * @param max_disp: The maximum disparity to search for
 * @return vector<unsinged char> output: The disparity map
 * @note used in the early implementation
*/

std::vector<unsigned char> CalcZNCC_multithreaded(
    const std::vector<unsigned char>& image1,
    const std::vector<unsigned char>& image2,
    unsigned int width, unsigned int height,
    int win_size, unsigned int max_disp,
    unsigned int num_threads = std::thread::hardware_concurrency())
{
    std::vector<unsigned char> output(image1.size());
    const int half_win = win_size / 2;
    std::vector<std::thread> threads;
    std::mutex mtx; // Only needed if writing to shared resources

    // Calculate rows per thread
    const int rows_per_thread = (height + num_threads - 1) / num_threads;

    auto worker = [&](int thread_id) {
        const int start_row = thread_id * rows_per_thread;
        const int end_row = std::min<int>(start_row + rows_per_thread, height);

        for (int j = start_row; j < end_row; j++) {
            for (int i = 0; i < width; i++) {
                float best_zncc = -1.0f;
                int best_d = max_disp;

                const int y_start = std::max(0, j - half_win);
                const int y_end = std::min<int>(height - 1, j + half_win);

                for (int d = 0; d < max_disp; d++) {
                    const int x_start = std::max(0, i - half_win);
                    const int x_end = std::min<int>(width - 1, i + half_win);
                    const int x_start_right = std::max(0, x_start - d);

                    if (x_start_right > x_end) continue;

                    // First pass: means
                    float mean1 = 0.0f, mean2 = 0.0f;
                    int valid_pixels = 0;

                    for (int wy = y_start; wy <= y_end; wy++) {
                        const size_t row_offset = wy * width * 4;
                        for (int wx = x_start; wx <= x_end; wx++) {
                            if (wx - d < 0) continue;

                            mean1 += image1[row_offset + wx * 4];
                            mean2 += image2[row_offset + (wx - d) * 4];
                            valid_pixels++;
                        }
                    }

                    if (valid_pixels == 0) continue;
                    mean1 /= valid_pixels;
                    mean2 /= valid_pixels;

                    // Second pass: ZNCC components
                    float std1 = 0.0f, std2 = 0.0f, covar = 0.0f;

                    for (int wy = y_start; wy <= y_end; wy++) {
                        const size_t row_offset = wy * width * 4;
                        for (int wx = x_start; wx <= x_end; wx++) {
                            if (wx - d < 0) continue;

                            const float val1 = image1[row_offset + wx * 4] - mean1;
                            const float val2 = image2[row_offset + (wx - d) * 4] - mean2;

                            std1 += val1 * val1;
                            std2 += val2 * val2;
                            covar += val1 * val2;
                        }
                    }

                    // Compute ZNCC
                    if (std1 > 0 && std2 > 0) {
                        const float zncc = covar / (sqrtf(std1) * sqrtf(std2));
                        if (zncc > best_zncc) {
                            best_zncc = zncc;
                            best_d = d;
                            if (best_zncc > 0.95f) break;
                        }
                    }
                }

                // Write output (no mutex needed as each thread writes to different rows)
                const size_t idx = (j * width + i) * 4;
                const unsigned char disp_val = static_cast<unsigned char>((best_d * 255) / max_disp);
                output[idx] = output[idx + 1] = output[idx + 2] = disp_val;
                output[idx + 3] = 255;
            }
        }
        };

    // Launch threads
    for (unsigned t = 0; t < num_threads; t++) {
        threads.emplace_back(worker, t);
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    return output;
}

/**
* @brief Fills zero-valued pixels in the disparity map using a simple neighbourhoodd search
* The function iterates the given disparity map and if it finds a disparity of 0, 
* it searches the neighboring pixels to find a non-zero disparity to use for filling the occluded pixel.
 * @param map: The input disparity map
 * @param width: The width of the disparity map
 * @param height: The height of the disparity map
 * @return vector<unsigned char>: Disparity map with occlusion fill.
 * @note used in the early implementation
*/
std::vector<unsigned char> OcculsionFill(std::vector<unsigned char> map, unsigned int width, unsigned int height) {

	int index;
	int index2;
	std::vector<unsigned char> outputMap = map;
	bool flag;

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {

			index = (y*width+x)*4;
			
			if(map[index] == 0) {

				flag = false;

				for(int n = 1; n < 10 && !flag; n++) {

					for(int i = -n; i <= n && !flag; i++) {

						for(int j = -n; j <= n; j++) {
	
							index2 = ((y+i)*width+(x+j))*4;
	
							if(index2 > map.size() || index2 < 0 || map[index2] == 0) {
								continue;
							} else {
								outputMap[index] = map[index2];
								outputMap[index+1] = map[index2+1];
								outputMap[index+2] = map[index2+2];
								flag = true;
								break;
							}
						}
					}
				}

			}

		}
	}

	std::cout << "Occulsion fill complete" << std::endl;

	return outputMap;
}


/**
* @brief Function performs cross-checkling for two disparity maps to find inconsistant pixels.
* 
* This function compares two disparity maps and their corresponding pixels in thee maps.
* If the difference between the values of the pixels is larger than the given threshold,
* the pixel is set to 0 in the output map. 
* 
* @param map1: First disparity map (Left-to-Right)
* @param map2: Second disparity map (Right-to-Left)
* @param threshold: The maximum allowed difference between the values of the disparity values.
* @return vector<unsigned char> outputMap: Cross-checked disparity map
* @note used in the early implementation
*/


std::vector<unsigned char> CrossCheck(std::vector<unsigned char> map1, std::vector<unsigned char> map2, unsigned int threshold) {

	std::vector<unsigned char> outputMap = map1;

	#pragma omp parallel for
	for(int i = 0; i < map1.size(); i++) {

		if(std::abs(map1[i] - map2[i]) > threshold) {
			outputMap[i] = 0;
		}
	}

	std::cout << "Cross check complete" << std::endl;
 
	return outputMap;
}

/**
* @brief Multithreaded version of the OcclusionFill function
* @param map: The disparity map
* @param width: Width of the disparity map
* @param height: Height of the disparity map
* @return vector<unsigned char> outputMap: Disparity map with occlusion fill. 
*/

std::vector<unsigned char> OcclusionFillMultithreaded(std::vector<unsigned char> map, unsigned int width, unsigned int height) {

    unsigned int num_threads = omp_get_max_threads();
    std::vector<std::vector<unsigned char>> threadOutputs(num_threads, map);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& localOutput = threadOutputs[thread_id];

        #pragma omp for collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                int index = (y * width + x) * 4;

                if (map[index] == 0) {

                    bool flag = false;

                    for (int n = 1; n < 10 && !flag; n++) {
                        for (int i = -n; i <= n && !flag; i++) {
                            for (int j = -n; j <= n; j++) {

								int index2 = ((y+i)*width+(x+j))*4;

								if(index2 > map.size() || index2 < 0 || map[index2] == 0) {
									continue;
								} else {
									localOutput[index] = map[index2];
									localOutput[index+1] = map[index2+1];
									localOutput[index+2] = map[index2+2];
									flag = true;
									break;
								}
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<unsigned char> outputMap = map;

    for (size_t i = 0; i < map.size(); i++) {
        if (map[i] == 0) {
            for (unsigned int t = 0; t < num_threads; ++t) {
                if (threadOutputs[t][i] != 0) {
                    outputMap[i] = threadOutputs[t][i];
                    break;
                }
            }
        }
    }

    std::cout << "Occlusion fill complete using OpenMP" << std::endl;
    return outputMap;
}


/**
* @brief Executes the OpenCL kernel to resize the image by 4
* 
* This function:
* Initialized OpenCL,
* Creates the context and command queue,
* Copies image data to the device,
* Sets the kernel arguments,
* Loads and compiles 'resizeKernel.cl'
* Executes the kernel,
* Reads the resized image's contents back to the host,
* Cleans up the memory after execution,
* Monitors the execution time
* 
* @param imptr: Pointer to the input image data
* @param width: Original width of the input image
* @param height: Original height of the input image
* @return unsigned char* outputImage: The pointer to a new char array that contains the resized image's data
*/
unsigned char * executeResizeKernel(unsigned char * imptr, unsigned width, unsigned height) {

	auto start = std::chrono::high_resolution_clock::now();

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
    clEnqueueWriteBuffer(queue, buffer_in, CL_TRUE, 0, width*height*4, imptr, 0, NULL, NULL);

    // Load and compile the kernel
	std::string sourceString = loadKernelFromFile("resizeKernel.cl");
    const char *source = sourceString.c_str();
	
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "ResizeImage", NULL);

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

	auto end = std::chrono::high_resolution_clock::now();
	opTime = end-start;

	std::cout << "Resize execution time: " << opTime.count() << "s" << std::endl;

	return outputImage;
}

/**
* @brief Executes the OpenCL kernel to convert the given image to grayscale
* 
* This function:
* Initialized OpenCL,
* Creates the context and command queue,
* Copies image data to the device,
* Sets the kernel arguments,
* Loads and compiles 'grayScaleKernel.cl'
* Executes the kernel,
* Reads the grayscaled image's contents back to the host,
* Cleans up the memory after execution,
* Monitors the execution time
* 
* @param imptr: Pointer to the input image data
* @param width: Width of the input image
* @param height: Height of the input image
* @return unsigned char* outputImage: The pointer to a new char arra that contains the grayscale image's data
*/
unsigned char * executeGrayScaleKernel(unsigned char * imptr, unsigned width, unsigned height) {

	auto start = std::chrono::high_resolution_clock::now();
	unsigned char* outputImage = new unsigned char[width*height*4];

    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create buffers
    cl_mem buffer_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*4, NULL, NULL);
	cl_mem buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height*4, NULL, NULL);

    // Copy data to the device
    clEnqueueWriteBuffer(queue, buffer_in, CL_TRUE, 0, width*height*4, imptr, 0, NULL, NULL);

    // Load and compile the kernel
	std::string sourceString = loadKernelFromFile("grayScaleKernel.cl");
    const char *source = sourceString.c_str();
	
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "GrayScaleImage", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_in);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_out);
    clSetKernelArg(kernel, 2, sizeof(unsigned), &width);
    clSetKernelArg(kernel, 3, sizeof(unsigned), &height);

    // Execute the kernel
	size_t global_work_size[2] = {width, height};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Read results back to the host
    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, width*height*4, outputImage, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(buffer_in);
	clReleaseMemObject(buffer_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

	auto end = std::chrono::high_resolution_clock::now();
	opTime = end-start;

	std::cout << "Grayscale execution time: " << opTime.count() << "s" << std::endl;

	return outputImage;
}

/**
* @brief Executes the OpenCL kernel to apply gaussian blur filter to the given image.
* 
* This function:
* Initialized OpenCL,
* Creates the context and command queue,
* Copies image data to the device,
* Sets the kernel arguments,
* Loads and compiles 'filterKernel.cl'
* Executes the kernel,
* Reads the filtered image's contents back to the host,
* Cleans up the memory after execution,
* Monitors the execution time
* 
* @param imptr: Pointer to the input image data
* @param width: Width of the input image
* @param height: Height of the input image
* @return unsigned char* outputImage: The pointer to a new char arra that contains the filtered image's data
*/
unsigned char * executeFilterKernel(unsigned char * imptr, unsigned width, unsigned height) {

	auto start = std::chrono::high_resolution_clock::now();
	unsigned char* outputImage = new unsigned char[width*height*4];

    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create buffers
    cl_mem buffer_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*4, NULL, NULL);
	cl_mem buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height*4, NULL, NULL);

    // Copy data to the device
    clEnqueueWriteBuffer(queue, buffer_in, CL_TRUE, 0, width*height*4, imptr, 0, NULL, NULL);

    // Load and compile the kernel
	std::string sourceString = loadKernelFromFile("filterKernel.cl");
    const char *source = sourceString.c_str();
	
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "FilterImage", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_in);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_out);
    clSetKernelArg(kernel, 2, sizeof(unsigned), &width);
    clSetKernelArg(kernel, 3, sizeof(unsigned), &height);

    // Execute the kernel
	size_t global_work_size[2] = {width, height};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Read results back to the host
    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, width*height*4, outputImage, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(buffer_in);
	clReleaseMemObject(buffer_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

	auto end = std::chrono::high_resolution_clock::now();
	opTime = end-start;

	std::cout << "Filter execution time: " << opTime.count() << "s" << std::endl;

	return outputImage;
}

/**
* @brief Executes the OpenCL kernel to calculate the Zero-Mean Normalized Cross-Correlation
* (ZNCC) dispary map between the given two images.
* 
* This function:
* Initialized OpenCL,
* Creates the context and command queue,
* Copies image data to the device,
* Sets the kernel arguments,
* Loads and compiles 'znccKernel.cl'
* Executes the kernel,
* Reads the disparity map back to the host,
* Cleans up the memory after execution,
* Monitors the execution time
* 
* @param imptr1: Pointer to the first input image's data
* @param imptr2: Pointer to the second input image's data
* @param width: Width of the input images
* @param height: Height of the input images
* @param windowSize: The size of the ZNCC window
* @param max_disp: The maximum disparity to search for
* @return unsigned char* outputImage: The pointer to a new char array that contains the output disparity map
*/
unsigned char * executeZNCCKernel(unsigned char * imptr, unsigned char * imptr2, unsigned width, unsigned height, unsigned windowSize, unsigned max_disp) {

	auto start = std::chrono::high_resolution_clock::now();
	unsigned char* outputImage = new unsigned char[width*height*4];

    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create buffers
    cl_mem buffer1_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*4, NULL, NULL);
	cl_mem buffer2_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*4, NULL, NULL);
	cl_mem buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height*4, NULL, NULL);

    // Copy data to the device
    clEnqueueWriteBuffer(queue, buffer1_in, CL_TRUE, 0, width*height*4, imptr, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, buffer2_in, CL_TRUE, 0, width*height*4, imptr2, 0, NULL, NULL);

    // Load and compile the kernel
	std::string sourceString = loadKernelFromFile("znccKernel.cl");
    const char *source = sourceString.c_str();
	
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "ZNCCKernel", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer1_in);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer2_in);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_out);
    clSetKernelArg(kernel, 3, sizeof(unsigned), &width);
    clSetKernelArg(kernel, 4, sizeof(unsigned), &height);
	clSetKernelArg(kernel, 5, sizeof(unsigned), &windowSize);
	clSetKernelArg(kernel, 6, sizeof(unsigned), &max_disp);

    // Execute the kernel
	size_t global_work_size[2] = {width, height};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Read results back to the host
    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, width*height*4, outputImage, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(buffer1_in);
	clReleaseMemObject(buffer2_in);
	clReleaseMemObject(buffer_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

	auto end = std::chrono::high_resolution_clock::now();
	opTime = end-start;

	std::cout << "ZNCC execution time: " << opTime.count() << "s" << std::endl;

	return outputImage;
}

/**
* @brief Executes the OpenCL kernel to perform cross-checking between two given disparity maps
* 
* This function:
* Initialized OpenCL,
* Creates the context and command queue,
* Copies image data to the device,
* Sets the kernel arguments,
* Loads and compiles 'XCheckKernel.cl'
* Executes the kernel,
* Reads the disparity map back to the host,
* Cleans up the memory after execution,
* Monitors the execution time
* 
* @param map1: Pointer to the first disparity map data
* @param map2: Pointer to the second disparity map data
* @param width: Width of the disparity maps
* @param height: Height of the disparity maps
* @param threshold: The maximum allowed difference between the values of the disparity values
* @return unsigned char* outputMap: The pointer to a new char array containing the cross-checked disparity map
* 
*/
unsigned char * executeXCheckKernel(unsigned char * map1, unsigned char * map2, unsigned width, unsigned height, unsigned threshold) {

	auto start = std::chrono::high_resolution_clock::now();
	unsigned char* outputMap = new unsigned char[width*height*4];

    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create buffers
    cl_mem buffer1_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*4, NULL, NULL);
	cl_mem buffer2_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*4, NULL, NULL);
	cl_mem buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height*4, NULL, NULL);

    // Copy data to the device
    clEnqueueWriteBuffer(queue, buffer1_in, CL_TRUE, 0, width*height*4, map1, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, buffer2_in, CL_TRUE, 0, width*height*4, map2, 0, NULL, NULL);

    // Load and compile the kernel
	std::string sourceString = loadKernelFromFile("XCheckKernel.cl");
    const char *source = sourceString.c_str();
	
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "CrossCheck", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer1_in);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer2_in);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_out);
    clSetKernelArg(kernel, 3, sizeof(unsigned), &width);
    clSetKernelArg(kernel, 4, sizeof(unsigned), &height);
	clSetKernelArg(kernel, 5, sizeof(unsigned), &threshold);

    // Execute the kernel
	size_t global_work_size[2] = {width, height};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Read results back to the host
    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, width*height*4, outputMap, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(buffer1_in);
	clReleaseMemObject(buffer2_in);
	clReleaseMemObject(buffer_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

	auto end = std::chrono::high_resolution_clock::now();
	opTime = end-start;

	std::cout << "Cross check execution time: " << opTime.count() << "s" << std::endl;

	return outputMap;
}

/**
* @brief Executes the OpenCL kernel to fill zero-valued pixels in a given disparity map.
* 
* This function:
* Initialized OpenCL,
* Creates the context and command queue,
* Copies image data to the device,
* Sets the kernel arguments,
* Loads and compiles 'OccFillKernel.cl'
* Executes the kernel,
* Reads the disparity map back to the host,
* Cleans up the memory after execution,
* Monitors the execution time
*
* @param map: Pointer to the input disparity map data
* @param width: Width of the disparity map
* @param height: Height of the disparity map
* @return unsinged char* outputMap: The pointer to a new char array containing the disparity map with occlusion fill 
* 
*/
unsigned char * executeOccFillKernel(unsigned char * map, unsigned width, unsigned height) {

	auto start = std::chrono::high_resolution_clock::now();
	unsigned char* outputMap = new unsigned char[width*height*4];

    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create buffers
    cl_mem buffer_in = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*4, NULL, NULL);
	cl_mem buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height*4, NULL, NULL);

    // Copy data to the device
    clEnqueueWriteBuffer(queue, buffer_in, CL_TRUE, 0, width*height*4, map, 0, NULL, NULL);

    // Load and compile the kernel
	std::string sourceString = loadKernelFromFile("OccFillKernel.cl");
    const char *source = sourceString.c_str();
	
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "OcculsionFill", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_in);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_out);
    clSetKernelArg(kernel, 2, sizeof(unsigned), &width);
    clSetKernelArg(kernel, 3, sizeof(unsigned), &height);

    // Execute the kernel
	size_t global_work_size[2] = {width, height};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Read results back to the host
    clEnqueueReadBuffer(queue, buffer_out, CL_TRUE, 0, width*height*4, outputMap, 0, NULL, NULL);

    // Cleanup
    clReleaseMemObject(buffer_in);
	clReleaseMemObject(buffer_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

	auto end = std::chrono::high_resolution_clock::now();
	opTime = end-start;

	std::cout << "Cross check execution time: " << opTime.count() << "s" << std::endl;

	return outputMap;
}

/**
* @brief Main function that contains the pipeline for executing different OpenCL kernels for image processing
* The main function first executes display()-function to check for errors and showcasing OpenCl information. 
* If an error arises during the execution of display()-function, the program exits gracefully.
* 
* After executing display()-function, the main function reads and loads images
" img0.png" and "img1.png". The images should be in the same directory as the main.cpp file,
* or an IDE should be set up in a way that they can be found and read.
* Afterwards the main function perforrms a series of image processing by
* executing the different OpenCL kernels.
* (resizeKernel.cl, grayScaleKernel.cl, filterKernel.cl, znccKernel.cl, XCheckKernel.cl, and OccFillKernel)
* 
* @return 0 if the program exists successfully, non-zero (1) in other cases.
*/

int main() {

    //Main for openCL

    display();

	ImageData inputImage1("im0.png");
	ImageData inputImage2("im1.png");

	unsigned width = inputImage1.getWidth();
	unsigned height = inputImage1.getHeight();

	unsigned char* inptr1 = inputImage1.ImageDataToCharPointer();
	unsigned char* inptr2 = inputImage2.ImageDataToCharPointer();

	unsigned char * outptr1 = executeResizeKernel(inptr1, width, height);
	unsigned char * outptr2 = executeResizeKernel(inptr2, width, height);

	outptr1 = executeGrayScaleKernel(outptr1, width/4, height/4);
	outptr2 = executeGrayScaleKernel(outptr2, width/4, height/4);

	outptr1 = executeFilterKernel(outptr1, width/4, height/4);
	outptr2 = executeFilterKernel(outptr2, width/4, height/4);

	unsigned char * map1 = executeZNCCKernel(outptr1, outptr2, width/4, height/4, 9, 260);
    unsigned char * map2 = executeZNCCKernel(outptr2, outptr1, width/4, height/4, 9, 260);

    unsigned char * XcheckMap = executeXCheckKernel(map1, map2, width/4, height/4, 8);

    unsigned char * result = executeOccFillKernel(XcheckMap, width/4, height/4);

	ImageData out(result, width/4, height/4);
	out.WriteImageToFile("zncc_output.png");

	delete[] outptr1;
	delete[] outptr2;
	delete[] inptr1;
	delete[] inptr2;
    delete[] map1;
    delete[] map2;
    delete[] XcheckMap;
    delete[] result;

    return 0;
}


/* int main() {

	ImageData image1("im0.png");
	ImageData image2("im1.png");

	image1.ResizeImage();
	image2.ResizeImage();

	image1.ImageToGrayscale();
	image2.ImageToGrayscale();

	std::vector<unsigned char> map1 = CalcZNCC_multithreaded(image1.getImage(), image2.getImage(), image1.getWidth(), image1.getHeight(), 9, 260);
	std::vector<unsigned char> map2 = CalcZNCC_multithreaded(image2.getImage(), image1.getImage(), image1.getWidth(), image1.getHeight(), 9, 260);

	std::vector<unsigned char> mapX = CrossCheck(map1, map2, 8);

	std::vector<unsigned char> mapOut = OcclusionFillMultithreaded(mapX, image1.getWidth(), image1.getHeight());

	ImageData outImage(mapOut, image1.getWidth(), image1.getHeight());

	outImage.WriteImageToFile("output.png");

	return 0;
} */


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

int main() {

	ImageData inputImage("im0.png");
	unsigned width = inputImage.getWidth();
	unsigned height = inputImage.getHeight();

	unsigned char* inptr = inputImage.ImageDataToCharPointer();
	unsigned char * outptr = executeResizeKernel(inptr, width, height);
	outptr = executeGrayScaleKernel(outptr, width/4, height/4);
	outptr = executeFilterKernel(outptr, width/4, height/4);

	ImageData out(outptr, width/4, height/4);
	out.WriteImageToFile("im0_filterd.png");

	delete[] outptr;
	delete[] inptr;

    return 0;
}





/*  int main() {

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




//Todo: 
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>

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

int main() {

	cl_int err;

	display();

	int size = MATRIXSIZE * MATRIXSIZE;

	float *matrix_1 = (float *)malloc(size * sizeof(float));
	float *matrix_2 = (float *)malloc(size * sizeof(float));
	float *result = (float *)malloc(size * sizeof(float));

	for (int i = 0; i < size; i++) {
		matrix_1[i] = (float)i;
		matrix_2[i] = (float)(i * 2);
	}

	cl_platform_id platform;
	cl_device_id device;
	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &err);

	cl_mem buffer_matrix_1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), matrix_1, NULL);
	cl_mem buffer_matrix_2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), matrix_2, NULL);
	cl_mem buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), NULL, NULL);

	const char *kernelSource = "__kernel void matrix_add(__global const float *matrix_1, __global const float *matrix_2, __global float *result) { int id = get_global_id(0); result[id] = matrix_1[id] + matrix_2[id]; }";
	cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
	clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "matrix_add", &err);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_matrix_1);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_matrix_2);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_result);

	size_t globalSize = size;
	double start_time = getTime();
	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
	double end_time = getTime();
	
	printf("Host execution time: %f seconds\n", end_time - start_time);

	clEnqueueReadBuffer(command_queue, buffer_result, CL_TRUE, 0, size * sizeof(float), result, 0, NULL, NULL);

	for (int i = 0; i < 10; i++) {
		printf("%f", result[i]);
	}
	printf("\n");

	clReleaseMemObject(buffer_matrix_1);
	clReleaseMemObject(buffer_matrix_2);
	clReleaseMemObject(buffer_result);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(matrix_1);
	free(matrix_2);
	free(result);

	
	return 0;
}



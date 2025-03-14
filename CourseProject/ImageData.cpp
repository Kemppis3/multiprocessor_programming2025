#include "ImageData.hpp"
#include "lodepng.h"
#include <iostream>

ImageData::~ImageData() {};

ImageData::ImageData() {};

ImageData::ImageData(std::vector<unsigned char> image, unsigned int w, unsigned int h) {
    imageVector = image;
    width = w;
    height = h;
}

ImageData::ImageData(std::string filename) {

	std::vector<unsigned char> image;
	unsigned int w, h;

	unsigned int error = lodepng::decode(image, w, h, filename);

	if (error) {
		std::cout << "decoder error" << lodepng_error_text(error) << std::endl;
	} else {
		std::cout << "ImageData created from .png file!" << std::endl;
		imageVector = image;
        width = w;
        height = h;
	}
}

ImageData::ImageData(unsigned char* imptr, unsigned w, unsigned h) {

    unsigned int size = w*h*4;
	std::vector<unsigned char> imdata(imptr, imptr+size);

    imageVector = imdata;
    width = w;
    height = h;

	std::cout << "ImageData created from char pointer" << std::endl;
}

std::vector<unsigned char> ImageData::getImage() {
    return imageVector;
}

unsigned int ImageData::getHeight() {
    return height;
}

unsigned int ImageData::getWidth() {
    return width;
}


void ImageData::ResizeImage() {

	unsigned int newWidth = width / 4;
	unsigned int newHeight = height / 4;
	std::vector<unsigned char> resizedImage(newWidth * newHeight * 4);

	for (unsigned y = 0; y < newHeight; ++y)
	{
		for (unsigned x = 0; x < newWidth; ++x)
		{
			unsigned int originalX = x * 4;
			unsigned int originalY = y * 4;
			unsigned int originalIndex = (originalY * width + originalX) * 4;
			unsigned int newIndex = (y * newWidth + x) * 4;

			resizedImage[newIndex] = imageVector[originalIndex];
			resizedImage[newIndex + 1] = imageVector[originalIndex + 1];
			resizedImage[newIndex + 2] = imageVector[originalIndex + 2];
			resizedImage[newIndex + 3] = imageVector[originalIndex + 3];
		}
	}

	std::cout << "Image rescaled to size: " << "(" << newWidth << "x" << newHeight << ") " << "Image rescaling time = " << std::endl;

    imageVector = resizedImage;
    height = newHeight;
    width = newWidth;
}

void ImageData::ImageToGrayscale() {

	for (unsigned i = 0; i < width * height; ++i){

		unsigned char r = imageVector[i * 4];
		unsigned char g = imageVector[i * 4 + 1];
		unsigned char b = imageVector[i * 4 + 2];

		unsigned char gray = static_cast<unsigned char>(0.2126 * r + 0.7152 * g + 0.0722 * b);

		imageVector[i * 4] = gray;
		imageVector[i * 4 + 1] = gray;
		imageVector[i * 4 + 2] = gray;
	}

	std::cout << "Image converted to greyscale!" << std::endl;
}

void ImageData::ApplyGaussianFilter() {
    
    const float gaussianKernel[5][5] = {
		{1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f},
		{4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
		{6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f},
		{4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
		{1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f}
    };

	for (unsigned int y = 2; y < height - 2; y++)
	{
		for (unsigned int x = 2; x < width - 2; x++) {

			float sumR = 0, sumG = 0, sumB = 0;
			unsigned int pixelIndex;

			// Apply 5x5 kernel
			for (int ky = -2; ky <= 2; ky++) {
				for (int kx = -2; kx <= 2; kx++) {
					
                    pixelIndex = ((y + ky) * width + (x + kx)) * 4;

					sumR += imageVector[pixelIndex] * gaussianKernel[ky + 2][kx + 2];
					sumG += imageVector[pixelIndex + 1] * gaussianKernel[ky + 2][kx + 2];
					sumB += imageVector[pixelIndex + 2] * gaussianKernel[ky + 2][kx + 2];
				}
			}
			
			pixelIndex = (y * width + x) * 4;
			imageVector[pixelIndex] = static_cast<unsigned char>(sumR);
			imageVector[pixelIndex + 1] = static_cast<unsigned char>(sumG);
			imageVector[pixelIndex + 2] = static_cast<unsigned char>(sumB);
		}
	}

	std::cout << "Image filtered with gaussian blur filter!" << std::endl;
}


int ImageData::WriteImageToFile(const char* filename) {

	unsigned int error;
	error = lodepng::encode(filename, imageVector, width, height);

	if (error)
	{
		std::cout << "Error saving image" << lodepng_error_text(error) << std::endl;
		return error;
	}
	std::cout << "Image saved to " << filename << " (" << width << "x" << height << ") " << std::endl;
	return 0;
}


unsigned char* ImageData::ImageDataToCharPointer() {

	std::vector<unsigned char> imdata = imageVector;
	size_t size = imdata.size();
	unsigned char* imptr = new unsigned char[size];
	
	memcpy(imptr, imdata.data(), size);

	std::cout << "Image converted to unsigned char pointer!";

	return imptr;
}

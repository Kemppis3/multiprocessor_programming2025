__kernel void ResizeImage(__global unsigned char* image, __global unsigned char* output, const unsigned width, const unsigned height) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    unsigned newWidth = width/4;
    unsigned newHeight = height/4;

    if(x < newWidth && y < newHeight) {

        unsigned originalX = x * 4;
		unsigned originalY = y * 4;
		unsigned originalIndex = (originalY * width + originalX) * 4;
		unsigned newIndex = (y * newWidth + x) * 4;

        output[newIndex] = image[originalIndex];
		output[newIndex + 1] = image[originalIndex + 1];
		output[newIndex + 2] = image[originalIndex + 2];
		output[newIndex + 3] = image[originalIndex + 3];

    }
 
}

__kernel void GrayScaleImage(__global unsigned char* image, __global unsigned char* output, const unsigned width, const unsigned height) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < width && y < height) {

        unsigned index = (y*width+x)*4;

        unsigned char r = image[index];
		unsigned char g = image[index + 1];
		unsigned char b = image[index + 2];

		unsigned char gray = (unsigned char)(0.2126 * r + 0.7152 * g + 0.0722 * b);

		output[index] = gray;
		output[index + 1] = gray;
		output[index + 2] = gray;
        output[index + 3] = image[index + 3];
    }
}


__kernel void FilterImage(__global unsigned char* image, __global unsigned char* output, const unsigned width, const unsigned height) {

    const float gaussianKernel[5][5] = {
	{1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f},
	{4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
	{6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f},
	{4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
	{1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f}};

    int x = get_global_id(0);
    int y = get_global_id(1);

    if(y >= 2 && y < height-2 && x >= 2 && x < width-2) {

        float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

		for (int ky = -2; ky <= 2; ky++) {
			for (int kx = -2; kx <= 2; kx++) {
				
                unsigned pixelIndex = ((y + ky) * width + (x + kx)) * 4;

				sumR += image[pixelIndex] * gaussianKernel[ky + 2][kx + 2];
				sumG += image[pixelIndex + 1] * gaussianKernel[ky + 2][kx + 2];
				sumB += image[pixelIndex + 2] * gaussianKernel[ky + 2][kx + 2];
			}
		}

        unsigned pixelIndex = (y*width+x) * 4;
	    output[pixelIndex] = sumR;
	    output[pixelIndex + 1] = sumG;
	    output[pixelIndex + 2] = sumB;
        output[pixelIndex + 3] = 255;
    }

}

__kernel void ProcessImage(__global unsigned char* image, __global unsigned char* output, const unsigned width, const unsigned height) {

    unsigned newWidth = width/4;
    unsigned newHeight = height/4;

    ResizeImage(image, output, width, height);
    
    GrayScaleImage(output, output, newWidth, newHeight);

    FilterImage(output, output, newWidth, newHeight);

}

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
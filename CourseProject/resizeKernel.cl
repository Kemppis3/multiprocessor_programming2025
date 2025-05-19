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
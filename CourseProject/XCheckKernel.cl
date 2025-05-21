__kernel void CrossCheck(__global unsigned char * map1, __global unsigned char * map2, global unsigned char * output, unsigned int width, unsigned height, unsigned int threshold) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = (y*width+x)*4;

    if(abs_diff(map1[index], map2[index]) > threshold) {
        output[index] = 0;
        output[index+1] = 0;
        output[index+2] = 0;
        output[index+3] = 255;
    } else {
        output[index] = ((int)map1[index] + (int)map2[index]) / 2;
        output[index+1] = ((int)map1[index+1] + (int)map2[index+1]) / 2;
        output[index+2] = ((int)map1[index+2] + (int)map2[index+2]) / 2;
        output[index+3] = 255;

    }
}
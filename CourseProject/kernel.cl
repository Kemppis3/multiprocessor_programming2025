double CalcMean(__global unsigned char* image, int x, int y, int width, int height, int windowSize, int d) {

    int halfWindow = windowSize/2;
    double result = 0.0;
    int validPixels = 0;

    for(int dy = -halfWindow; dy < halfWindow; dy++) {

        for(int dx = -halfWindow; dx < halfWindow; dx++) {

            if(y+dy < 0 || y+dy > height || x+dx-d < 0 || x+dx > width) continue;

            int index = ((y+dy)*width+(x+dx-d))*4;
            result += image[index];
            validPixels++;   
        }
    }

    if(validPixels != 0) {
        result /= validPixels;
    }
    
    return result;
}


double CalcZNCC(__global unsigned char * image1, __global unsigned char * image2, int x, int y, int width, int height, int windowSize, int d) {

    int halfWindow = windowSize/2;
    double result = 0;
    int index1;
    int index2;
    double mean1;
    double mean2;
    double std1;
    double std2;

    for(int dy = -halfWindow; dy < halfWindow; dy++) {

        for(int dx = -halfWindow; dx < halfWindow; dx++) {

            if (y+dy >= height || x+dx >= width || x+dx-d < 0) continue;

            index1 = ((y+dy)*width+(x+dx))*4;
            index2 = ((y+dy)*width+(x+dx-d))*4;

            mean1 = CalcMean(image1, x, y, width, height, windowSize, 0);
            mean2 = CalcMean(image2, x, y, width, height, windowSize, d);

            std1 += (image1[index1] - mean1) * (image1[index1] - mean1);
            std2 += (image2[index2] - mean2) * (image2[index2] - mean2);
            result += (image1[index1] - mean1) * (image2[index2] - mean2);
        }
    }

    if(std1 != 0 && std2 != 0) {
        result /= sqrt(std1)*sqrt(std2);
    }

    return result;
}


__kernel void ZNCCKernel(__global unsigned char* image1, __global unsigned char* image2, global unsigned char* outputImage, int width, int height, int windowSize, int max_disp) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    double best_val;
    double best_d;

    double mean1 = CalcMean(image1, x, y, width, height, windowSize, 0);

    for(int d = 0; d < max_disp; d++) {

        double mean2 = CalcMean(image2, x, y, width, height, windowSize, d);

        double zncc_val = CalcZNCC(image1, image2, x, y, width, height, windowSize, d);
        
        if(zncc_val > best_val) {
            best_val = zncc_val;
            best_d = d;
        }

    }

    int index = (y*width+x)*4;

    best_d = round((best_d/max_disp)*255.0);

	outputImage[index] = best_d;
	outputImage[index+1] = best_d;
	outputImage[index+2] = best_d;
	outputImage[index+3] = 255;

}

__kernel void ProcessImage(__global unsigned char* image, __global unsigned char* output, const unsigned width, const unsigned height) {

}

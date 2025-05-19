__kernel void FilterImage(__global unsigned char* image, __global unsigned char* output, const unsigned width, const unsigned height) {

    __constant float gaussianKernel[5][5] = {
    {1.0f / 256.0f, 4.0f / 256.0f, 6.0f / 256.0f, 4.0f / 256.0f, 1.0f / 256.0f},
    {4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f},
    {6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f},
    {4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f},
    {1.0f / 256.0f, 4.0f / 256.0f, 6.0f / 256.0f, 4.0f / 256.0f, 1.0f / 256.0f}
    };

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {

            if(y+ky < 0 || y+ky >= height || x+kx < 0 || x+kx >= width) continue;

            unsigned index = ((y + ky) * width + (x + kx)) * 4;

            sumR += image[index] * gaussianKernel[ky + 2][kx + 2];
            sumG += image[index + 1] * gaussianKernel[ky + 2][kx + 2];
            sumB += image[index + 2] * gaussianKernel[ky + 2][kx + 2];
        }
    }

    unsigned pixelIndex = (y*width+x) * 4;
    output[pixelIndex] = (unsigned char)sumR;
    output[pixelIndex + 1] = (unsigned char)sumG;
    output[pixelIndex + 2] = (unsigned char)sumB;
    output[pixelIndex + 3] = (unsigned char)255;

}
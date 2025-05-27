__kernel void OcculsionFill(__global unsigned char * map, __global unsigned char * output, unsigned int width, unsigned height) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = (y * width + x) * 4;

    if (map[index] == 0) {

        bool flag = false;

        for (int n = 1; n < 100 && !flag; n++) {
            for (int i = -n; i <= n && !flag; i++) {
                for (int j = -n; j <= n; j++) {

                    int index2 = ((y+i)*width+(x+j))*4;

                    if(y+i > height || x+j > width || index2 < 0 || map[index2] == 0) {
                        continue;
                    } else {
                        output[index] = map[index2];
                        output[index+1] = map[index2+1];
                        output[index+2] = map[index2+2];
                        output[index+3] = 255;
                        flag = true;
                        break;
                    }
                }
            }
        }
    } else {
        output[index] = map[index];
        output[index+1] = map[index+1];
        output[index+2] = map[index+2];
        output[index+3] = 255;
    }
}
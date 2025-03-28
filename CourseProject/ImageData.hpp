#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include<vector>
#include<string>

class ImageData {

    public:

        //Constructors

        //Default constructor
        ImageData();

        //Create ImageData from existing image vector
        ImageData(std::vector<unsigned char> image, unsigned int w, unsigned int h);

        //Create ImageData from .png file
        ImageData(std::string filename);

        //Create ImageData from unsigned char pointer
        ImageData(unsigned char* imptr, unsigned int w, unsigned int h);

        ~ImageData();

        //Writes image to a .png file
        int WriteImageToFile(const char* filename);

        //Get
        std::vector<unsigned char> & getImage();
        unsigned int & getHeight();
        unsigned int & getWidth();

        //Image processing functions:

        //Resizes image by 4 -> new image widht and height are divided by 4
        void ResizeImage();

        //Converts image to grayscale
        void ImageToGrayscale();

        //Applies gaussian blur filter on image
        void ApplyGaussianFilter();

        //Converts image to a char pointer
        unsigned char* ImageDataToCharPointer();

        //Converts char pointer to ImageData object
        ImageData CharPointerToImageData(unsigned char* imptr, unsigned width, unsigned height);



    private:

        unsigned int width = 0;
        unsigned int height = 0;
        std::vector<unsigned char> imageVector;

};

#endif
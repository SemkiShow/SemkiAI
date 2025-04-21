#include <iostream>
#include <random>
#include <ctime>
#include <pnglib.hpp>
using namespace puzniakowski::png;

int main()
{
    srand(time(0));
    int currentNumber = 0;
    pngimage_t inputImage;
    inputImage = read_png_file("../dataset/MNIST/"+std::to_string(currentNumber)+"/"+std::to_string(0)+".png");
    for (int i = 0; i < 28*28; i++)
    {
        // int currentNumber = rand() % 10;
        // inputImage = read_png_file("../dataset/MNIST/"+std::to_string(currentNumber)+"/"+std::to_string(rand()%10000)+".png");
        unsigned int color = inputImage.get(i % 28, i / 28);
        // unsigned int color = inputImage.get(0, 0);
        int8_t r,g,b,a;
        getRGBAFromColor(color, &r, &g, &b, &a);
        std::cout << currentNumber << ",";
        std::cout << std::to_string(((uint8_t)a) * 1.0 / 255) << "; ";   
    }
    std::cout << std::endl;
}

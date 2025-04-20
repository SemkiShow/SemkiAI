#include "SemkiAI.hpp"
#include <pnglib.hpp>
using namespace puzniakowski::png;

int main()
{
    /* Initialisation */
    // Initialising the perceptron
    Perceptron perceptron;
    // Use the GPU
    perceptron.useGPU = true;
    // Load the saved weights
    std::cout << "Loading weights..." << std::endl;
    perceptron.LoadWeights("weights");

    /* Using the loaded data */
    // Fill in the input
    srand(time(0));
    int currentNumber = -1;
    int8_t r,g,b,a;
    unsigned int color;
    int x, y = 0;
    currentNumber = rand() % 10;
    pngimage_t inputImage;
    inputImage = read_png_file("../dataset/MNIST/"+std::to_string(currentNumber)+"/"+std::to_string(rand()%10000)+".png");
    for (int i = 0; i < perceptron.neuronsConfig[0]; i++)
    {
        if (x > 28)
        {
            x = 0;
            y++;
        }
        color = inputImage.get(x, y);
        getRGBAFromColor(color, &r, &g, &b, &a);
        perceptron.neurons[i] = (uint8_t)r * 1.0 / 255;
    }
    std::cout << "Input data was loaded" << std::endl;
    // Calculate the output
    perceptron.CalculateNeurons(Perceptron::ActivationFunction::Sigmoid);
    // Output the answer
    for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
    {
        std::cout << perceptron.neurons[perceptron.maxNeurons * (perceptron.layers - 1) + i] << ", ";
    }
    std::cout << std::endl;
    return 0;
}

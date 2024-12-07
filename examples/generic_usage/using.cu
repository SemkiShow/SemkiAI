#include "../../src/SemkiAI.hpp"

int main()
{
    /* Initialisation */
    // Initialising the perceptron
    Perceptron perceptron;
    // Use the GPU
    perceptron.useGPU = true;
    // Load the saved weights
    perceptron.LoadWeights("weights");

    /* Using the loaded data */
    // Fill in the input
    srand(time(0));
    for (int i = 0; i < perceptron.neuronsConfig[0]; i++)
    {
        perceptron.neurons[i] = rand() % 1000000 * 1.0 / 1000000;
    }
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

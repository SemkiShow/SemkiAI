#include "SemkiAI.hpp"

int main()
{
    /* Initialisation */
    // Initialising the perceptron
    Perceptron perceptron;
    // Use the GPU
    perceptron.useGPU = true;
    // Load the saved weights
    std::cout << "Loading weights...\n";
    perceptron.LoadWeights("weights.csv");

    /* Using the loaded data */
    // Fill in the input
    std::cout << "Loading the input data...\n";
    srand(time(0));
    for (int i = 0; i < perceptron.neuronsConfig[0]-1; i++)
    {
        perceptron.neurons[i] = rand() % 1000000 * 1.0 / 1000000;
    }
    // Calculate the output
    std::cout << "Calculating the output...\n";
    perceptron.CalculateNeurons(Perceptron::ActivationFunction::Sigmoid);
    // Output the answer
    for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]-1; i++)
    {
        std::cout << perceptron.neurons[perceptron.neuronsIndexes[perceptron.layers-1] + i] << ", ";
    }
    std::cout << "\n";
    return 0;
}

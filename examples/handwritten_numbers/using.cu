#include "SemkiAI.hpp"
#include "DatasetParser.hpp"

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
    std::vector<std::vector<std::string>> dataset;
    dataset = ParseCSV("dataset/mnist_test.csv");
    int currentDatasetIndex = rand() % dataset.size();
    for (int i = 0; i < perceptron.neuronsConfig[0]-1; i++)
    {
        perceptron.neurons[i] = stoi(dataset[currentDatasetIndex][i+1]) / 255.0;
    }
    // Calculate the output
    std::cout << "Calculating the output...\n";
    perceptron.CalculateNeurons(Perceptron::ActivationFunction::Sigmoid);
    // Output the answer
    int answer = 0;
    for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
    {
        if (perceptron.neurons[perceptron.maxNeurons * (perceptron.layers - 1) + i] > perceptron.neurons[perceptron.maxNeurons * (perceptron.layers - 1) + answer])
            answer = i;
        std::cout << perceptron.neurons[perceptron.maxNeurons * (perceptron.layers - 1) + i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "The input number was " << dataset[currentDatasetIndex][0] << std::endl;
    std::cout << "The neural network thinks that the number is " << answer << std::endl;
    return 0;
}

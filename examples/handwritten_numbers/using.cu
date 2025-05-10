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

    // Load the dataset
    std::cout << "Loading the input data...\n";
    srand(time(0));
    std::vector<std::vector<std::string>> dataset;
    ParseCSV("dataset/mnist_test.csv", &dataset);

    /* Calculate the guess rate */
    int guessRate = 0;
    int guessesNumber = 100;
    for (int j = 0; j < guessesNumber; j++)
    {
        std::cout << "\rIteration: " << j << " (" << (j * 1.0 / guessesNumber * 100) << "%)" << std::flush;

        // Fill in the input
        int currentDatasetIndex = rand() % dataset.size();
        for (int i = 0; i < perceptron.neuronsConfig[0]-1; i++)
        {
            perceptron.neurons[i] = stoi(dataset[currentDatasetIndex][i+1]) / 255.0;
        }

        // Calculate the output
        perceptron.CalculateNeurons(Perceptron::ActivationFunction::Sigmoid);

        int answer = 0;
        for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]-1; i++)
        {
            if (perceptron.neurons[perceptron.neuronsIndexes[perceptron.layers-1] + i] > perceptron.neurons[perceptron.neuronsIndexes[perceptron.layers-1] + answer])
                answer = i;
        }

        if (answer == stoi(dataset[currentDatasetIndex][0])) guessRate++;
    }
    std::cout << "\nThe guess rate is " << (guessRate * 1.0 / guessesNumber * 100) << "%\n";
    return 0;
}

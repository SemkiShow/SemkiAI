#include "SemkiAI.hpp"
#include "DatasetParser.hpp"

int main()
{
    /* Initialisation */
    // Initialising the perceptron
    Perceptron perceptron;
    // Use the GPU
    perceptron.useGPU = true;
    // Set the amount of layers
    perceptron.layers = 123;
    // Init CUDA
    perceptron.InitCuda();
    // Set the amount of neurons in each layer
    perceptron.neuronsConfig[0] = 28*28;
    for (int i = 1; i < perceptron.layers-1; i++)
    {
        perceptron.neuronsConfig[i] = 128;
    }
    perceptron.neuronsConfig[perceptron.layers-1] = 10;
    // Init the right answers array
    perceptron.rightAnswer = new double[perceptron.neuronsConfig[perceptron.layers-1]];
    // Init random
    srand(time(0));
    // Init neurons and weights
    perceptron.Init();
    // Set the required variables for SimulatedAnnealing
    perceptron.temperature = 5000;
    perceptron.temperatureDecreaseRate = 0.99;
    // Set the learning rate
    perceptron.learningRate = 0.1;
    // Init the training info variables
    double initialError = 10;
    double endError = 10;
    int maxIterations = 123456;
    int iteration = 0;
    double acceptableError = 0.01;
    // An error buffer
    double buf = 0;
    // Load the dataset
    std::cout << "Loading the dataset...\n";
    int currentDatasetIndex = -1;
    std::vector<std::vector<std::string>> dataset;
    dataset = ParseCSV("dataset/mnist.csv");

    /* Training */
    std::cout << "Training...\n";
    while (endError > acceptableError && iteration < maxIterations)
    {
        // Print the current iteration number
        std::cout << '\r' << "Iteration: " << iteration << ", Current error: " << endError << std::flush;
        // Fill in the input
        currentDatasetIndex = rand() % dataset.size();
        for (int i = 0; i < perceptron.neuronsConfig[0]-1; i++)
        {
            perceptron.neurons[i] = stoi(dataset[currentDatasetIndex][i+1]) / 255.0;
        }
        // Set the right answer
        for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
        {
            perceptron.rightAnswer[i] = i == stoi(dataset[currentDatasetIndex][0]) ? 1 : 0;
        }
        // Run a training cycle
        buf = perceptron.Train(
            Perceptron::ActivationFunction::ReLU,
            Perceptron::CostFunction::MeanSquared, 
            Perceptron::LearningAlgorithm::Backpropagation);
        // Remember the error
        if (iteration == 0) initialError = buf; else endError = buf;
        // Increase the iterations counter
        iteration++;
    }
    // Print the error stats
    std::cout << '\n';
    std::cout << "Initial error was " << initialError << '\n';
    std::cout << "Now the error is " << endError << '\n';

    // Save the trained model to a .csv file
    perceptron.SaveWeights("weights.csv");
    return 0;
}

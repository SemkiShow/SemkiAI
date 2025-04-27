#include "SemkiAI.hpp"

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
    for (int i = 0; i < perceptron.layers; i++)
    {
        perceptron.neuronsConfig[i] = 999;
    }
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
    perceptron.learningRate = 1.0;
    // Init the training info variables
    double initialError = 10;
    double endError = 10;
    int maxIterations = 123;
    int iteration = 0;
    double acceptableError = -1;
    // An error buffer
    double buf = 0;

    /* Training */
    std::cout << "Training...\n";
    while (endError > acceptableError && iteration < maxIterations)
    {
        // Print the current iteration number
        std::cout << "\r" << "Iteration: " << iteration;
        // Set the input data
        for (int i = 0; i < perceptron.neuronsConfig[0]-1; i++)
        {
            perceptron.neurons[i] = rand() % 1000000 * 1.0 / 1000000;
        }
        // Set the right answer
        for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
        {
            perceptron.rightAnswer[i] = rand() % 1000000 * 1.0 / 1000000;
        }
        // Run a training cycle
        buf = perceptron.Train(
            Perceptron::ActivationFunction::Sigmoid,
            Perceptron::CostFunction::MeanAbsolute, 
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

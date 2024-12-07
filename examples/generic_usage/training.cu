#include "../../src/SemkiAI.hpp"

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
    perceptron.temperatureDecreaseRate = 0.01;
    // Set the learning rate
    perceptron.learningRate = 1.0;
    // Init the training info variables
    double initialError = 10;
    double endError = 10;
    int maxIterations = 7890;
    int iteration = 0;
    double acceptableError = -1;
    // An error buffer
    double buf = 0;

    /* Training */
    while (endError > acceptableError && iteration < maxIterations)
    {
        // Print the current iteration number
        std::cout << "Iteration: " << iteration << std::endl;
        // Set the input data
        for (int i = 0; i < perceptron.neuronsConfig[0]; i++)
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
            Perceptron::LearningAlgorithm::SimulatedAnnealing);
        // Remember the error
        if (iteration == 0) initialError = buf; else endError = buf;
        // Increase the iterations counter
        iteration++;
    }
    // Print the error stats
    std::cout << std::endl;
    std::cout << "Initial error was " << initialError << std::endl;
    std::cout << "Now the error is " << endError << std::endl;

    // Calculate the amount of RAM needed to run the model
    double spaceTaken = perceptron.layers;
    spaceTaken /= 1024;
    spaceTaken *= perceptron.neuronsConfig[0];
    spaceTaken /= 1024;
    spaceTaken *= 8;
    spaceTaken /= 1024;
    spaceTaken *= perceptron.neuronsConfig[0]-1;
    // Save the trained model to a .csv file
    perceptron.SaveWeights("weights"+std::to_string(iteration)+" "+std::to_string(spaceTaken)+"GB");
    return 0;
}

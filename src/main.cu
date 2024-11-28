#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include "AI.cuh"

int main()
{
    Perceptron perceptron;
    perceptron.useGPU = true;
    perceptron.layers = 123;
    perceptron.InitCuda();
    std::cout << perceptron.layers << std::endl;
    for (int i = 0; i < perceptron.layers; i++)
    {
        perceptron.neuronsConfig[i] = 999;
    }
    std::cout << "neuronsConfig was set" << std::endl;

    double rightAnswer[perceptron.neuronsConfig[perceptron.layers-1]];
    perceptron.rightAnswer = rightAnswer;
    srand(time(0));
    for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
    {
        perceptron.rightAnswer[i] = rand() % 1000000 * 1.0 / 1000000;
    }
    std::cout << "Right answer was set" << std::endl;
    
    perceptron.Init();
    perceptron.temperature = 5000;
    perceptron.temperatureDecreaseRate = 0.99;
    perceptron.learningRate = 1.0;
    // perceptron.delta = 1;
    double initialError = perceptron.Train(
        Perceptron::ActivationFunction::Sigmoid,
        Perceptron::CostFunction::MeanSquared, 
        Perceptron::LearningAlgorithm::Backpropagation);
    double endError = 10;
    int maxIterations = 7890;
    int iteration = 0;
    double acceptableError = 0.0001;
    while (endError > acceptableError && iteration < maxIterations)
    {
        std::cout << "Iteration: " << iteration << std::endl;
        for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
        {
            perceptron.rightAnswer[i] = rand() % 1000000 * 1.0 / 1000000;
        }
        endError = perceptron.Train(
            Perceptron::ActivationFunction::Sigmoid,
            Perceptron::CostFunction::MeanAbsolute, 
            Perceptron::LearningAlgorithm::SimulatedAnnealing);
        iteration++;
    }
    std::cout << std::endl;
    std::cout << "Initial error was " << initialError << std::endl;
    std::cout << "Now the error is " << endError << std::endl;

    double spaceTaken = perceptron.layers;
    spaceTaken /= 1024;
    spaceTaken *= perceptron.neuronsConfig[0];
    spaceTaken /= 1024;
    spaceTaken *= 8;
    spaceTaken /= 1024;
    spaceTaken *= perceptron.neuronsConfig[0]-1;
    perceptron.SaveWeights("weights"+std::to_string(iteration)+" "+std::to_string(spaceTaken)+"GB");
    return 0;
}

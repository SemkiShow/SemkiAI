#include <iostream>
#include <string>
#include <ctime>
#include "AI.cuh"
// #include "DatasetParser.h"
// using namespace std;

int main()
{
    // int N = 1<<20;
    // float *a, *b, *c;
    // cudaMallocManaged(&a, N*sizeof(float));
    // cudaMallocManaged(&b, N*sizeof(float));
    // cudaMallocManaged(&c, N*sizeof(float)); 
    // for (int i = 0; i < N; i++)
    // {
    //     a[i] = 1.0f;
    //     b[i] = 2.0f;
    // }

    // float* c = new float[N];
    // add(a, b, c, N);
    // std::cout << c[500] << std::endl;

    Perceptron perceptron;
    perceptron.useGPU = true;
    perceptron.layers = 123;
    // int neuronsConfig[perceptron.layers];
    perceptron.InitCuda();
    // std::cout << sizeof(neuronsConfig) / sizeof(*neuronsConfig) << std::endl;
    std::cout << perceptron.layers << std::endl;
    for (int i = 0; i < perceptron.layers; i++)
    {
        // neuronsConfig[i] = 345;
        perceptron.neuronsConfig[i] = 456;
    }
    // perceptron.neuronsConfig = neuronsConfig;
    std::cout << "neuronsConfig was set" << std::endl;
    // for (int i = 0; i < perceptron.layers; i++)
    // {
    //     if (i < 10)
    //     {
    //         std::cout << " " << i << "; ";
    //     }
    //     else
    //     {
    //         std::cout << i << "; ";
    //     }
        
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < perceptron.layers; i++)
    // {
    //     std::cout << perceptron.neuronsConfig[i] << "; ";
    // }
    // std::cout << std::endl;
    double rightAnswer[perceptron.neuronsConfig[perceptron.layers-1]];
    perceptron.rightAnswer = rightAnswer;
    for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
    {
        perceptron.rightAnswer[i] = 0.27158953;
    }
    std::cout << "Right answer was set" << std::endl;
    // cudaMallocManaged(&perceptron.neurons, perceptron.layers*perceptron.neuronsConfig[0]*sizeof(double));
    perceptron.Init();
    // perceptron.temperature = 5000;
    // perceptron.temperatureDecreaseRate = 0.99;
    perceptron.learningRate = 1.0;
    // perceptron.delta = 1;
    double initialError = perceptron.Train(
        Perceptron::ActivationFunction::Sigmoid,
        Perceptron::CostFunction::MeanSquared, 
        Perceptron::LearningAlgorithm::Backpropagation);
    double endError = 0;
    int iterations = 789;
    for (int i = 0; i < iterations; i++)
    {
        std::cout << "Iteration: " << i << std::endl;
        endError = perceptron.Train(
            Perceptron::ActivationFunction::Sigmoid,
            Perceptron::CostFunction::MeanSquared, 
            Perceptron::LearningAlgorithm::Backpropagation);
    }
    // for (int i = 0; i < 1000; i++)
    // {
    //     std::cout << "Iteration: " << i << std::endl;
    //     endError = perceptron.Train(
    //         Perceptron::ActivationFunction::Sigmoid,
    //         Perceptron::CostFunction::MeanSquared, 
    //         Perceptron::LearningAlgorithm::Backpropagation);
    // }
    // double endError = perceptron.Train(
    //     Perceptron::ActivationFunction::Sigmoid,
    //     Perceptron::CostFunction::MeanSquared, 
    //     Perceptron::LearningAlgorithm::Backpropagation);
    std::cout << std::endl;
    std::cout << "Initial error was " << initialError << std::endl;
    std::cout << "Now the error is " << endError << std::endl;

    // string path = "/home/semkishow/Wikipedia100/A";
    // string outputDirectory = "./dataset";
    // ParseHTML(&path[0], &outputDirectory[0]);

    // std::time_t time = std::time(nullptr);
    double spaceTaken = perceptron.layers;
    spaceTaken /= 1024;
    spaceTaken *= perceptron.neuronsConfig[0];
    spaceTaken /= 1024;
    spaceTaken *= 8;
    spaceTaken /= 1024;
    spaceTaken *= perceptron.neuronsConfig[0]-1;
    perceptron.SaveWeights("weights"+std::to_string(iterations)+" "+std::to_string(spaceTaken)+"GB");
    // perceptron.Free();
    // cudaFree(a);
    // cudaFree(b);
    // cudaFree(c);
    return 0;
}

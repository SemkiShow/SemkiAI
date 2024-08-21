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
    int neuronsConfig[42];
    perceptron.layers = 42;
    perceptron.InitCuda();
    // std::cout << sizeof(neuronsConfig) / sizeof(*neuronsConfig) << std::endl;
    for (int i = 0; i < perceptron.layers; i++)
    {
        neuronsConfig[i] = 42;
        perceptron.neuronsConfig[i] = 42;
    }
    std::cout << "neuronsConfig was set" << std::endl;
    // perceptron.neuronsConfig = neuronsConfig;
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
    double rightAnswer[neuronsConfig[perceptron.layers-1]];
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
    int iterations = 10;
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

    std::time_t time = std::time(nullptr);
    perceptron.SaveWeights("weights"+std::to_string(iterations)+" "+std::asctime(std::localtime(&time)));
    perceptron.Free();
    // cudaFree(a);
    // cudaFree(b);
    // cudaFree(c);
    return 0;
}

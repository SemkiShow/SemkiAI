#include <iostream>
#include <string>
#include "AI.cuh"
// #include "DatasetParser.h"
using namespace std;

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
    // cout << c[500] << endl;

    Perceptron perceptron;
    int neuronsConfig[42];
    perceptron.layers = 42;
    perceptron.InitCuda();
    // cout << sizeof(neuronsConfig) / sizeof(*neuronsConfig) << endl;
    for (int i = 0; i < perceptron.layers; i++)
    {
        neuronsConfig[i] = 42;
        perceptron.neuronsConfig[i] = 42;
    }
    cout << "neuronsConfig was set" << endl;
    // perceptron.neuronsConfig = neuronsConfig;
    // for (int i = 0; i < perceptron.layers; i++)
    // {
    //     if (i < 10)
    //     {
    //         cout << " " << i << "; ";
    //     }
    //     else
    //     {
    //         cout << i << "; ";
    //     }
        
    // }
    // cout << endl;
    // for (int i = 0; i < perceptron.layers; i++)
    // {
    //     cout << perceptron.neuronsConfig[i] << "; ";
    // }
    // cout << endl;
    double rightAnswer[neuronsConfig[perceptron.layers-1]];
    perceptron.rightAnswer = rightAnswer;
    for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
    {
        perceptron.rightAnswer[i] = 1.0;
    }
    cout << "Right answer was set" << endl;
    // cudaMallocManaged(&perceptron.neurons, perceptron.layers*perceptron.neuronsConfig[0]*sizeof(double));
    perceptron.Init();
    // perceptron.delta = 1;
    double initialError = perceptron.Train(
        Perceptron::ActivationFunction::Sigmoid,
        Perceptron::CostFunction::MeanSquared, 
        Perceptron::LearningAlgorithm::Backpropagation, 
        1.0);
    double endError;
    for (int i = 0; i < 5; i++)
    {
        cout << "Iteration: " << i << endl;
        endError = perceptron.Train(
            Perceptron::ActivationFunction::Sigmoid,
            Perceptron::CostFunction::MeanSquared, 
            Perceptron::LearningAlgorithm::Backpropagation, 
            10.0);
    }
    // double endError = perceptron.Train(
    //     Perceptron::ActivationFunction::Sigmoid,
    //     Perceptron::CostFunction::MeanSquared, 
    //     Perceptron::LearningAlgorithm::Backpropagation, 
    //     1.0);
    cout << endl;
    cout << "Initial error was " << initialError << endl;
    cout << "Now the error is " << endError << endl;

    // string path = "/home/semkishow/Wikipedia100/A";
    // string outputDirectory = "./dataset";
    // ParseHTML(&path[0], &outputDirectory[0]);

    perceptron.Free();
    // cudaFree(a);
    // cudaFree(b);
    // cudaFree(c);
    return 0;
}

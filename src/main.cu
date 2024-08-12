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
    for (int i = 0; i < sizeof(neuronsConfig) / sizeof(*neuronsConfig); i++)
    {
        neuronsConfig[i] = 42;
    }
    perceptron.neuronsConfig = neuronsConfig;
    perceptron.layers = sizeof(neuronsConfig) / sizeof(*neuronsConfig);
    // cudaMallocManaged(&perceptron.neurons, perceptron.layers*perceptron.neuronsConfig[0]*sizeof(float));
    perceptron.Init();
    perceptron.TrainGenerations(100);

    // string path = "/home/semkishow/Wikipedia100/A";
    // string outputDirectory = "./dataset";
    // ParseHTML(&path[0], &outputDirectory[0]);

    perceptron.Free();
    // cudaFree(a);
    // cudaFree(b);
    // cudaFree(c);
    return 0;
}

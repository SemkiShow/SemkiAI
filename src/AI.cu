#include "AI.cuh"

void test()
{
    std::cout << "Hello from the AI.cuh header file!" << std::endl;
}

__global__
void CalculateNeuronsKernel(float* neurons, float* weights, int* neuronsConfig, int layers, int layer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // int index = threadIdx.x;
    // int stride = blockDim.x;

    for (int i = index; i < neuronsConfig[layer+1]; i+=stride)
    {
        neurons[neuronsConfig[0]*(layer+1)+i] = 0;
        for (int j = 0; j < neuronsConfig[layer]; j++)
        {
            neurons[neuronsConfig[0]*(layer+1)+i] += neurons[neuronsConfig[0]*layer+j] * weights[layer*neuronsConfig[0]*(neuronsConfig[0]-1)+j*(neuronsConfig[0]-1)+i];
        }
    }
}

// void add(float* a, float* b, float* c, int N)
// {    
//     int threads = 256;
//     int blocks = (N + threads - 1) / threads;
//     addKernel<<<blocks, threads>>>(a, b, c, N);
//     cudaDeviceSynchronize();
// }

int Perceptron::Init()
{
    cudaMallocManaged(&neuronsConfig, layers*sizeof(int));

    cudaMallocManaged(&neurons, layers*neuronsConfig[0]*sizeof(float));
    // cout << layers*neuronsConfig[0] << endl;
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < neuronsConfig[0]; j++)
        {
            // cout << i*neuronsConfig[0]+j << endl;
            neurons[i*neuronsConfig[0]+j] = 4.2f;
        }
    }
    cout << "Neurons were initialized successfully" << endl;

    cudaMallocManaged(&weights, layers*neuronsConfig[0]*(neuronsConfig[0]-1)*sizeof(float));
    // int lastIndex = 0;
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            for (int k = 0; k < neuronsConfig[i]-1; k++)
            {
                // if (i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k - lastIndex != 1)
                // {
                //     cout << lastIndex << "->" << i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k << endl;
                // }
                // lastIndex = i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k;
                // cout << i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k << endl;
                weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] = 6.9f;
            }
        }
    }
    cout << "Weights were initialized successfully" << endl;
    return 0;
}

int Perceptron::CalculateNeurons()
{
    for (int i = 0; i < layers-1; i++)
    {
        // cout << i << endl;
        int threads = 256;
        int blocks = (neuronsConfig[i] + threads - 1) / threads;
        CalculateNeuronsKernel<<<blocks, threads>>>(neurons, weights, neuronsConfig, layers, i);
        cudaDeviceSynchronize();
    }
    cout << "Neurons were recalculated" << endl;
    return 0;
}

float Perceptron::MeanSquaredError()
{
    float output = 0;
    for (int i = 0; i < neuronsConfig[layers-1]; i++)
    {
        output += pow(neurons[neuronsConfig[0]*(layers-1)+i] - rightAnswer[i], 2);
    }
    output /= neuronsConfig[layers-1];
    return output;
}

float Perceptron::MeanAbsoluteError()
{
    float output = 0;
    for (int i = 0; i < neuronsConfig[layers-1]; i++)
    {
        output += abs(neurons[neuronsConfig[0]*(layers-1)+i] - rightAnswer[i]);
    }
    output /= neuronsConfig[layers-1];
    return output;
}

float Perceptron::HuberLoss(float delta)
{
    float output = 0;
    float output = 0;
    for (int i = 0; i < neuronsConfig[layers-1]; i++)
    {
        if (abs(neurons[neuronsConfig[0]*(layers-1)+i] - rightAnswer[i]) > delta)
        {
            output += delta * (abs(neurons[neuronsConfig[0]*(layers-1)+i] - rightAnswer[i]) - 0.5f * delta);
        }
        else
        {
            output += pow(neurons[neuronsConfig[0]*(layers-1)+i] - rightAnswer[i], 2);
        }
    }
    output /= neuronsConfig[layers-1];
    return output;
}

float Perceptron::BinaryCrossEntropyError(float clip)
{
    float output = 0;
    for (int i = 0; i < neuronsConfig[layers-1]; i++)
    {
        rightAnswer[i] = max(clip, min(rightAnswer[i], 1-clip));
        output += (neurons[neuronsConfig[0]*(layers-1)+i]*log10(rightAnswer[i]+clip)) + 
        (1-neurons[neuronsConfig[0]*(layers-1)+i]) + (1-rightAnswer[i]+clip);
    }
    output /= neuronsConfig[layers-1];
    return output;
}

float Perceptron::CategoricalCrossEntropyError()
{
    // Work in progress...
    float output = 0;
    for (int i = 0; i < neuronsConfig[layers-1]; i++)
    {
        // output += ;
    }
    output /= neuronsConfig[layers-1];
    return output;
}

float Perceptron::Train(CostFunction costFunction)
{
    cout << "Hello from the training function" << endl;
    CalculateNeurons();
    float cost = 0;
    switch (costFunction)
    {
        case MeanSquared:
            cost = MeanSquaredError();
            break;
        case MeanAbsolute:
            cost = MeanAbsoluteError();
            break;
        
        default:
            break;
    }
    return cost;
}

int Perceptron::Free()
{
    cudaFree(neurons);
    cudaFree(weights);
    cudaFree(neuronsConfig);
    return 0;
}

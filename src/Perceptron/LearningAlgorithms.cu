#include "SemkiAI.hpp"

__global__
void CalculateErrorKernel(double* neurons, double* weights, int layers, int* neuronsConfig, double* error, int* neuronsIndexes, int* weightsIndexes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double currentNeuron = 0.0;

    for (int i = layers-2; i >= index+1; i -= stride)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            currentNeuron = neurons[neuronsIndexes[i] + j];
            error[neuronsIndexes[i] + j] = 0.0;
            for (int k = 0; k < neuronsConfig[i+1]-1; k++)
            {
                error[neuronsIndexes[i] + j] += currentNeuron * (1 - currentNeuron) * 
                    (weights[weightsIndexes[neuronsIndexes[i] + j] + k] * error[neuronsIndexes[layers+1] + k]);
            }
        }
    }
}

__global__
void BackpropagationKernel(double* neurons, double* weights, int layers, int* neuronsConfig, double* error, double learningRate, int* neuronsIndexes, int* weightsIndexes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < layers-1; i += stride)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            for (int k = 0; k < neuronsConfig[i+1]-1; k++)
            {
                weights[weightsIndexes[neuronsIndexes[i] + j] + k] += learningRate * 
                    error[neuronsIndexes[i+1] + k] * neurons[neuronsIndexes[i] + j];
            }
        }
    }
}

// https://www.geeksforgeeks.org/backpropagation-in-neural-network/
int Perceptron::Backpropagation(CostFunction costFunction)
{
    if (learningRate == -1)
    {
        throw MyException("You must set learningRate to use Backpropagation!");
    }

    double* error = new double[neuronsIndexes[layers]];
    double currentNeuron = 0.0;
    for (int i = 0; i < neuronsConfig[layers-1]-1; i++)
    {
        currentNeuron = neurons[neuronsIndexes[layers-1]+i];
        error[neuronsIndexes[layers-1]+i] = currentNeuron * (1 - currentNeuron) * (rightAnswer[i] - currentNeuron);
    }
    if (useGPU)
    {
        cudaMallocManaged(&error, neuronsIndexes[layers]*sizeof(double));
        gpuBlocks = (layers + gpuThreads - 1) / gpuThreads;
        CalculateErrorKernel<<<gpuBlocks, gpuThreads>>>(neurons, weights, layers, neuronsConfig, error, neuronsIndexes, weightsIndexes);
    }
    else
    {
        for (int i = layers-2; i >= 0; i--)
        {
            for (int j = 0; j < neuronsConfig[i]; j++)
            {
                currentNeuron = neurons[neuronsIndexes[i] + j];
                error[neuronsIndexes[i] + j] = 0.0;
                for (int k = 0; k < neuronsConfig[i+1]; k++)
                {
                    error[neuronsIndexes[i] + j] += currentNeuron * (1 - currentNeuron) * 
                        (weights[weightsIndexes[neuronsIndexes[i] + j] + k] * error[neuronsIndexes[layers+1] + k]);
                }
            }
        }
    }

    if (useGPU)
    {
        BackpropagationKernel<<<gpuBlocks, gpuThreads>>>(neurons, weights, layers, neuronsConfig, error, learningRate, neuronsIndexes, weightsIndexes);
        cudaDeviceSynchronize();
    }
    else
    {
        for (int i = 0; i < layers-1; i++)
        {
            for (int j = 0; j < neuronsConfig[i]; j++)
            {
                for (int k = 0; k < neuronsConfig[i+1]-1; k++)
                {
                    weights[weightsIndexes[neuronsIndexes[i] + j] + k] += learningRate * 
                        error[neuronsIndexes[i+1] + k] * neurons[neuronsIndexes[i] + j];
                }
            }
        }
    }
    return 0;
}

int Perceptron::SimulatedAnnealing(ActivationFunction activationFunction, CostFunction costFunction)
{
    if (temperature == -1 || temperatureDecreaseRate == -1)
    {
        throw MyException("You must set temperature and temperatureDecreaseRate to use SimulatedAnnealing!");
    }
    
    Perceptron candidate = *this;
    for (int i = 0; i < temperature; i++)
    {
        candidate.weights[rand() % weightsIndexes[neuronsIndexes[layers-1]] - 1] = rand() % 1000 * 1.0 / 1000;
    }
    candidate.CalculateNeurons(activationFunction);
    
    CalculateCost(costFunction, layers-1);
    candidate.CalculateCost(costFunction, layers-1);

    if (cost < candidate.cost)
    {
        weights = candidate.weights;
    }
    else
    {
        double deltaCost = cost - candidate.cost;
        if ((rand() % 1000 * 1.0 / 1000) > exp(deltaCost / temperature))
        {
            weights = candidate.weights;
        }
    }
    
    temperature *= temperatureDecreaseRate;
    // std::cout << "Temperature: " << temperature << "\n";
    return 0;
}

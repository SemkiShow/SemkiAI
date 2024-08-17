#include "AI.cuh"

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_int_distribution<std::mt19937::result_type> dist1000(0, 1000);


class MyException : public std::exception { 
private: 
    std::string message; 
  
public: 
    // Constructor accepts a const char* that is used to set 
    // the exception message 
    MyException(const char* msg) 
        : message(msg) 
    { 
    } 
  
    // Override the what() method to return our message 
    const char* what() const throw() 
    { 
        return message.c_str(); 
    } 
};

// void add(double* a, double* b, double* c, int N)
// {    
//     int threads = 256;
//     int blocks = (N + threads - 1) / threads;
//     addKernel<<<blocks, threads>>>(a, b, c, N);
//     cudaDeviceSynchronize();
// }

int Perceptron::InitCuda()
{
    cudaMallocManaged(&neuronsConfig, layers*sizeof(int));
    // cudaMallocManaged(&neurons, layers*neuronsConfig[0]*sizeof(double));
    // cudaMallocManaged(&weights, layers*neuronsConfig[0]*(neuronsConfig[0]-1)*sizeof(double));
    std::cout << "cuda was initialized" << std::endl; 
    return 0;
}

int Perceptron::Init()
{
    // cudaMallocManaged(&neuronsConfig, layers*sizeof(int));

    cudaMallocManaged(&neurons, layers*neuronsConfig[0]*sizeof(double));
    // std::cout << layers*neuronsConfig[0] << std::endl;
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < neuronsConfig[0]; j++)
        {
            // std::cout << i*neuronsConfig[0]+j << std::endl;
            // std::cout << dist1000(rng) * 1.0 / 1000 << std::endl;
            neurons[i*neuronsConfig[0]+j] = dist1000(rng) * 1.0 / 1000;
            // std::cout << neurons[i*neuronsConfig[0]+j] << std::endl;
        }
    }
    std::cout << "Neurons were initialized" << std::endl;

    cudaMallocManaged(&weights, layers*neuronsConfig[0]*(neuronsConfig[0]-1)*sizeof(double));
    // int lastIndex = 0;
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            for (int k = 0; k < neuronsConfig[i]-1; k++)
            {
                // if (i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k - lastIndex != 1)
                // {
                //     std::cout << lastIndex << "->" << i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k << std::endl;
                // }
                // lastIndex = i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k;
                // std::cout << i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k << std::endl;
                weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] = dist1000(rng) * 1.0 / 1000;
            }
        }
    }
    std::cout << "Weights were initialized" << std::endl;
    return 0;
}

double Perceptron::Sigmoid(double input)
{
    return 1/(1+exp(-input));
}

double Perceptron::ReLU(double input)
{
    if (input > 0){return input;}
    else {return 0;}
}

double Perceptron::Tanh(double input)
{
    return tanh(input);
}

__global__
void CalculateNeuronsKernel(double* neurons, double* weights, int* neuronsConfig, int layers, int layer)
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

int Perceptron::CalculateNeurons(ActivationFunction activationFunction)
{
    for (int i = 0; i < layers-1; i++)
    {
        // std::cout << i << std::endl;
        gpuThreads = 256;
        gpuBlocks = (neuronsConfig[i] + gpuThreads - 1) / gpuThreads;
        CalculateNeuronsKernel<<<gpuBlocks, gpuThreads>>>(neurons, weights, neuronsConfig, layers, i);
        cudaDeviceSynchronize();
    }
    for (int i = neuronsConfig[0]*(layers-1); i < neuronsConfig[0]*layers; i++)
    {
        // std::cout << neurons[i] << std::endl;
        switch (activationFunction)
        {
            case ActivationFunction::Sigmoid:
                neurons[i] = Sigmoid(neurons[i]);
                break;
            case ActivationFunction::ReLU:
                neurons[i] = ReLU(neurons[i]);
                break;
            case ActivationFunction::Tanh:
                neurons[i] = Tanh(neurons[i]);
                break;
        
            default:
                break;
        }
        // std::cout << neurons[i] << std::endl;
    }
    
    std::cout << "Neurons were recalculated" << std::endl;
    return 0;
}

double Perceptron::MeanSquaredError(int layer)
{
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]; i++)
    {
        // std::cout << neurons[neuronsConfig[0]*layer+i] << " - " << rightAnswer[i] << " = " << neurons[neuronsConfig[0]*layer+i] - rightAnswer[i] << std::endl;
        // std::cout << output << " + " << pow(neurons[neuronsConfig[0]*layer+i] - rightAnswer[i], 2) << " = ";
        output += pow(neurons[neuronsConfig[0]*layer+i] - rightAnswer[i], 2);
        // std::cout << output << std::endl;
    }
    // std::cout << output << std::endl;
    // std::cout << neuronsConfig[layer] << std::endl;
    // std::cout << layer << std::endl;
    output /= neuronsConfig[layer];
    return output;
}

double Perceptron::MeanAbsoluteError(int layer)
{
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layers]; i++)
    {
        output += abs(neurons[neuronsConfig[0]*(layer)+i] - rightAnswer[i]);
    }
    output /= (1.0 * neuronsConfig[layer]);
    return output;
}

double Perceptron::HuberLoss(int layer/* double delta */)
{
    if (delta == -1)
    {
        throw MyException("You must set the delta variable to use HuberLoss!");
    }
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]; i++)
    {
        if (abs(neurons[neuronsConfig[0]*(layer)+i] - rightAnswer[i]) > delta)
        {
            output += delta * (abs(neurons[neuronsConfig[0]*(layer)+i] - rightAnswer[i]) - 0.5f * delta);
        }
        else
        {
            output += pow(neurons[neuronsConfig[0]*(layer)+i] - rightAnswer[i], 2);
        }
    }
    output /= neuronsConfig[layer];
    return output;
}

double Perceptron::BinaryCrossEntropyLoss(int layer/* double clip */)
{
    if (clip == -1)
    {
        throw MyException("You must set the clip variable to use BinaryCrossEntropyLoss!");
    }
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]; i++)
    {
        rightAnswer[i] = max(clip, min(rightAnswer[i], 1-clip));
        output += (neurons[neuronsConfig[0]*(layer)+i]*log10(rightAnswer[i]+clip)) + 
        (1-neurons[neuronsConfig[0]*(layer)+i]) + (1-rightAnswer[i]+clip);
    }
    output /= neuronsConfig[layer];
    return output;
}

double Perceptron::CategoricalCrossEntropyLoss(int layer)
{
    // Work in progress...
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]; i++)
    {
        // output += ;
    }
    output /= neuronsConfig[layer];
    return output;
}

__global__
void CalculateErrorKernel(double* neurons, int* neuronsConfig, double* rightAnswer, int layers, int i, double* error)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // int index = threadIdx.x;
    // int stride = blockDim.x;
    double tmp;

    for (int j = index; j < neuronsConfig[i]; j+=stride)
    {
        tmp = neurons[neuronsConfig[0]*(layers-1)+i];
        error[i*neuronsConfig[0]+j] = tmp*(1-tmp)*(rightAnswer[i]-tmp);
    }
}

__global__
void BackpropagationKernel(double* neurons, double* weights, int* neuronsConfig, double* error, double learningRate, int i, int j)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // int index = threadIdx.x;
    // int stride = blockDim.x;

    for (int k = index; k < neuronsConfig[i+1]; k+=stride)
    {
        weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] += 
        learningRate*neurons[neuronsConfig[0]*i+j]*error[k];
    }
}

int Perceptron::Backpropagation(CostFunction costFunction, double learningRate)
{
    double* error = new double[neuronsConfig[0]*layers];
    // double tmp;
    for (int i = 0; i < layers; i++)
    {
        gpuThreads = 256;
        gpuBlocks = (neuronsConfig[i] + gpuThreads - 1) / gpuThreads;
        CalculateErrorKernel<<<gpuBlocks, gpuThreads>>>(neurons, neuronsConfig, rightAnswer, layers, i, error);
        // for (int j = 0; j < neuronsConfig[i]; j++)
        // {
        //     tmp = neurons[neuronsConfig[0]*(layers-1)+i];
        //     error[i*neuronsConfig[0]+j] = tmp*(1-tmp)*(rightAnswer[i]-tmp);
        // }
    }
    cudaDeviceSynchronize();
    for (int i = layers-2; i > 0; i--)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            BackpropagationKernel<<<gpuBlocks, gpuThreads>>>(neurons, weights, neuronsConfig, error, learningRate, i, j);
            // for (int k = 0; k < neuronsConfig[i+1]; k++)
            // {
            //     weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] += 
            //     learningRate*neurons[neuronsConfig[0]*i+j]*error[k];
            // }
        }
    }
    cudaDeviceSynchronize();
    return 0;
}

int Perceptron::SimulatedAnnealing(CostFunction costFunction, double learningRate)
{
    if (temperature == -1 || temperatureDecreaseRate == -1)
    {
        throw MyException("You must set temperature and temperatureDecreaseRate to use SimulatedAnnealing!");
    }
    // Work in progress...
    return 0;
}

double Perceptron::Train(ActivationFunction activationFunction, CostFunction costFunction, LearningAlgorithm learningAlgorithm, double learningRate)
{
    // std::cout << "Hello from the training function" << std::endl;
    CalculateNeurons(activationFunction);
    Backpropagation(costFunction, learningRate);
    double cost = 0;
    switch (costFunction)
    {
        case CostFunction::MeanSquared:
            cost = MeanSquaredError(layers-1);
            break;
        case CostFunction::MeanAbsolute:
            cost = MeanAbsoluteError(layers-1);
            break;
        case CostFunction::Huber:
            cost = HuberLoss(layers-1);
            break;
        case CostFunction::BinaryCrossEntropy:
            cost = BinaryCrossEntropyLoss(layers-1);
            break;
        
        default:
            break;
    }
    std::cout << "Error is: " << cost << std::endl;
    return cost;
}

int Perceptron::Free()
{
    cudaFree(neurons);
    cudaFree(weights);
    cudaFree(neuronsConfig);
    return 0;
}

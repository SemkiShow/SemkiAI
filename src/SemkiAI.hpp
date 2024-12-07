#include <iostream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>

class Perceptron
{
    private:
    public:
        double* neurons;
        double* weights;
        enum class CostFunction {MeanSquared, MeanAbsolute, Huber, BinaryCrossEntropy, CategoricalCrossEntropy};
        enum class ActivationFunction {Sigmoid, ReLU, Tanh};
        enum class LearningAlgorithm {Backpropagation, SimulatedAnnealing};
        bool useGPU = true;
        int gpuThreads;
        int gpuBlocks;
        int layers;
        int* neuronsConfig;
        int maxNeurons = 0;
        double* rightAnswer;
        double cost = -1;
        double delta = -1;
        double clip = -1;
        double learningRate = -1;
        double temperature = -1;
        double temperatureDecreaseRate = -1;
        int InitCuda();
        int Init(bool confirm);

        double Sigmoid(double input);
        double ReLU(double input);
        double Tanh(double input);
        int CalculateNeurons(ActivationFunction activationFunction);

        double MeanSquaredError(int layer);
        double MeanAbsoluteError(int layer);
        double HuberLoss(int layer/* double delta */);
        double BinaryCrossEntropyLoss(int layer/* double clip */);
        double CategoricalCrossEntropyLoss(int layer);
        int CalculateCost(CostFunction costFunction, int layer);

        int Backpropagation(CostFunction costFunction);
        int SimulatedAnnealing(ActivationFunction activationFunction, CostFunction costFunction);
        double Train(ActivationFunction activationFunction, CostFunction costFunction, LearningAlgorithm leraningAlgorithm);
        
        int SaveWeights(std::string fileName);
        int LoadWeights(std::string fileName);
        // int TrainScore(int score, CostFunction costFunction);
        // int TrainGenerations(int generations, CostFunction costFunction);
        int Free();
};

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

int Perceptron::InitCuda()
{
    neuronsConfig = new int[layers];
    if (useGPU)
    {
        cudaMallocManaged(&neuronsConfig, layers*sizeof(int));
        std::cout << "Cuda was initialized" << std::endl; 
    }
    return 0;
}

int Perceptron::Init(bool confirm = true)
{
    for (int i = 0; i < layers; i++)
    {
        if (neuronsConfig[i] > maxNeurons) maxNeurons = neuronsConfig[i];
    }
    std::cout << maxNeurons << std::endl;
    if (confirm)
    {
        double spaceTaken = layers;
        spaceTaken /= 1024;
        spaceTaken *= maxNeurons;
        spaceTaken /= 1024;
        spaceTaken *= 8;
        spaceTaken /= 1024;
        spaceTaken *= maxNeurons-1;
        spaceTaken += spaceTaken/(maxNeurons-1);
        std::cout << "The neural network requires " << (spaceTaken) << " GiB of RAM. Continue? (Enter/Ctrl-C)" << std::endl;
        std::getchar();
    }
    neurons = new double[layers*maxNeurons];
    if (useGPU)
    {
        cudaMallocManaged(&neurons, layers*maxNeurons*sizeof(double));
    }
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < maxNeurons; j++)
        {
            neurons[i*maxNeurons+j] = 0.213580;
        }
    }
    std::cout << "Neurons were initialized" << std::endl;

    weights = new double[layers*maxNeurons*(maxNeurons-1)];
    if (useGPU)
    {
        cudaMallocManaged(&weights, layers*maxNeurons*(maxNeurons-1)*sizeof(double));
    }
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            for (int k = 0; k < neuronsConfig[i]-1; k++)
            {
                weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] = 0.21345;
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
void CalculateNeuronsKernel(double* neurons, double* weights, int* neuronsConfig, int layers, int i, int maxNeurons)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < neuronsConfig[i+1]; j+=stride)
    {
        neurons[maxNeurons*(i+1)+j] = 0;
        for (int k = 0; k < neuronsConfig[i]; k++)
        {
            neurons[maxNeurons*(i+1)+j] += neurons[maxNeurons*i+k] * weights[i*maxNeurons*(maxNeurons-1)+k*(maxNeurons-1)+j];
        }
    }
}

int Perceptron::CalculateNeurons(ActivationFunction activationFunction)
{
    for (int i = 0; i < layers-1; i++)
    {
        if (useGPU)
        {
            gpuThreads = 256;
            gpuBlocks = (neuronsConfig[i] + gpuThreads - 1) / gpuThreads;
            CalculateNeuronsKernel<<<gpuBlocks, gpuThreads>>>(neurons, weights, neuronsConfig, layers, i, maxNeurons);
            cudaDeviceSynchronize();
        }
        else
        {
            for (int j = 0; j < neuronsConfig[i+1]; j++)
            {
                neurons[maxNeurons*(i+1)+j] = 0;
                for (int k = 0; k < neuronsConfig[i]; k++)
                {
                    neurons[maxNeurons*(i+1)+j] += neurons[maxNeurons*i+k] * weights[i*maxNeurons*(maxNeurons-1)+k*(maxNeurons-1)+j];
                }
            }
        }
    }
    for (int i = maxNeurons*(layers-1); i < maxNeurons*layers; i++)
    {
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
    }
    
    std::cout << "Neurons were recalculated" << std::endl;
    return 0;
}

// __global__
// void MeanSquaredErrorKernel(double* neurons, int* neuronsConfig, double* rightAnswer, int layer, double output)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int i = index; i < neuronsConfig[layer]; i+=stride)
//     {
//         output += pow(neurons[maxNeurons*layer+i] - rightAnswer[i], 2);
//     }
// }

double Perceptron::MeanSquaredError(int layer)
{
    double output = 0.0;
    // gpuThreads = 256;
    // gpuBlocks = (neuronsConfig[layer] + gpuThreads - 1) / gpuThreads;
    // MeanSquaredErrorKernel<<<gpuBlocks, gpuThreads>>>(neurons, neuronsConfig, rightAnswer, layer, output);
    // cudaDeviceSynchronize();
    for (int i = 0; i < neuronsConfig[layer]; i++)
    {
        // std::cout << neurons[maxNeurons*layer+i] << " - " << rightAnswer[i] << " = " << neurons[maxNeurons*layer+i] - rightAnswer[i] << std::endl;
        // std::cout << output << " + " << pow(neurons[maxNeurons*layer+i] - rightAnswer[i], 2) << " = ";
        output += pow(neurons[maxNeurons*layer+i] - rightAnswer[i], 2);
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
        output += abs(neurons[maxNeurons*(layer)+i] - rightAnswer[i]);
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
        if (abs(neurons[maxNeurons*(layer)+i] - rightAnswer[i]) > delta)
        {
            output += delta * (abs(neurons[maxNeurons*(layer)+i] - rightAnswer[i]) - 0.5f * delta);
        }
        else
        {
            output += pow(neurons[maxNeurons*(layer)+i] - rightAnswer[i], 2);
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
        output += (neurons[maxNeurons*(layer)+i]*log10(rightAnswer[i]+clip)) + 
        (1-neurons[maxNeurons*(layer)+i]) + (1-rightAnswer[i]+clip);
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

int Perceptron::CalculateCost(CostFunction costFunction, int layer)
{
    switch (costFunction)
    {
        case CostFunction::MeanSquared:
            cost = MeanSquaredError(layer);
            break;
        case CostFunction::MeanAbsolute:
            cost = MeanAbsoluteError(layer);
            break;
        case CostFunction::Huber:
            cost = HuberLoss(layer);
            break;
        case CostFunction::BinaryCrossEntropy:
            cost = BinaryCrossEntropyLoss(layer);
            break;
        
        default:
            break;
    }
    return 0;
}

__global__
void CalculateErrorKernel(double* neurons, int* neuronsConfig, double* rightAnswer, int layers, int i, double* error, int maxNeurons)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double tmp;

    for (int j = index; j < neuronsConfig[i]; j+=stride)
    {
        tmp = neurons[maxNeurons*(layers-1)+i];
        error[i*maxNeurons+j] = tmp*(1-tmp)*(rightAnswer[i]-tmp);
    }
}

__global__
void BackpropagationKernel(double* neurons, double* weights, int* neuronsConfig, double* error, double learningRate, int i, int j, int maxNeurons)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < neuronsConfig[i+1]; k+=stride)
    {
        weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] += 
        learningRate*neurons[maxNeurons*i+j]*error[k];
    }
}

int Perceptron::Backpropagation(CostFunction costFunction)
{
    if (learningRate == -1)
    {
        throw MyException("You must set learningRate to use Backpropagation!");
    }

    double* error = new double[maxNeurons*layers];
    double tmp = 0.0;
    for (int i = 0; i < layers; i++)
    {
        if (useGPU)
        {
            gpuThreads = 256;
            gpuBlocks = (neuronsConfig[i] + gpuThreads - 1) / gpuThreads;
            CalculateErrorKernel<<<gpuBlocks, gpuThreads>>>(neurons, neuronsConfig, rightAnswer, layers, i, error, maxNeurons);
        }
        else
        {
            for (int j = 0; j < neuronsConfig[i]; j++)
            {
                tmp = neurons[maxNeurons*(layers-1)+i];
                error[i*maxNeurons+j] = tmp*(1-tmp)*(rightAnswer[i]-tmp);
            }
        }
    }
    if (useGPU)
    {
        cudaDeviceSynchronize();
    }
    for (int i = layers-2; i > 0; i--)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            if (useGPU)
            {
                BackpropagationKernel<<<gpuBlocks, gpuThreads>>>(neurons, weights, neuronsConfig, error, learningRate, i, j, maxNeurons);
            }
            else
            {
                for (int k = 0; k < neuronsConfig[i+1]; k++)
                {
                    weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] += 
                    learningRate*neurons[maxNeurons*i+j]*error[k];
                }
            }
        }
    }
    if (useGPU)
    {
        cudaDeviceSynchronize();
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
        candidate.weights[(int)(dist1000(rng) * (layers*maxNeurons*(maxNeurons-1) / 1000))] = dist1000(rng) * 1.0 / 1000;
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
        if ((dist1000(rng) * 1.0 / 1000) > exp(deltaCost / temperature))
        {
            weights = candidate.weights;
        }
    }
    
    temperature *= temperatureDecreaseRate;
    std::cout << "Temperature: " << temperature << std::endl;
    return 0;
}

double Perceptron::Train(ActivationFunction activationFunction, CostFunction costFunction, LearningAlgorithm learningAlgorithm)
{
    CalculateNeurons(activationFunction);
    switch (learningAlgorithm)
    {
        case LearningAlgorithm::Backpropagation:
            Backpropagation(costFunction);
            break;
        case LearningAlgorithm::SimulatedAnnealing:
            SimulatedAnnealing(activationFunction, costFunction);
            break;
        
        default:
            break;
    }
    CalculateCost(costFunction, layers-1);
    std::cout << "Error: " << cost << std::endl;
    return cost;
}

int Perceptron::Free()
{
    if (useGPU)
    {
        cudaFree(neurons);
        cudaFree(weights);
        cudaFree(neuronsConfig);        
    }
    else
    {
        free(neurons);
        free(weights);
        free(neuronsConfig);
    }
    return 0;
}

int Perceptron::SaveWeights(std::string fileName)
{
    std::fstream weightsFile;
    std::string path = "./weights/"+fileName+".csv";
    weightsFile.open(path, std::ios::out);
    weightsFile << layers << ",";
    for (int i = 0; i < layers; i++)
    {
        weightsFile << neuronsConfig[i] << ",";
    }
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            for (int k = 0; k < neuronsConfig[i]-1; k++)
            {
                weightsFile << weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] << ",";
            }
        }
        weightsFile << std::endl;
        std::cout << "Saving the weights..." << std::endl;
        double progress = i*1.0/layers*100;
        std::cout << "Progress: " << progress << "%" << std::endl;
    }
    Free();
    weightsFile.close();
    return 0;
}

int Perceptron::LoadWeights(std::string fileName)
{
    std::fstream weightsFile;
    std::string path = "../weights/"+fileName+".csv";
    weightsFile.open(path, std::ios::in);
    // Load the data into a temporary string
    std::string buf;
    std::vector<std::string> input;
    while (std::getline(weightsFile, buf))
    {
        input.push_back(buf);
    }
    weightsFile.close();
    // Split the data
    std::vector<std::string> data;
    data.push_back("");
    unsigned long long index = 0;
    for (int j = 0; j < input.size(); j++)
    {
        for (unsigned long long i = 0; i < input[j].size(); i++)
        {
            if (input[j][i] == ',')
            {
                data.push_back("");
                index++;
                continue;
            }
            data[index] += input[j][i];
        }
    }
    // Debug info
    // for (int i = 0; i < 1000; i++)
    // {
    //     std::cout << data[i] << ", ";
    // }
    // std::cout << std::endl;
    // Work with the collected data
    layers = stoi(data[0]);
    InitCuda();
    for (int i = 0; i < layers; i++)
    {
        neuronsConfig[i] = stoi(data[i+1]);
    }
    rightAnswer = new double[neuronsConfig[layers-1]];
    Init(false);
    return 0;
}

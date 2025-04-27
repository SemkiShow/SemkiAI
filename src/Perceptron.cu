#include "SemkiAI.hpp"
#include "Perceptron/CostFunctions.hpp"
#include "Perceptron/ActivationFunctions.hpp"
#include "Perceptron/LearningAlgorithms.hpp"

int Perceptron::InitCuda()
{
    std::cout << "Initialising CUDA...\n"; 
    neuronsConfig = new int[layers];
    if (useGPU)
    {
        cudaMallocManaged(&neuronsConfig, layers*sizeof(int));
    }
    return 0;
}

int Perceptron::Init(bool confirm, bool randomize)
{
    for (int i = 0; i < layers; i++)
    {
        neuronsConfig[i]++;
    }
    for (int i = 0; i < layers; i++)
    {
        if (neuronsConfig[i] > maxNeurons) maxNeurons = neuronsConfig[i];
    }
    // std::cout << maxNeurons << std::endl;
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
        std::cout << "The neural network requires " << (spaceTaken) << " GiB of RAM. Continue? (Enter/Ctrl-C)\n";
        std::getchar();
    }
    std::cout << "Initialising neurons...\n";
    srand(time(0));
    neurons = new double[layers*maxNeurons];
    if (useGPU)
    {
        cudaMallocManaged(&neurons, layers*maxNeurons*sizeof(double));
    }
    if (randomize)
    {
        for (int i = 0; i < layers; i++)
        {
            for (int j = 0; j < maxNeurons; j++)
            {
                neurons[i*maxNeurons+j] = rand() % 1000 * 1.0 / 1000;
            }
            neurons[i*maxNeurons+(maxNeurons-1)] = 1;
        }
    }

    std::cout << "Initialising weights...\n";
    weights = new double[layers*maxNeurons*(maxNeurons-1)];
    if (useGPU)
    {
        cudaMallocManaged(&weights, layers*maxNeurons*(maxNeurons-1)*sizeof(double));
    }
    if (randomize)
    {
        for (int i = 0; i < layers; i++)
        {
            for (int j = 0; j < neuronsConfig[i]; j++)
            {
                for (int k = 0; k < neuronsConfig[i]-1; k++)
                {
                    weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] = rand() % 1000 * 1.0 / 1000;
                }
            }
        }
    }
    return 0;
}

__global__
void CalculateNeuronsKernel(double* neurons, double* weights, int* neuronsConfig, int layers, int i, int maxNeurons)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < neuronsConfig[i+1]-1; j+=stride)
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
            gpuThreads = 1256;
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
        neurons[i*maxNeurons+(maxNeurons-1)] = 1;
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
    return 0;
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
    // std::cout << "Error: " << cost << std::endl;
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
    std::string path = "weights/" + fileName;
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
        // weightsFile << std::endl;
        std::cout << '\r' << "Saving the weights...    " << i * 1000 / layers / 10.0 << "%" << std::flush;
    }
    std::cout << '\n';
    weightsFile.close();
    Free();
    return 0;
}

int Perceptron::LoadWeights(std::string fileName)
{
    std::cout << "Processing the weights file...\n";
    std::fstream weightsFile;
    std::string path = "weights/"+fileName;
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
    input.clear();
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
        neuronsConfig[i] = stoi(data[i+1])-1;
    }
    rightAnswer = new double[neuronsConfig[layers-1]];
    Init(false, false);
    for (unsigned long i = layers+2; i < data.size()-1; i++)
    {
        weights[i] = stod(data[i]);
    }
    data.clear();
    return 0;
}

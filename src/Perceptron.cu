#include "SemkiAI.hpp"

double Perceptron::GetRAMRequirements()
{
    double spaceTaken = weightsIndexes[neuronsIndexes[layers-1]] * sizeof(double) * 1.0 / 1024 / 1024 / 1024;
    return spaceTaken;
}

__global__
void CheckCudaKernel(double* a, double* b, double* c, int length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < length; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

int Perceptron::InitCuda()
{
    std::cout << "Initialising CUDA...\n";
    // Checking if CUDA is available
    if (useGPU)
    {
        int length = 1000;
        double* a = new double[length];
        double* b = new double[length];
        double* c = new double[length];

        cudaMallocManaged(&a, length * sizeof(double));
        cudaMallocManaged(&b, length * sizeof(double));
        cudaMallocManaged(&c, length * sizeof(double));

        for (int i = 0; i < length; i++)
        {
            a[i] = 1.0;
            b[i] = 2.0;
            c[i] = 0.0;
        }

        gpuBlocks = (length + gpuThreads - 1) / gpuThreads;
        CheckCudaKernel<<<gpuBlocks, gpuThreads>>>(a, b, c, length);
        cudaDeviceSynchronize();

        if (c[0] != 3.0) throw MyException("CUDA is not available! You must set useGPU to false!");
    }

    // Initialising neuronsConfig
    neuronsConfig = new int[layers];
    if (useGPU)
        cudaMallocManaged(&neuronsConfig, layers*sizeof(int));
    return 0;
}

int Perceptron::Init(bool confirm, bool randomize)
{
    // Adding biases
    for (int i = 0; i < layers; i++)
        neuronsConfig[i]++;

    // Initailising neuronsIndexes
    neuronsIndexes = new int[layers+1];
    if (useGPU)
        cudaMallocManaged(&neuronsIndexes, (layers+1)*sizeof(int));
    neuronsIndexes[0] = 0;
    for (int i = 1; i < layers; i++)
        neuronsIndexes[i] = neuronsIndexes[i-1] + neuronsConfig[i-1];
    neuronsIndexes[layers] = neuronsIndexes[layers-1] + neuronsConfig[layers-1];

    // Initialising weightsIndexes
    weightsIndexes = new int[neuronsIndexes[layers-1]+1];
    if (useGPU)
        cudaMallocManaged(&weightsIndexes, (neuronsIndexes[layers-1]+1)*sizeof(int));
    for (int i = 0; i < layers-1; i++)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            if (i == 0 && j == 0)
                weightsIndexes[0] = 0;
            else
                weightsIndexes[neuronsIndexes[i] + j] = weightsIndexes[neuronsIndexes[i] + j - 1] + neuronsConfig[i+1];
        }
    }
    weightsIndexes[neuronsIndexes[layers-1]] = weightsIndexes[neuronsIndexes[layers-1] - 1] + neuronsConfig[layers-1];

    if (confirm)
    {
        std::cout << "The neural network requires " << GetRAMRequirements() << " GiB of RAM. Continue? (Enter/Ctrl-C)\n";
        std::getchar();
    }
    
    // Initialising neurons
    std::cout << "Initialising neurons...\n";
    srand(time(0));
    neurons = new double[neuronsIndexes[layers]];
    if (useGPU)
        cudaMallocManaged(&neurons, neuronsIndexes[layers]*sizeof(double));
    for (int i = 0; i < layers; i++)
    {
        for (int j = 0; j < neuronsConfig[i]-1; j++)
        {
            if (randomize)
                neurons[neuronsIndexes[i] + j] = rand() % 1000 * 1.0 / 1000;
            else
                neurons[neuronsIndexes[i] + j] = 1;
        }
        neurons[neuronsIndexes[i] + neuronsConfig[i]-1] = 1;
    }

    // Initialising weights
    std::cout << "Initialising weights...\n";
    weights = new double[weightsIndexes[neuronsIndexes[layers-1]]];
    if (useGPU)
        cudaMallocManaged(&weights, weightsIndexes[neuronsIndexes[layers-1]]*sizeof(double));
    for (int i = 0; i < layers-1; i++)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            for (int k = 0; k < neuronsConfig[i+1]-1; k++)
            {
                if (randomize)
                    weights[weightsIndexes[neuronsIndexes[i] + j] + k] = rand() % 1000 * 1.0 / 1000;
                else
                    weights[weightsIndexes[neuronsIndexes[i] + j] + k] = 1;
            }
        }
    }
    return 0;
}

__global__
void CalculateNeuronsKernel(double* neurons, double* weights, int* neuronsConfig, int layers, int* neuronsIndexes, int* weightsIndexes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < layers-1; i += stride)
    {
        for (int j = 0; j < neuronsConfig[i+1]-1; j++)
        {
            neurons[neuronsIndexes[i+1] + j] = 0;
            for (int k = 0; k < neuronsConfig[i]; k++)
            {
                neurons[neuronsIndexes[i+1] + j] += neurons[neuronsIndexes[i] + k] * 
                    weights[weightsIndexes[neuronsIndexes[i] + k] + j];
            }
        }
    }
}

int Perceptron::CalculateNeurons(ActivationFunction activationFunction)
{
    if (useGPU)
    {
        gpuBlocks = (layers-1 + gpuThreads - 1) / gpuThreads;
        CalculateNeuronsKernel<<<gpuBlocks, gpuThreads>>>(neurons, weights, neuronsConfig, layers, neuronsIndexes, weightsIndexes);
        cudaDeviceSynchronize();
    }
    else
    {
        for (int i = 0; i < layers-1; i++)
        {
            for (int j = 0; j < neuronsConfig[i+1]-1; j++)
            {
                neurons[neuronsIndexes[i+1] + j] = 0;
                for (int k = 0; k < neuronsConfig[i]; k++)
                {
                    neurons[neuronsIndexes[i+1] + j] += neurons[neuronsIndexes[i] + k] * 
                        weights[weightsIndexes[neuronsIndexes[i] + k] + j];
                }
            }
        }
    }
    for (int i = 1; i < layers; i++)
    {
        for (int j = 0; j < neuronsConfig[i]-1; j++)
        {
            switch (activationFunction)
            {
                case ActivationFunction::Sigmoid:
                    neurons[neuronsIndexes[i] + j] = Sigmoid(neurons[neuronsIndexes[i] + j]);
                    break;
                case ActivationFunction::ReLU:
                    neurons[neuronsIndexes[i] + j] = ReLU(neurons[neuronsIndexes[i] + j]);
                    break;
                case ActivationFunction::Tanh:
                    neurons[neuronsIndexes[i] + j] = Tanh(neurons[neuronsIndexes[i] + j]);
                    break;
                default:
                    break;
            }
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
    weightsFile << layers << "\n";
    for (int i = 0; i < layers; i++)
    {
        weightsFile << neuronsConfig[i] << "\n";
    }
    for (int i = 0; i < layers-1; i++)
    {
        for (int j = 0; j < neuronsConfig[i]; j++)
        {
            for (int k = 0; k < neuronsConfig[i+1]-1; k++)
            {
                weightsFile << weights[weightsIndexes[neuronsIndexes[i] + j] + k] << "\n";
            }
        }
        std::cout << '\r' << "Saving the weights...    " << i * 1000 / layers / 10.0 << "%" << std::flush;
    }
    std::cout << '\n';
    weightsFile.close();
    Free();
    return 0;
}

int Perceptron::LoadWeights(std::string fileName)
{
    // Open the weights file
    std::cout << "Processing the weights file...\n";
    std::fstream weightsFile;
    std::string path = "weights/"+fileName;
    weightsFile.open(path, std::ios::in);

    // Load and process the data
    std::string buf = "";
    int counter = 0;
    while (std::getline(weightsFile, buf))
    {
        if (counter == 0)
        {
            layers = stoi(buf);
            InitCuda();
        }
        if (counter > 0 && counter < layers+1)
        {
            neuronsConfig[counter-1] = stoi(buf)-1;
        }
        if (counter == layers+1)
            Init(false, false);
        if (counter >= layers+1)
        {
            weights[counter-layers-1] = stod(buf);
        }
        counter++;
    }
    weightsFile.close();
    return 0;
}

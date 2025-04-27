__global__
void CalculateErrorKernel(double* neurons, int* neuronsConfig, double* rightAnswer, int layers, int i, double* error, int maxNeurons)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double currentNeuron;

    for (int j = index; j < neuronsConfig[i]-1; j+=stride)
    {
        currentNeuron = neurons[maxNeurons*(layers-1)+i];
        error[i*maxNeurons+j] = currentNeuron*(1-currentNeuron)*(rightAnswer[i]-currentNeuron);
    }
}

__global__
void BackpropagationKernel(double* neurons, double* weights, int* neuronsConfig, double* error, double learningRate, int i, int j, int maxNeurons)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < neuronsConfig[i+1]-1; k+=stride)
    {
        weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] += learningRate*neurons[maxNeurons*i+j]*error[k];
    }
}

int Perceptron::Backpropagation(CostFunction costFunction)
{
    if (learningRate == -1)
    {
        throw MyException("You must set learningRate to use Backpropagation!");
    }

    double* error = new double[maxNeurons*layers];
    double currentNeuron = 0.0;
    for (int i = 0; i < layers; i++)
    {
        if (useGPU)
        {
            // cudaMallocManaged(&error, maxNeurons*layers*sizeof(double));
            gpuThreads = 256;
            gpuBlocks = (neuronsConfig[i] + gpuThreads - 1) / gpuThreads;
            CalculateErrorKernel<<<gpuBlocks, gpuThreads>>>(neurons, neuronsConfig, rightAnswer, layers, i, error, maxNeurons);
        }
        else
        {
            for (int j = 0; j < neuronsConfig[i]; j++)
            {
                currentNeuron = neurons[maxNeurons*(layers-1)+i];
                error[i*maxNeurons+j] = currentNeuron*(1-currentNeuron)*(rightAnswer[i]-currentNeuron);
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
                    weights[i*neuronsConfig[i]*(neuronsConfig[i]-1)+j*(neuronsConfig[i]-1)+k] += learningRate*neurons[maxNeurons*i+j]*error[k];
                }
            }
        }
    }
    if (useGPU)
    {
        cudaDeviceSynchronize();
    }
    // cudaFree(error);
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
        candidate.weights[(int)(rand() % 1000 * (layers*maxNeurons*(maxNeurons-1) / 1000))] = rand() % 1000 * 1.0 / 1000;
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
    // std::cout << "Temperature: " << temperature << std::endl;
    return 0;
}

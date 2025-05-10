#include <iostream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <filesystem>

class MyException : public std::exception
{ 
    private: 
        std::string message; 
    
    public: 
        // Constructor accepts a const char* that is used to set 
        // the exception message 
        MyException(const char* msg) : message(msg) {} 
    
        // Override the what() method to return our message 
        const char* what() const throw() 
        { 
            return message.c_str(); 
        }
};

class Perceptron
{
    private:
    public:
        double* neurons;
        double* weights;
        enum class CostFunction {MeanSquared, MeanAbsolute, Huber, BinaryCrossEntropy};
        enum class ActivationFunction {Sigmoid, ReLU, Tanh};
        enum class LearningAlgorithm {Backpropagation, SimulatedAnnealing};
        bool useGPU = true;
        int gpuThreads = (int)pow(2, 10);
        int gpuBlocks;
        int layers;
        int* neuronsConfig;
        // Usage: neurons[neuronsIndexes[layer] + neuronIndex]
        int* neuronsIndexes;
        // Usage: weights[weightsIndexes[neuronsIndexes[sourceNeuronLayer] + sourceNeuronIndex] + destinationNeuronIndex]
        int* weightsIndexes;
        double* rightAnswer;
        double cost = -1;
        double delta = -1;
        double clip = -1;
        double learningRate = -1;
        double temperature = -1;
        double temperatureDecreaseRate = -1;
        double GetRAMRequirements();
        int InitCuda();
        int Init(bool confirm = true, bool randomize = true);

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
        int Free();
};

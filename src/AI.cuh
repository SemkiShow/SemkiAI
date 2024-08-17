#include <iostream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
// #include <hip/hip_runtime.h>
// using namespace std;

void test();
// __global__ void addKernel(double* a, double* b, double* c, int N);
// void add(double* a, double* b, double* c, int N);
class Perceptron
{
    private:
        double* neurons;
        double* weights;
    public:
        enum class CostFunction {MeanSquared, MeanAbsolute, Huber, BinaryCrossEntropy, CategoricalCrossEntropy};
        enum class ActivationFunction {Sigmoid, ReLU, Tanh};
        enum class LearningAlgorithm {Backpropagation, SimulatedAnnealing};
        int gpuThreads;
        int gpuBlocks;
        int* neuronsConfig;
        int layers;
        double* rightAnswer;
        double delta = -1;
        double clip = -1;
        double temperature = -1;
        double temperatureDecreaseRate = -1;
        int InitCuda();
        int Init();
        double Sigmoid(double input);
        double ReLU(double input);
        double Tanh(double input);
        int CalculateNeurons(ActivationFunction activationFunction);
        double MeanSquaredError(int layer);
        double MeanAbsoluteError(int layer);
        double HuberLoss(int layer/* double delta */);
        double BinaryCrossEntropyLoss(int layer/* double clip */);
        double CategoricalCrossEntropyLoss(int layer);
        int Backpropagation(CostFunction costFunction, double learningRate);
        int SimulatedAnnealing(CostFunction costFunction, double learningRate);
        double Train(ActivationFunction activationFunction, CostFunction costFunction, LearningAlgorithm leraningAlgorithm, double learningRate);
        // int TrainScore(int score, CostFunction costFunction);
        // int TrainGenerations(int generations, CostFunction costFunction);
        int Free();
};

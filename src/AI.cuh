#include <iostream>
#include <random>
#include <string>
#include <cmath>
using namespace std;

void test();
// __global__ void addKernel(float* a, float* b, float* c, int N);
// void add(float* a, float* b, float* c, int N);
class Perceptron
{
    private:
        float* neurons;
        float* weights;
    public:
        enum CostFunction {MeanSquared, MeanAbsolute};
        int* neuronsConfig;
        int layers;
        float* rightAnswer;
        int Init();
        int CalculateNeurons();
        float MeanSquaredError();
        float MeanAbsoluteError();
        float HuberLoss(float delta);
        float BinaryCrossEntropyError(float clip);
        float CategoricalCrossEntropyError();
        float Train(CostFunction costFunction);
        // int TrainScore(int score, CostFunction costFunction);
        // int TrainGenerations(int generations, CostFunction costFunction);
        int Free();
};

#include "SemkiAI.hpp"

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

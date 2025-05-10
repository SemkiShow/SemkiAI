#include "SemkiAI.hpp"

double Perceptron::Sigmoid(double input)
{
    return 1/(1+exp(-input));
}

double Perceptron::ReLU(double input)
{
    return std::min(std::max(0.0, input), 10.0);
}

double Perceptron::Tanh(double input)
{
    return tanh(input);
}

#include "SemkiAI.hpp"

double Perceptron::MeanSquaredError(int layer)
{
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]-1; i++)
    {
        output += pow(neurons[neuronsIndexes[layer] + i] - rightAnswer[i], 2);
    }
    return output;
}

double Perceptron::MeanAbsoluteError(int layer)
{
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]-1; i++)
    {
        output += abs(neurons[neuronsIndexes[layer] + i] - rightAnswer[i]);
    }
    output /= (1.0 * neuronsConfig[layer]);
    return output;
}

double Perceptron::HuberLoss(int layer)
{
    if (delta == -1)
    {
        throw MyException("You must set the delta variable to use HuberLoss!");
    }
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]-1; i++)
    {
        if (abs(neurons[neuronsIndexes[layer] + i] - rightAnswer[i]) > delta)
            output += delta * (abs(neurons[neuronsIndexes[layer] + i] - rightAnswer[i]) - 0.5 * delta);
        else
            output += pow(neurons[neuronsIndexes[layer] + i] - rightAnswer[i], 2);
    }
    output /= neuronsConfig[layer];
    return output;
}

double Perceptron::BinaryCrossEntropyLoss(int layer)
{
    if (clip == -1)
    {
        throw MyException("You must set the clip variable to use BinaryCrossEntropyLoss!");
    }
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]-1; i++)
    {
        rightAnswer[i] = max(clip, min(rightAnswer[i], 1-clip));
        output += (neurons[neuronsIndexes[layer] + i] * log10(rightAnswer[i] + clip)) + 
        (1 - neurons[neuronsIndexes[layer] + i]) + (1 - rightAnswer[i] + clip);
    }
    output /= neuronsConfig[layer];
    return output;
}

double Perceptron::MeanSquaredError(int layer)
{
    double output = 0.0;
    for (int i = 0; i < neuronsConfig[layer]; i++)
    {
        output += pow(neurons[maxNeurons*layer+i] - rightAnswer[i], 2);
    }
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

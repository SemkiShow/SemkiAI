#include "SemkiAI.hpp"
#include "DatasetParser.hpp"

int main()
{
    /* Initialisation */
    // Initialising the perceptron
    Perceptron perceptron;
    // Use the GPU
    perceptron.useGPU = true;
    // Set the amount of layers
    perceptron.layers = 123;
    // Init CUDA
    perceptron.InitCuda();
    // Set the amount of neurons in each layer
    perceptron.neuronsConfig[0] = 28*28;
    for (int i = 1; i < perceptron.layers-1; i++)
    {
        perceptron.neuronsConfig[i] = 28*28;
    }
    perceptron.neuronsConfig[perceptron.layers-1] = 10;
    // Init the right answers array
    perceptron.rightAnswer = new double[perceptron.neuronsConfig[perceptron.layers-1]];
    // Init random
    srand(time(0));
    // Init neurons and weights
    perceptron.Init();
    // Set the required variables for SimulatedAnnealing
    perceptron.temperature = 5000;
    perceptron.temperatureDecreaseRate = 0.99;
    // Set the learning rate
    perceptron.learningRate = 0.1;
    // Init the training info variables
    double initialError = 10;
    double endError = 10;
    int maxIterations = 1234;
    int iteration = 0;
    double acceptableError = 0.01;
    // An error buffer
    double buf = 0;

    /* Training */
    int currentNumber = -1;
    int8_t r,g,b,a;
    unsigned int color;
    while (endError > acceptableError && iteration < maxIterations)
    {
        // Print the current iteration number
        std::cout << '\r' << "Iteration: " << iteration << ", Current error: " << endError;
        currentNumber = rand() % 10;
        pngimage_t inputImage;
        inputImage = read_png_file("dataset/MNIST/"+std::to_string(currentNumber)+"/"+std::to_string(rand()%10000)+".png");

        int x = 0;
        int y = 0;
        for (int i = 0; i < perceptron.neuronsConfig[0]; i++)
        {
            if (x >= 28)
            {
                x = 0;
                y++;
            }
            color = inputImage.get(x, y);
            getRGBAFromColor(color, &r, &g, &b, &a);
            perceptron.neurons[i] = (uint8_t)a * 1.0 / 255;
            x++;
        }
        // Set the right answer
        for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
        {
            perceptron.rightAnswer[i] = i == currentNumber ? 1 : 0;
        }
        // Run a training cycle
        buf = perceptron.Train(
            Perceptron::ActivationFunction::ReLU,
            Perceptron::CostFunction::MeanSquared, 
            Perceptron::LearningAlgorithm::Backpropagation);
        // Remember the error
        if (iteration == 0) initialError = buf; else endError = buf;
        // Increase the iterations counter
        iteration++;
    }
    // Print the error stats
    std::cout << std::endl;
    std::cout << "Initial error was " << initialError << std::endl;
    std::cout << "Now the error is " << endError << std::endl;

    // Save the trained model to a .csv file
    perceptron.SaveWeights("weights.csv");
    return 0;
}

# SemkiAI
## How to use
### Perceptron
1. Create an instance of the Perceptron class ```Perceptron perceptron;```
2. Set the amount of layers ```perceptron.layers = 6;```
3. Initialize CUDA ```perceptron.InitCuda();```
4. Initialize neuronsConfig. The first layer must contain the biggest amount of neurons ```perceptron.neuronsConfig[i] = 123;```
5. Initialize the perceptron ```perceptron.Init();```
6. Run a training cycle
   1. Set the right answer ```perceptron.rightAnswer[i] = 1.0;```
   2. Set the [required variables](#required-variables) for your chosen training function. Example for Backpropagation: ```perceptron.learningRate = 1.0;```
   3. Call the Train function ```perceptron.Train(ActivationFunction, CostFunction, LearningAlgorithm);```
7. Save your weights to a file. The weights are located in the weights directory ```perceptron.SaveWeights("NameOfTheWeightsFile");```
8. Free the memory taken by neurons and weights ```perceptron.Free();```

#### Required variables
| Training function | Required variables |
| --- | --- |
| Backpropagation | learningRate |
| SimulatedAnnealing | temperature, temperatureDecreaseRate |

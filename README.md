# SemkiAI
## How to compile
0. You must have a CUDA-capable NVIDIA GPU to run this library or set the useGPU variable to false
1. Install the latest CUDA SDK
2. Run compile.sh
3. Run the compiled binary manually. The compiled binaries are located in examples/*/build/
## How to use
See the examples in the examples/ directory
<!-- 1. Create an instance of the Perceptron class ```Perceptron perceptron;```
2. Set the amount of layers ```perceptron.layers = 6;```
3. Set the useGPU variable to false if you don't have a CUDA-capable NVIDIA GPU ```perceptron.useGPU = false;```
4. Initialize CUDA ```perceptron.InitCuda();```
5. Initialize neuronsConfig.
   ```c++
   for (int i = 0; i < perceptron.layers; i++)
   {
      perceptron.neuronsConfig[i] = 123;
   }
   ```
6. Initialize the perceptron ```perceptron.Init();```
7. Run a training cycle
   1. Set the right answer 
   ```c++
   double rightAnswer[perceptron.neuronsConfig[perceptron.layers-1]];
   perceptron.rightAnswer = rightAnswer;
   for (int i = 0; i < perceptron.neuronsConfig[perceptron.layers-1]; i++)
   {
      perceptron.rightAnswer[i] = 0.27158953;
   }
   ```
   2. Set the [required variables](#required-variables) for your chosen training function. Example for Backpropagation: ```perceptron.learningRate = 1.0;```
   3. Call the Train function (See also: [Activation functions](#activation-functions), [Cost functions](#cost-functions), [Learning algorithms](#learning-algorithms)) ```perceptron.Train(ActivationFunction, CostFunction, LearningAlgorithm);```
8. Save your weights to a file. The weights are located in the weights directory ```perceptron.SaveWeights("NameOfTheWeightsFile");``` -->

#### Required variables
| Training function | Required variables |
| --- | --- |
| Backpropagation | learningRate |
| SimulatedAnnealing | temperature, temperatureDecreaseRate |
#### Activation functions
Sigmoid, ReLU, Tanh
#### Cost functions
MeanSquared, MeanAbsolute, Huber, BinaryCrossEntropy, CategoricalCrossEntropy
#### Learning algorithms
Backpropagation, SimulatedAnnealing

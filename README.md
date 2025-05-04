# SemkiAI

## How to compile

0. You must have a CUDA-capable NVIDIA GPU to run this library or set the useGPU variable to false
1. Install the latest CUDA SDK
2. Download the latest release. **The main branch is not stable!**
3. Run compile_examples.sh
4. Run the compiled binary manually. The compiled binaries are located in examples/*/build/

## How to use

See the examples in the examples/ directory

### generic_usage

This example shows how to use the library in a generic way. Useless for any practical purposes without modifications.

### handwritten_numbers

This is a classic neural networks example. A neural network that recognises handwritten digits.

### utils/png_to_csv/png_to_csv.cpp

This is a utility program for converting png datasets to csv. It can be useful for the handwritten_numbers example to convert the MNIST dataset to csv.

A huge thanks to [Tadeusz Puźniakowski](https://github.com/pantadeusz) for [pnglibcpp](https://github.com/pantadeusz/pnglibcpp)

**This program depends on [libpng](http://www.libpng.org/pub/png/libpng.html) and zlib**

Install on Debian-based distributions:
```bash
sudo apt-get install libpng-dev zlib1g-dev
```

### utils/DatasetParser

This is a library for parsing datasets. Currently supports parsing CSV files.

## Perceptron

### Required variables

| Training function | Required variables |
| --- | --- |
| Backpropagation | learningRate |
| SimulatedAnnealing | temperature, temperatureDecreaseRate |

### Activation functions

Sigmoid, ReLU, Tanh

### Cost functions

MeanSquared, MeanAbsolute, Huber, BinaryCrossEntropy, CategoricalCrossEntropy

### Learning algorithms

Backpropagation, SimulatedAnnealing

if [[ $1 == "generic_usage" || $1 == "" ]]; then
    mkdir examples/geneic_usage/build
    mkdir examples/generic_usage/weights
    if [[ $2 == "training" || $2 == "" ]]; then
        cd examples/generic_usage
        nvcc -o build/training examples/generic_usage/training.cu
        cd ../..
    fi
    if [[ $2 == "using" || $2 == "" ]]; then
        cd examples/generic_usage
        nvcc -o build/using using.cu
        cd ../..
    fi
fi
if [[ $1 == "handwritten_numbers" || $1 == "" ]]; then
    mkdir examples/handwritten_numbers/build
    mkdir examples/handwritten_numbers/weights
    if [[ $2 == "training" || $2 == "" ]]; then
        cd examples/handwritten_numbers
        nvcc -c training.cu -o build/training.o -I./lib/include
        nvcc build/training.o build/pnglib.o -lpng -o build/training
        cd ../..
    fi
    if [[ $2 == "using" || $2 == "" ]]; then
        cd examples/handwritten_numbers
        nvcc -c using.cu -o build/using.o -I./lib/include
        cd build
        g++ -ggdb -I../lib/include ../lib/pnglib.cpp -c
        cd ..
        # g++ -c UI.cpp -o build/UI.o -I./lib/include
        # nvcc build/using.o build/UI.o build/pnglib.o -lpng -o build/using
        nvcc build/using.o build/pnglib.o -lpng -o build/using
        cd ../..
    fi
    if [[ $2 == "UI" || $2 == "" ]]; then
        cd examples/handwritten_numbers
        g++ -c UI.cpp -o build/UI.o -I./lib/include
        g++ build/UI.o build/pnglib.o -lpng -o build/UI
        cd ../..
    fi
fi

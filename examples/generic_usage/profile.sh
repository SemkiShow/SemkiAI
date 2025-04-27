#!/bin/bash

set -e

if [ ! -d build ]; then
    mkdir build
fi
cd build
cmake -DCMAKE_CXX_FLAGS=-pg ..
make
cd ..

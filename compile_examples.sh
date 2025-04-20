#!/bin/bash

set -e

cd examples/generic_usage/
./compile.sh
cd ../..

cd examples/handwritten_numbers/
./compile.sh
cd ../..

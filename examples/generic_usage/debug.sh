#!/bin/bash

set -e

if [ ! -d debug ]; then
    mkdir debug
fi
cd debug
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
cd ..

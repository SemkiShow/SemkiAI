./compile.sh
cd examples/$1/build
gdb -ex run ./$2
cd ../../..

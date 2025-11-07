#!/bin/bash

# Simple test script for quick verification

echo "Testing compilation..."

# Compile all programs
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""
echo "To run tests, use:"
echo "  mpirun -np 4 ./task1_monte_carlo_pi 1000000"
echo "  mpirun -np 4 ./task2_matrix_vector 100 r"
echo "  mpirun -np 4 ./task3_cannon_matrix_mult 100"
echo "  mpirun -np 4 ./task4_dirichlet 50 0.0"


#!/bin/bash

# Script to run experiments for all MPI tasks

echo "========================================="
echo "Running MPI Experiments"
echo "========================================="

# Clean previous results
rm -f task*.txt

# Task 1: Monte Carlo π calculation
echo ""
echo "Task 1: Monte Carlo π Calculation"
echo "-----------------------------------"
for processes in 1 2 4 8; do
    for points in 1000000 10000000 100000000; do
        echo "Running with $processes processes, $points points..."
        mpirun -np $processes ./task1_monte_carlo_pi $points > /dev/null 2>&1
    done
done

# Task 2: Matrix-vector multiplication
echo ""
echo "Task 2: Matrix-Vector Multiplication"
echo "-----------------------------------"
for method in r c b; do
    method_name=""
    case $method in
        r) method_name="Row" ;;
        c) method_name="Column" ;;
        b) method_name="Block" ;;
    esac
    echo "Method: $method_name"
    
    for processes in 1 4 9 16; do
        # For block partitioning, need perfect square
        if [ "$method" == "b" ] && [ "$processes" != "1" ] && [ "$processes" != "4" ] && [ "$processes" != "9" ] && [ "$processes" != "16" ]; then
            continue
        fi
        
        for size in 100 500 1000; do
            # Check if size is divisible by sqrt(processes) for block method
            if [ "$method" == "b" ]; then
                grid_size=$(echo "sqrt($processes)" | bc)
                if [ $((size % grid_size)) -ne 0 ]; then
                    continue
                fi
            fi
            
            echo "  Running $method_name with $processes processes, size $size..."
            mpirun -np $processes ./task2_matrix_vector $size $method > /dev/null 2>&1
        done
    done
done

# Task 3: Cannon's algorithm
echo ""
echo "Task 3: Cannon's Matrix Multiplication"
echo "-----------------------------------"
for processes in 1 4 9 16; do
    grid_size=$(echo "sqrt($processes)" | bc)
    for size in 300 600 900; do
        # Check if size is divisible by grid_size
        if [ $((size % grid_size)) -eq 0 ]; then
            echo "Running with $processes processes, size $size..."
            mpirun -np $processes ./task3_cannon_matrix_mult $size > /dev/null 2>&1
        fi
    done
done

# Task 4: Dirichlet problem
echo ""
echo "Task 4: Dirichlet Problem"
echo "-----------------------------------"
for processes in 1 4 9 16; do
    grid_size=$(echo "sqrt($processes)" | bc)
    for n in 50 100 200; do
        # Check if (n-2) is divisible by grid_size
        n_minus_2=$((n - 2))
        if [ $((n_minus_2 % grid_size)) -eq 0 ]; then
            echo "Running with $processes processes, grid size $n..."
            mpirun -np $processes ./task4_dirichlet $n 0.0 > /dev/null 2>&1
        fi
    done
done

echo ""
echo "========================================="
echo "Experiments completed!"
echo "========================================="
echo "Results saved in task*.txt files"
echo "Run 'python3 plot_results.py' to generate plots"


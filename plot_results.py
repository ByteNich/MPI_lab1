#!/usr/bin/env python3
"""
Script for plotting results from MPI experiments
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_task1_results():
    """Plot results for Task 1: Monte Carlo π calculation"""
    if not os.path.exists('task1_results.txt'):
        print("task1_results.txt not found")
        return
    
    data = np.loadtxt('task1_results.txt')
    if len(data) == 0:
        print("No data in task1_results.txt")
        return
    
    # Group by number of processes
    processes = np.unique(data[:, 0])
    points = np.unique(data[:, 1])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Execution time vs number of processes
    ax = axes[0]
    for p in points:
        mask = data[:, 1] == p
        if np.any(mask):
            proc_data = data[mask]
            proc_data = proc_data[proc_data[:, 0].argsort()]
            ax.plot(proc_data[:, 0], proc_data[:, 2], marker='o', label=f'{int(p)} points')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Task 1: Execution Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Speedup
    ax = axes[1]
    for p in points:
        mask = data[:, 1] == p
        if np.any(mask):
            proc_data = data[mask]
            proc_data = proc_data[proc_data[:, 0].argsort()]
            times = proc_data[:, 2]
            if len(times) > 0:
                speedup = times[0] / times
                ax.plot(proc_data[:, 0], speedup, marker='o', label=f'{int(p)} points')
                ax.plot(proc_data[:, 0], proc_data[:, 0], '--', alpha=0.5, label='Linear speedup' if p == points[0] else '')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Speedup')
    ax.set_title('Task 1: Speedup')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Efficiency
    ax = axes[2]
    for p in points:
        mask = data[:, 1] == p
        if np.any(mask):
            proc_data = data[mask]
            proc_data = proc_data[proc_data[:, 0].argsort()]
            times = proc_data[:, 2]
            if len(times) > 0:
                speedup = times[0] / times
                efficiency = speedup / proc_data[:, 0]
                ax.plot(proc_data[:, 0], efficiency, marker='o', label=f'{int(p)} points')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Efficiency')
    ax.set_title('Task 1: Efficiency')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('task1_plots.png', dpi=300)
    print("Saved task1_plots.png")

def plot_task2_results():
    """Plot results for Task 2: Matrix-vector multiplication"""
    if not os.path.exists('task2_results.txt'):
        print("task2_results.txt not found")
        return
    
    data = np.loadtxt('task2_results.txt', dtype=str)
    if len(data) == 0:
        print("No data in task2_results.txt")
        return
    
    # Convert to numeric
    methods = data[:, 0]
    processes = data[:, 1].astype(int)
    sizes = data[:, 2].astype(int)
    times = data[:, 3].astype(float)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for method_idx, method in enumerate(['r', 'c', 'b']):
        method_mask = methods == method
        method_name = {'r': 'Row', 'c': 'Column', 'b': 'Block'}[method]
        
        method_processes = processes[method_mask]
        method_sizes = sizes[method_mask]
        method_times = times[method_mask]
        
        unique_processes = np.unique(method_processes)
        unique_sizes = np.unique(method_sizes)
        
        # Execution time
        ax = axes[0, method_idx]
        for proc in unique_processes:
            mask = (method_processes == proc) & method_mask
            if np.any(mask):
                proc_sizes = method_sizes[mask]
                proc_times = method_times[mask]
                sorted_indices = proc_sizes.argsort()
                ax.plot(proc_sizes[sorted_indices], proc_times[sorted_indices], 
                       marker='o', label=f'{proc} processes')
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'{method_name} Partitioning: Execution Time')
        ax.legend()
        ax.grid(True)
        
        # Speedup
        ax = axes[1, method_idx]
        for size in unique_sizes:
            size_mask = (method_sizes == size) & method_mask
            if np.any(size_mask):
                size_proc = method_processes[size_mask]
                size_times = method_times[size_mask]
                sorted_indices = size_proc.argsort()
                sorted_proc = size_proc[sorted_indices]
                sorted_times = size_times[sorted_indices]
                if len(sorted_times) > 0:
                    speedup = sorted_times[0] / sorted_times
                    ax.plot(sorted_proc, speedup, marker='o', label=f'Size {size}')
                    if size == unique_sizes[0]:
                        ax.plot(sorted_proc, sorted_proc, '--', alpha=0.5, label='Linear')
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Speedup')
        ax.set_title(f'{method_name} Partitioning: Speedup')
        ax.legend()
        ax.grid(True)
        
        # Efficiency
        ax = axes[2, method_idx]
        for size in unique_sizes:
            size_mask = (method_sizes == size) & method_mask
            if np.any(size_mask):
                size_proc = method_processes[size_mask]
                size_times = method_times[size_mask]
                sorted_indices = size_proc.argsort()
                sorted_proc = size_proc[sorted_indices]
                sorted_times = size_times[sorted_indices]
                if len(sorted_times) > 0:
                    speedup = sorted_times[0] / sorted_times
                    efficiency = speedup / sorted_proc
                    ax.plot(sorted_proc, efficiency, marker='o', label=f'Size {size}')
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Efficiency')
        ax.set_title(f'{method_name} Partitioning: Efficiency')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('task2_plots.png', dpi=300)
    print("Saved task2_plots.png")

def plot_task3_results():
    """Plot results for Task 3: Cannon's algorithm"""
    if not os.path.exists('task3_results.txt'):
        print("task3_results.txt not found")
        return
    
    data = np.loadtxt('task3_results.txt')
    if len(data) == 0:
        print("No data in task3_results.txt")
        return
    
    processes = np.unique(data[:, 0])
    sizes = np.unique(data[:, 1])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Execution time
    ax = axes[0]
    for proc in processes:
        mask = data[:, 0] == proc
        if np.any(mask):
            proc_data = data[mask]
            proc_data = proc_data[proc_data[:, 1].argsort()]
            ax.plot(proc_data[:, 1], proc_data[:, 2], marker='o', label=f'{int(proc)} processes')
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Task 3: Execution Time')
    ax.legend()
    ax.grid(True)
    
    # Speedup
    ax = axes[1]
    for size in sizes:
        mask = data[:, 1] == size
        if np.any(mask):
            size_data = data[mask]
            size_data = size_data[size_data[:, 0].argsort()]
            times = size_data[:, 2]
            if len(times) > 0:
                speedup = times[0] / times
                ax.plot(size_data[:, 0], speedup, marker='o', label=f'Size {int(size)}')
                if size == sizes[0]:
                    ax.plot(size_data[:, 0], size_data[:, 0], '--', alpha=0.5, label='Linear')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Speedup')
    ax.set_title('Task 3: Speedup')
    ax.legend()
    ax.grid(True)
    
    # Efficiency
    ax = axes[2]
    for size in sizes:
        mask = data[:, 1] == size
        if np.any(mask):
            size_data = data[mask]
            size_data = size_data[size_data[:, 0].argsort()]
            times = size_data[:, 2]
            if len(times) > 0:
                speedup = times[0] / times
                efficiency = speedup / size_data[:, 0]
                ax.plot(size_data[:, 0], efficiency, marker='o', label=f'Size {int(size)}')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Efficiency')
    ax.set_title('Task 3: Efficiency')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('task3_plots.png', dpi=300)
    print("Saved task3_plots.png")

def plot_task4_results():
    """Plot results for Task 4: Dirichlet problem"""
    if not os.path.exists('task4_results.txt'):
        print("task4_results.txt not found")
        return
    
    data = np.loadtxt('task4_results.txt')
    if len(data) == 0:
        print("No data in task4_results.txt")
        return
    
    processes = np.unique(data[:, 0])
    sizes = np.unique(data[:, 1])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Execution time
    ax = axes[0]
    for proc in processes:
        mask = data[:, 0] == proc
        if np.any(mask):
            proc_data = data[mask]
            proc_data = proc_data[proc_data[:, 1].argsort()]
            ax.plot(proc_data[:, 1], proc_data[:, 2], marker='o', label=f'{int(proc)} processes')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Task 4: Execution Time')
    ax.legend()
    ax.grid(True)
    
    # Speedup
    ax = axes[1]
    for size in sizes:
        mask = data[:, 1] == size
        if np.any(mask):
            size_data = data[mask]
            size_data = size_data[size_data[:, 0].argsort()]
            times = size_data[:, 2]
            if len(times) > 0:
                speedup = times[0] / times
                ax.plot(size_data[:, 0], speedup, marker='o', label=f'Size {int(size)}')
                if size == sizes[0]:
                    ax.plot(size_data[:, 0], size_data[:, 0], '--', alpha=0.5, label='Linear')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Speedup')
    ax.set_title('Task 4: Speedup')
    ax.legend()
    ax.grid(True)
    
    # Efficiency
    ax = axes[2]
    for size in sizes:
        mask = data[:, 1] == size
        if np.any(mask):
            size_data = data[mask]
            size_data = size_data[size_data[:, 0].argsort()]
            times = size_data[:, 2]
            if len(times) > 0:
                speedup = times[0] / times
                efficiency = speedup / size_data[:, 0]
                ax.plot(size_data[:, 0], efficiency, marker='o', label=f'Size {int(size)}')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Efficiency')
    ax.set_title('Task 4: Efficiency')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('task4_plots.png', dpi=300)
    print("Saved task4_plots.png")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        task = sys.argv[1]
        if task == '1':
            plot_task1_results()
        elif task == '2':
            plot_task2_results()
        elif task == '3':
            plot_task3_results()
        elif task == '4':
            plot_task4_results()
        else:
            print("Usage: python3 plot_results.py [1|2|3|4]")
    else:
        plot_task1_results()
        plot_task2_results()
        plot_task3_results()
        plot_task4_results()


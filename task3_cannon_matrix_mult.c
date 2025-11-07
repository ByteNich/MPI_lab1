#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

// Cannon's algorithm for matrix multiplication
void cannon_matrix_mult(int *A, int *B, int *C, int n, int grid_size, 
                        int rank, int size) {
    int block_size = n / grid_size;
    int row = rank / grid_size;
    int col = rank % grid_size;
    
    // Allocate local blocks
    int *local_A = (int*)malloc(block_size * block_size * sizeof(int));
    int *local_B = (int*)malloc(block_size * block_size * sizeof(int));
    int *local_C = (int*)calloc(block_size * block_size, sizeof(int));
    
    // Initialize local blocks from global matrices
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            int global_i = row * block_size + i;
            int global_j = col * block_size + j;
            local_A[i * block_size + j] = A[global_i * n + global_j];
            local_B[i * block_size + j] = B[global_i * n + global_j];
        }
    }
    
    // Create row and column communicators
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
    
    int row_rank, row_size;
    int col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);
    
    // Calculate neighbor ranks in row and column communicators
    int left_rank = (row_rank - 1 + row_size) % row_size;
    int right_rank = (row_rank + 1) % row_size;
    int up_rank = (col_rank - 1 + col_size) % col_size;
    int down_rank = (col_rank + 1) % col_size;
    
    // Initial alignment: shift A left by 'row' positions, shift B up by 'col' positions
    for (int shift = 0; shift < row; shift++) {
        MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_INT,
                            left_rank, 0, right_rank, 0, row_comm, MPI_STATUS_IGNORE);
    }
    
    // Shift B up by col positions
    for (int shift = 0; shift < col; shift++) {
        MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_INT,
                            up_rank, 0, down_rank, 0, col_comm, MPI_STATUS_IGNORE);
    }
    
    // Main computation loop
    for (int step = 0; step < grid_size; step++) {
        // Multiply local blocks
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    local_C[i * block_size + j] += 
                        local_A[i * block_size + k] * local_B[k * block_size + j];
                }
            }
        }
        
        // Shift A left by 1
        if (step < grid_size - 1) {
            MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_INT,
                                left_rank, 0, right_rank, 0, row_comm, MPI_STATUS_IGNORE);
        }
        
        // Shift B up by 1
        if (step < grid_size - 1) {
            MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_INT,
                                up_rank, 0, down_rank, 0, col_comm, MPI_STATUS_IGNORE);
        }
    }
    
    // Gather results to rank 0
    // Each process computes its block, we need to gather all blocks to rank 0
    if (rank == 0) {
        // Root process: copy its own block first
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                int global_i = row * block_size + i;
                int global_j = col * block_size + j;
                C[global_i * n + global_j] = local_C[i * block_size + j];
            }
        }
        
        // Receive blocks from other processes
        for (int src_rank = 1; src_rank < size; src_rank++) {
            int src_row = src_rank / grid_size;
            int src_col = src_rank % grid_size;
            
            MPI_Recv(local_C, block_size * block_size, MPI_INT, src_rank, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Place received block in correct position
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    int global_i = src_row * block_size + i;
                    int global_j = src_col * block_size + j;
                    C[global_i * n + global_j] = local_C[i * block_size + j];
                }
            }
        }
    } else {
        // Other processes: send their blocks to rank 0
        MPI_Send(local_C, block_size * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
    free(local_A);
    free(local_B);
    free(local_C);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n;
    int *A = NULL;
    int *B = NULL;
    int *C = NULL;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    n = atoi(argv[1]);
    int grid_size = (int)sqrt(size);
    
    if (grid_size * grid_size != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of processes must be a perfect square\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (n % grid_size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Matrix size must be divisible by grid size\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Allocate and initialize matrices on root
    if (rank == 0) {
        A = (int*)malloc(n * n * sizeof(int));
        B = (int*)malloc(n * n * sizeof(int));
        C = (int*)malloc(n * n * sizeof(int));
        
        srand(time(NULL));
        for (int i = 0; i < n * n; i++) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
        }
    } else {
        A = (int*)malloc(n * n * sizeof(int));
        B = (int*)malloc(n * n * sizeof(int));
        C = (int*)malloc(n * n * sizeof(int));
    }
    
    // Broadcast matrices to all processes
    MPI_Bcast(A, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    cannon_matrix_mult(A, B, C, n, grid_size, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        double execution_time = end_time - start_time;
        printf("========================================\n");
        printf("Cannon's Matrix Multiplication\n");
        printf("========================================\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Processes: %d (grid: %d x %d)\n", size, grid_size, grid_size);
        printf("Execution time: %.6f seconds\n", execution_time);
        printf("========================================\n");
        
        // Write results to file
        FILE *fp = fopen("task3_results.txt", "a");
        if (fp != NULL) {
            fprintf(fp, "%d %d %.6f\n", size, n, execution_time);
            fclose(fp);
        }
    }
    
    free(A);
    free(B);
    free(C);
    
    MPI_Finalize();
    return 0;
}


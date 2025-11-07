#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

// Row partitioning: each process gets a subset of rows
void matrix_vector_mult_row_partition(int *matrix, int *vector, int *result, 
                                       int local_rows, int n) {
    for (int i = 0; i < local_rows; i++) {
        result[i] = 0;
        for (int j = 0; j < n; j++) {
            result[i] += matrix[i * n + j] * vector[j];
        }
    }
}

// Column partitioning: each process gets a subset of columns
void matrix_vector_mult_col_partition(int *matrix, int *vector, int *result,
                                       int local_cols, int n, int rank, int size) {
    // Each process computes partial results for its columns
    int *partial_result = (int*)calloc(n, sizeof(int));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < local_cols; j++) {
            int global_col = rank * local_cols + j;
            partial_result[i] += matrix[i * n + global_col] * vector[global_col];
        }
    }
    
    // Reduce all partial results
    MPI_Allreduce(partial_result, result, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    free(partial_result);
}

// Block partitioning: 2D decomposition
void matrix_vector_mult_block_partition(int *matrix, int *vector, int *result,
                                         int block_rows, int block_cols, int n,
                                         int rank, int size) {
    int grid_size = (int)sqrt(size);
    int row = rank / grid_size;
    int col = rank % grid_size;
    
    int *local_matrix = (int*)malloc(block_rows * block_cols * sizeof(int));
    int *local_vector = (int*)malloc(block_cols * sizeof(int));
    int *local_result = (int*)calloc(block_rows, sizeof(int));
    
    // Extract local matrix block
    for (int i = 0; i < block_rows; i++) {
        for (int j = 0; j < block_cols; j++) {
            int global_i = row * block_rows + i;
            int global_j = col * block_cols + j;
            local_matrix[i * block_cols + j] = matrix[global_i * n + global_j];
        }
    }
    
    // Extract local vector segment
    for (int j = 0; j < block_cols; j++) {
        int global_j = col * block_cols + j;
        local_vector[j] = vector[global_j];
    }
    
    // Local computation: multiply block with vector segment
    for (int i = 0; i < block_rows; i++) {
        for (int j = 0; j < block_cols; j++) {
            local_result[i] += local_matrix[i * block_cols + j] * local_vector[j];
        }
    }
    
    // Reduce along columns (processes in same column contribute to same rows)
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
    
    int *row_results = (int*)malloc(block_rows * sizeof(int));
    MPI_Reduce(local_result, row_results, block_rows, MPI_INT, MPI_SUM, 0, col_comm);
    
    // Gather results from column leaders (processes with col == 0) to root
    if (col == 0) {
        // Column leaders gather results along their row
        if (rank == 0) {
            // Root process collects from all column leaders
            int *temp_result = (int*)malloc(block_rows * sizeof(int));
            for (int r = 0; r < grid_size; r++) {
                int src_rank = r * grid_size;  // Column leaders are at rank r*grid_size
                if (src_rank == rank) {
                    // Copy own data
                    for (int i = 0; i < block_rows; i++) {
                        result[r * block_rows + i] = row_results[i];
                    }
                } else {
                    // Receive from other column leaders
                    MPI_Recv(temp_result, block_rows, MPI_INT, src_rank, 0, 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int i = 0; i < block_rows; i++) {
                        result[r * block_rows + i] = temp_result[i];
                    }
                }
            }
            free(temp_result);
        } else {
            // Other column leaders send to root
            MPI_Send(row_results, block_rows, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Comm_free(&col_comm);
    free(local_matrix);
    free(local_vector);
    free(local_result);
    free(row_results);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n;
    int *matrix = NULL;
    int *vector = NULL;
    int *result = NULL;
    double start_time, end_time;
    char method;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <matrix_size> <method>\n", argv[0]);
            fprintf(stderr, "Methods: r (row), c (column), b (block)\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    n = atoi(argv[1]);
    method = argv[2][0];
    
    // Allocate and initialize matrix and vector on root
    if (rank == 0) {
        matrix = (int*)malloc(n * n * sizeof(int));
        vector = (int*)malloc(n * sizeof(int));
        result = (int*)malloc(n * sizeof(int));
        
        srand(time(NULL));
        for (int i = 0; i < n * n; i++) {
            matrix[i] = rand() % 100;
        }
        for (int i = 0; i < n; i++) {
            vector[i] = rand() % 100;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    if (method == 'r') {
        // Row partitioning
        int local_rows = n / size;
        int *local_matrix = (int*)malloc(local_rows * n * sizeof(int));
        int *local_result = (int*)malloc(local_rows * sizeof(int));
        
        // Scatter matrix rows
        MPI_Scatter(matrix, local_rows * n, MPI_INT, local_matrix, 
                   local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Broadcast vector to all processes
        if (rank == 0) {
            MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            vector = (int*)malloc(n * sizeof(int));
            MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Compute local matrix-vector product
        matrix_vector_mult_row_partition(local_matrix, vector, local_result, local_rows, n);
        
        // Gather results
        MPI_Gather(local_result, local_rows, MPI_INT, result, local_rows, 
                   MPI_INT, 0, MPI_COMM_WORLD);
        
        free(local_matrix);
        free(local_result);
        if (rank != 0) free(vector);
        
    } else if (method == 'c') {
        // Column partitioning
        int local_cols = n / size;
        
        if (rank == 0) {
            MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            vector = (int*)malloc(n * sizeof(int));
            MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);
            matrix = (int*)malloc(n * n * sizeof(int));
        }
        
        MPI_Bcast(matrix, n * n, MPI_INT, 0, MPI_COMM_WORLD);
        result = (int*)malloc(n * sizeof(int));
        
        matrix_vector_mult_col_partition(matrix, vector, result, local_cols, n, rank, size);
        
        if (rank != 0) {
            free(matrix);
            free(vector);
        }
        
    } else if (method == 'b') {
        // Block partitioning (assumes size is a perfect square)
        int grid_size = (int)sqrt(size);
        if (grid_size * grid_size != size) {
            if (rank == 0) {
                fprintf(stderr, "Error: Number of processes must be a perfect square for block partitioning\n");
            }
            MPI_Finalize();
            return 1;
        }
        
        int block_rows = n / grid_size;
        int block_cols = n / grid_size;
        
        if (rank == 0) {
            result = (int*)malloc(n * sizeof(int));
        } else {
            matrix = (int*)malloc(n * n * sizeof(int));
            vector = (int*)malloc(n * sizeof(int));
            result = (int*)malloc(n * sizeof(int));
        }
        
        MPI_Bcast(matrix, n * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);
        
        matrix_vector_mult_block_partition(matrix, vector, result, block_rows, 
                                          block_cols, n, rank, size);
        
        if (rank != 0) {
            free(matrix);
            free(vector);
        }
    } else {
        if (rank == 0) {
            fprintf(stderr, "Unknown method: %c\n", method);
        }
        MPI_Finalize();
        return 1;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        double execution_time = end_time - start_time;
        printf("========================================\n");
        printf("Matrix-Vector Multiplication\n");
        printf("========================================\n");
        printf("Method: %s\n", method == 'r' ? "Row" : (method == 'c' ? "Column" : "Block"));
        printf("Matrix size: %d x %d\n", n, n);
        printf("Processes: %d\n", size);
        printf("Execution time: %.6f seconds\n", execution_time);
        printf("========================================\n");
        
        // Write results to file
        FILE *fp = fopen("task2_results.txt", "a");
        if (fp != NULL) {
            fprintf(fp, "%c %d %d %.6f\n", method, size, n, execution_time);
            fclose(fp);
        }
    }
    
    if (rank == 0) {
        free(matrix);
        free(vector);
        free(result);
    }
    
    MPI_Finalize();
    return 0;
}


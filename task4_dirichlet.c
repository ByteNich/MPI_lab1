#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_ITERATIONS 10000
#define TOLERANCE 1e-6

// Compute function f(x, y) - source term
double f(double x, double y) {
    return 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

// Gauss-Seidel iteration with wavefront computation
void gauss_seidel_wavefront(double **u, double **u_new, double **f_grid,
                            int local_rows, int local_cols, int n,
                            double h, double h2, int rank, int size) {
    int grid_size = (int)sqrt(size);
    int row = rank / grid_size;
    int col = rank % grid_size;
    
    // Wavefront computation: process diagonals
    for (int diag = 0; diag < local_rows + local_cols - 1; diag++) {
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                if (i + j == diag) {
                    int global_i = row * local_rows + i + 1;
                    int global_j = col * local_cols + j + 1;
                    
                    if (global_i > 0 && global_i < n - 1 && 
                        global_j > 0 && global_j < n - 1) {
                        u_new[i][j] = 0.25 * (
                            u[i + 1][j] + u_new[i - 1][j] +
                            u[i][j + 1] + u_new[i][j - 1] -
                            h2 * f_grid[i][j]
                        );
                    }
                }
            }
        }
        
        // Synchronize with neighbors after processing each diagonal
        // Send boundary values to neighbors
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Exchange boundary values with neighbors
        if (row > 0) {
            // Send top row to top neighbor, receive from top
            MPI_Sendrecv(u_new[0], local_cols, MPI_DOUBLE, rank - grid_size, 0,
                        u[0], local_cols, MPI_DOUBLE, rank - grid_size, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (row < grid_size - 1) {
            // Send bottom row to bottom neighbor, receive from bottom
            MPI_Sendrecv(u_new[local_rows - 1], local_cols, MPI_DOUBLE, rank + grid_size, 0,
                        u[local_rows - 1], local_cols, MPI_DOUBLE, rank + grid_size, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (col > 0) {
            // Send left column to left neighbor, receive from left
            double *send_left = (double*)malloc(local_rows * sizeof(double));
            double *recv_left = (double*)malloc(local_rows * sizeof(double));
            for (int k = 0; k < local_rows; k++) {
                send_left[k] = u_new[k][0];
            }
            MPI_Sendrecv(send_left, local_rows, MPI_DOUBLE, rank - 1, 0,
                        recv_left, local_rows, MPI_DOUBLE, rank - 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int k = 0; k < local_rows; k++) {
                u[k][0] = recv_left[k];
            }
            free(send_left);
            free(recv_left);
        }
        if (col < grid_size - 1) {
            // Send right column to right neighbor, receive from right
            double *send_right = (double*)malloc(local_rows * sizeof(double));
            double *recv_right = (double*)malloc(local_rows * sizeof(double));
            for (int k = 0; k < local_rows; k++) {
                send_right[k] = u_new[k][local_cols - 1];
            }
            MPI_Sendrecv(send_right, local_rows, MPI_DOUBLE, rank + 1, 0,
                        recv_right, local_rows, MPI_DOUBLE, rank + 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int k = 0; k < local_rows; k++) {
                u[k][local_cols - 1] = recv_right[k];
            }
            free(send_right);
            free(recv_right);
        }
        
        // Copy u_new to u for next iteration
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                u[i][j] = u_new[i][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n;
    double c;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <grid_size> <boundary_value>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    n = atoi(argv[1]);
    c = atof(argv[2]);
    
    int grid_size = (int)sqrt(size);
    if (grid_size * grid_size != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of processes must be a perfect square\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if ((n - 2) % grid_size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: (grid_size - 2) must be divisible by process grid size\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    int row = rank / grid_size;
    int col = rank % grid_size;
    int local_rows = (n - 2) / grid_size;
    int local_cols = (n - 2) / grid_size;
    
    double h = 1.0 / (n - 1);
    double h2 = h * h;
    
    // Allocate local grids (with ghost cells)
    double **u = (double**)malloc((local_rows + 2) * sizeof(double*));
    double **u_new = (double**)malloc((local_rows + 2) * sizeof(double*));
    double **f_grid = (double**)malloc(local_rows * sizeof(double*));
    
    for (int i = 0; i < local_rows + 2; i++) {
        u[i] = (double*)malloc((local_cols + 2) * sizeof(double));
        u_new[i] = (double*)malloc((local_cols + 2) * sizeof(double));
    }
    for (int i = 0; i < local_rows; i++) {
        f_grid[i] = (double*)malloc(local_cols * sizeof(double));
    }
    
    // Initialize interior to zero
    for (int i = 0; i < local_rows + 2; i++) {
        for (int j = 0; j < local_cols + 2; j++) {
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
        }
    }
    
    // Set boundary conditions
    // Top boundary (global_i == 0)
    if (row == 0) {
        for (int j = 0; j < local_cols + 2; j++) {
            u[0][j] = c;
            u_new[0][j] = c;
        }
    }
    
    // Bottom boundary (global_i == n-1)
    if (row == grid_size - 1) {
        for (int j = 0; j < local_cols + 2; j++) {
            u[local_rows + 1][j] = c;
            u_new[local_rows + 1][j] = c;
        }
    }
    
    // Left boundary (global_j == 0)
    if (col == 0) {
        for (int i = 0; i < local_rows + 2; i++) {
            u[i][0] = c;
            u_new[i][0] = c;
        }
    }
    
    // Right boundary (global_j == n-1)
    if (col == grid_size - 1) {
        for (int i = 0; i < local_rows + 2; i++) {
            u[i][local_cols + 1] = c;
            u_new[i][local_cols + 1] = c;
        }
    }
    
    // Initialize source term for interior points
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < local_cols; j++) {
            // Map to global indices (add 1 because local grid has ghost cells)
            int global_i = row * local_rows + i + 1;
            int global_j = col * local_cols + j + 1;
            double x = global_i * h;
            double y = global_j * h;
            f_grid[i][j] = f(x, y);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Gauss-Seidel iteration
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        double max_diff = 0.0;
        double local_max_diff = 0.0;
        
        // Update interior points (local indices 1..local_rows, 1..local_cols)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                // These are interior points, update using Gauss-Seidel
                double old_val = u_new[i][j];
                u_new[i][j] = 0.25 * (
                    u[i + 1][j] + u_new[i - 1][j] +
                    u[i][j + 1] + u_new[i][j - 1] -
                    h2 * f_grid[i - 1][j - 1]
                );
                
                double diff = fabs(u_new[i][j] - old_val);
                if (diff > local_max_diff) {
                    local_max_diff = diff;
                }
            }
        }
        
        // Exchange boundary values
        if (row > 0) {
            MPI_Sendrecv(u_new[1], local_cols, MPI_DOUBLE, rank - grid_size, 0,
                        u[0], local_cols, MPI_DOUBLE, rank - grid_size, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (row < grid_size - 1) {
            MPI_Sendrecv(u_new[local_rows], local_cols, MPI_DOUBLE, rank + grid_size, 0,
                        u[local_rows + 1], local_cols, MPI_DOUBLE, rank + grid_size, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (col > 0) {
            double *send_left = (double*)malloc(local_rows * sizeof(double));
            double *recv_left = (double*)malloc(local_rows * sizeof(double));
            for (int k = 0; k < local_rows; k++) {
                send_left[k] = u_new[k + 1][1];
            }
            MPI_Sendrecv(send_left, local_rows, MPI_DOUBLE, rank - 1, 0,
                        recv_left, local_rows, MPI_DOUBLE, rank - 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int k = 0; k < local_rows; k++) {
                u[k + 1][0] = recv_left[k];
            }
            free(send_left);
            free(recv_left);
        }
        if (col < grid_size - 1) {
            double *send_right = (double*)malloc(local_rows * sizeof(double));
            double *recv_right = (double*)malloc(local_rows * sizeof(double));
            for (int k = 0; k < local_rows; k++) {
                send_right[k] = u_new[k + 1][local_cols];
            }
            MPI_Sendrecv(send_right, local_rows, MPI_DOUBLE, rank + 1, 0,
                        recv_right, local_rows, MPI_DOUBLE, rank + 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int k = 0; k < local_rows; k++) {
                u[k + 1][local_cols + 1] = recv_right[k];
            }
            free(send_right);
            free(recv_right);
        }
        
        // Update u from u_new
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                u[i][j] = u_new[i][j];
            }
        }
        
        // Check convergence
        MPI_Allreduce(&local_max_diff, &max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        if (max_diff < TOLERANCE) {
            if (rank == 0) {
                printf("Converged after %d iterations\n", iter + 1);
            }
            break;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        double execution_time = end_time - start_time;
        printf("========================================\n");
        printf("Dirichlet Problem - Gauss-Seidel\n");
        printf("========================================\n");
        printf("Grid size: %d x %d\n", n, n);
        printf("Processes: %d (grid: %d x %d)\n", size, grid_size, grid_size);
        printf("Boundary value: %.2f\n", c);
        printf("Execution time: %.6f seconds\n", execution_time);
        printf("========================================\n");
        
        // Write results to file
        FILE *fp = fopen("task4_results.txt", "a");
        if (fp != NULL) {
            fprintf(fp, "%d %d %.6f\n", size, n, execution_time);
            fclose(fp);
        }
    }
    
    // Free memory
    for (int i = 0; i < local_rows + 2; i++) {
        free(u[i]);
        free(u_new[i]);
    }
    for (int i = 0; i < local_rows; i++) {
        free(f_grid[i]);
    }
    free(u);
    free(u_new);
    free(f_grid);
    
    MPI_Finalize();
    return 0;
}


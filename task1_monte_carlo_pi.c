#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SQUARE_SIDE 2.0

// Generate random double in range [min, max]
double random_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

int main(int argc, char *argv[]) {
    int rank, size;
    long long int total_points;
    long long int points_per_process;
    long long int local_hits = 0;
    long long int total_hits = 0;
    double start_time, end_time;
    double x, y, distance;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <total_points>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    total_points = atoll(argv[1]);
    points_per_process = total_points / size;
    
    // Each process uses different seed
    srand(time(NULL) + rank);
    
    start_time = MPI_Wtime();
    
    // Each process generates its portion of random points
    for (long long int i = 0; i < points_per_process; i++) {
        // Generate point in square [-1, 1] x [-1, 1]
        x = random_double(-1.0, 1.0);
        y = random_double(-1.0, 1.0);
        
        // Check if point is inside unit circle
        distance = sqrt(x * x + y * y);
        if (distance <= 1.0) {
            local_hits++;
        }
    }
    
    // Gather all hits from all processes
    MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        double pi_estimate = 4.0 * (double)total_hits / (double)total_points;
        double execution_time = end_time - start_time;
        
        printf("========================================\n");
        printf("Monte Carlo π Calculation\n");
        printf("========================================\n");
        printf("Total points: %lld\n", total_points);
        printf("Processes: %d\n", size);
        printf("Points per process: %lld\n", points_per_process);
        printf("Total hits: %lld\n", total_hits);
        printf("π estimate: %.10f\n", pi_estimate);
        printf("π actual:   %.10f\n", M_PI);
        printf("Error:      %.10f\n", fabs(pi_estimate - M_PI));
        printf("Execution time: %.6f seconds\n", execution_time);
        printf("========================================\n");
        
        // Write results to file for plotting
        FILE *fp = fopen("task1_results.txt", "a");
        if (fp != NULL) {
            fprintf(fp, "%d %lld %.6f\n", size, total_points, execution_time);
            fclose(fp);
        }
    }
    
    MPI_Finalize();
    return 0;
}


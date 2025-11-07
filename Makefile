CC = mpicc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

TARGETS = task1_monte_carlo_pi task2_matrix_vector task3_cannon_matrix_mult task4_dirichlet

all: $(TARGETS)

task1_monte_carlo_pi: task1_monte_carlo_pi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

task2_matrix_vector: task2_matrix_vector.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

task3_cannon_matrix_mult: task3_cannon_matrix_mult.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

task4_dirichlet: task4_dirichlet.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o *.txt

.PHONY: all clean


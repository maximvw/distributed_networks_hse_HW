CC=mpicc
CFLAGS=-O2 -std=c11
TARGETS=task1_pi task2_matvec_rows task2_matvec_cols task2_matvec_block task3_cannon task4_dirichlet

all: $(TARGETS)

task1_pi: tasks/task1_pi.c
	$(CC) $(CFLAGS) -o o_files/$@ $<

task2_matvec_rows: tasks/task2_matvec_rows.c
	$(CC) $(CFLAGS) -o o_files/$@ $<

task2_matvec_cols: tasks/task2_matvec_cols.c
	$(CC) $(CFLAGS) -o o_files/$@ $<

task2_matvec_block: tasks/task2_matvec_block.c
	$(CC) $(CFLAGS) -o o_files/$@ $<

task3_cannon: tasks/task3_cannon.c
	$(CC) $(CFLAGS) -o o_files/$@ $<

task4_dirichlet: tasks/task4_dirichlet.c
	$(CC) $(CFLAGS) -o o_files/$@ $< -lm

clean:
	rm -f o_files/* results/*.csv

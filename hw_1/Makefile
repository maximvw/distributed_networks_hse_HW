CC=mpicc
CFLAGS=-O2 -std=c11
LIBS=-lm
TARGETS=task1_pi task2_matvec_rows task2_matvec_cols task2_matvec_block task3_cannon task4_dirichlet

O_DIR=o_files
RESULTS_DIR=results

all: $(O_DIR) $(RESULTS_DIR) $(TARGETS)

$(O_DIR):
	mkdir -p $(O_DIR)

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

task1_pi: tasks/task1_pi.c
	$(CC) $(CFLAGS) -o $(O_DIR)/$@ $< $(LIBS)

task2_matvec_rows: tasks/task2_matvec_rows.c
	$(CC) $(CFLAGS) -o $(O_DIR)/$@ $< $(LIBS)

task2_matvec_cols: tasks/task2_matvec_cols.c
	$(CC) $(CFLAGS) -o $(O_DIR)/$@ $< $(LIBS)

task2_matvec_block: tasks/task2_matvec_block.c
	$(CC) $(CFLAGS) -o $(O_DIR)/$@ $< $(LIBS)

task3_cannon: tasks/task3_cannon.c
	$(CC) $(CFLAGS) -o $(O_DIR)/$@ $< $(LIBS)

task4_dirichlet: tasks/task4_dirichlet.c
	$(CC) $(CFLAGS) -o $(O_DIR)/$@ $< $(LIBS)

clean:
	rm -f $(O_DIR)/* $(RESULTS_DIR)/*.csv

.PHONY: all clean
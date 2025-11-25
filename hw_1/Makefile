CC       = mpicc
CFLAGS   = -O2 -std=c11 -Wall
LIBS     = -lm

TASK_DIR   = tasks
TEST_DIR   = tests
O_DIR      = o_files
RESULTS_DIR= results


TASKS = task1_pi task2_matvec_rows task2_matvec_cols \
        task2_matvec_block task3_cannon task4_dirichlet

all: $(O_DIR) $(RESULTS_DIR) $(addprefix $(O_DIR)/, $(TASKS))

$(O_DIR):
	mkdir -p $(O_DIR)

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

$(O_DIR)/%: $(TASK_DIR)/%.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf $(O_DIR) $(RESULTS_DIR) 
	@echo "Clean done."

.PHONY: all clean tests test1 test2r test2c test2b test3 test4

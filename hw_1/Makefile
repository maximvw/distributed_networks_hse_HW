CC       = mpicc
CFLAGS   = -O2 -std=c11 -Wall
LIBS     = -lm

TASK_DIR   = tasks
TEST_DIR   = tests
O_DIR      = o_files
RESULTS_DIR= results
TEST_OUT   = build_tests

TASKS = task1_pi task2_matvec_rows task2_matvec_cols \
        task2_matvec_block task3_cannon task4_dirichlet

TESTS = test_task1_pi test_task2_matvec_rows test_task2_matvec_cols \
        test_task2_matvec_block test_task3_cannon test_task4_dirichlet

all: $(O_DIR) $(RESULTS_DIR) $(addprefix $(O_DIR)/, $(TASKS))

$(O_DIR):
	mkdir -p $(O_DIR)

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

$(O_DIR)/%: $(TASK_DIR)/%.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

tests: $(TEST_OUT) $(addprefix $(TEST_OUT)/, $(TESTS))
	@echo "--------------------------------------"
	@echo "Running all tests"
	@echo "--------------------------------------"
	@for t in $(TESTS); do \
		echo "→ Running $$t"; \
		mpirun --oversubscribe -np 4 $(TEST_OUT)/$$t 8 200 || exit 1; \
	done
	@echo "All tests completed successfully."

$(TEST_OUT):
	mkdir -p $(TEST_OUT)

$(TEST_OUT)/%: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test1:
	$(CC) $(CFLAGS) $(TEST_DIR)/test_task1_pi.c -o $(TEST_OUT)/test_task1_pi $(LIBS)
	mpirun --oversubscribe -np 4 $(TEST_OUT)/test_task1_pi 10000000

test2r:
	$(CC) $(CFLAGS) $(TEST_DIR)/test_task2_matvec_rows.c -o $(TEST_OUT)/test_task2_matvec_rows $(LIBS)
	mpirun --oversubscribe -np 4 $(TEST_OUT)/test_task2_matvec_rows 8

test2c:
	$(CC) $(CFLAGS) $(TEST_DIR)/test_task2_matvec_cols.c -o $(TEST_OUT)/test_task2_matvec_cols $(LIBS)
	mpirun --oversubscribe -np 4 $(TEST_OUT)/test_task2_matvec_cols 8

test2b:
	$(CC) $(CFLAGS) $(TEST_DIR)/test_task2_matvec_block.c -o $(TEST_OUT)/test_task2_matvec_block $(LIBS)
	mpirun --oversubscribe -np 6 $(TEST_OUT)/test_task2_matvec_block 12

test3:
	$(CC) $(CFLAGS) $(TEST_DIR)/test_task3_cannon.c -o $(TEST_OUT)/test_task3_cannon $(LIBS)
	mpirun --oversubscribe -np 4 $(TEST_OUT)/test_task3_cannon 4

test4:
	$(CC) $(CFLAGS) $(TEST_DIR)/test_task4_dirichlet.c -o $(TEST_OUT)/test_task4_dirichlet $(LIBS)
	mpirun --oversubscribe -np 6 $(TEST_OUT)/test_task4_dirichlet 100 200

clean:
	rm -rf $(O_DIR) $(RESULTS_DIR) $(TEST_OUT)
	@echo "Clean done."

.PHONY: all clean tests test1 test2r test2c test2b test3 test4

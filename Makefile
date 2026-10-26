CC = gcc
SRCS = opt.c gemm.c kernel.c pack.c report.c test.c test_main.c #$(wildcard *.c)
HDRS = gemm.h ops.h sse.h util.h report.h # Header files
TEST_EXE = run

FLAGS = -std=c11 -march=native -O2 -fopenmp -Wall
LIB = -lm

run:
	$(CC) -o run $(SRCS) $(HDRS) $(FLAGS) $(LIB)

.PHONY: run

clean :
	rm $(TEST_EXE)

# Code formatting (.clang-format)
FORMAT_FILES = $(wildcard *.c *.h reference/*.c)

format:
	clang-format -i $(FORMAT_FILES)

format-check:
	clang-format --dry-run --Werror $(FORMAT_FILES)

.PHONY: format format-check

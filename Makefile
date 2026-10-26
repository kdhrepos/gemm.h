CC = gcc
SRCS = opt.c gemm.c kernel.c pack.c test.c test_main.c #$(wildcard *.c)
HDRS = gemm.h ops.h sse.h util.h # Header files
TEST_EXE = tt

FLAGS = -std=c11 -march=native -O2 -fopenmp -Wall
LIB = -lm

tt:
	$(CC) -o tt $(SRCS) $(HDRS) $(FLAGS) $(LIB)

.PHONY: tt

clean :
	rm $(TEST_EXE)

# Code formatting (.clang-format)
FORMAT_FILES = $(wildcard *.c *.h)

format:
	clang-format -i $(FORMAT_FILES)

format-check:
	clang-format --dry-run --Werror $(FORMAT_FILES)

.PHONY: format format-check

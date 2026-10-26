/**********************************************************************************************
 * File   : report.h
 * Author : kdh
 * Github : https://github.com/kdhrepos/gemm.h
 *
 * Description:
 *      Result of one benchmark case and its printer. This is shared by gemm.h's
 *      tester (run) and reference/openblas_ref, so both programs print exactly
 *      the same text / CSV format and their outputs are easy to compare.
 *
 *      It depends only on <stdio.h> so it can be built without gemm.h's headers.
 *
**********************************************************************************************/

#ifndef REPORT_H
#define REPORT_H

#include <stdio.h>

typedef struct {
  const char* lib;  /* "gemm.h", "openblas" */
  const char* name; /* "sgemm", "dgemm", ... */
  const char* isa;  /* instruction set (gemm.h) or core name (OpenBLAS) */
  int threads;
  int M, K, N;
  int niter, warmup;
  double exec_time[3]; /* sum, worst, best (ms) */
  double gflops[3];    /* sum, best, worst */
  int is_valid;
} TestResult;

/* Print one result as a text block, or as a CSV row (header printed once). */
void print_result(FILE* stream, const TestResult* res, const int csv);

#endif  // REPORT_H

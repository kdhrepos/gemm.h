/**********************************************************************************************
 * File   : openblas_ref.c
 *
 * Description:
 *      OpenBLAS cblas_sgemm reference benchmark for gemm.h.
 *      Measures under the same conditions as gemm.h's tester (run):
 *        - same semantics: C += A * B  (beta = 1, C reset to 0 before every call)
 *        - same FLOP count: 2 * M * N * K
 *        - same timer: CLOCK_MONOTONIC_RAW around the gemm call only
 *        - warm-up calls are excluded from the statistics
 *        - the result is validated against a naive GEMM, like run does
 *
 *      The output comes from ../report.c, which gemm.h's tester also uses, so
 *      both programs print the same text block / CSV row. Option names follow
 *      run too (only sgemm is supported here).
 *
 *      Run as a separate process from gemm.h (two thread pools in one process interfere).
 *
 * Build (no cblas.h needed; the few OpenBLAS functions used are declared below):
 *      make
 *      OpenBLAS built in your home directory (PREFIX=$HOME/opt/openblas) is the default.
 *      If only the runtime library exists (libopenblas.so.0, no libopenblas.so),
 *      replace "-lopenblas" in the Makefile with "-l:libopenblas.so.0".
 *
 * Usage:
 *      OPENBLAS_NUM_THREADS=28 taskset -c 0-27 ./openblas_ref -m 2048 -n 2048 -k 2048 -i 5
 *
 *      -m, -n, -k : matrix sizes (A: M x K, B: K x N, C: M x N, row-major)
 *      -i         : measured iterations (default 1)
 *      -w         : warm-up iterations, not measured (default 1)
 *      -b         : random value bound (default 5)
 *      -c         : also report the naive GEMM baseline and speedup
 *      -p         : accepted for compatibility with run (always printed here)
 *      -C, --csv  : print the result as a CSV row
**********************************************************************************************/

#define _POSIX_C_SOURCE 199309L

#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "report.h"

/* OpenBLAS declarations (same as its cblas.h), so the build does not need the header.
   Valid for the standard 32-bit-integer OpenBLAS (libopenblas, not libopenblas64). */
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
};
void cblas_sgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA,
                 enum CBLAS_TRANSPOSE transB, int M, int N, int K, float alpha,
                 const float* A, int lda, const float* B, int ldb, float beta,
                 float* C, int ldc);
char* openblas_get_corename(void);
int openblas_get_num_threads(void);

/* OpenBLAS reports a core name (e.g. SKYLAKEX). Print the SIMD level its sgemm
   kernel uses instead, so the ISA column means the same as in run. An unknown
   core is printed as it is. */
static const char* isa_of_core(const char* core) {
  static const struct {
    const char* core;
    const char* isa;
  } table[] = {{"SKYLAKEX", "AVX512F"},
               {"COOPERLAKE", "AVX512F"},
               {"SAPPHIRERAPIDS", "AVX512F"},
               {"HASWELL", "AVX2"},
               {"ZEN", "AVX2"},
               {"SANDYBRIDGE", "AVX"}};

  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
    if (strcasecmp(core, table[i].core) == 0)
      return table[i].isa;
  return core;
}

static uint64_t timer_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

/* Same distribution as gemm.h's fp32_get_rand_mat: integers in [-bound/2, bound) */
static void rand_mat(int row, int col, float* mat, int bound) {
  for (long i = 0; i < (long)row * col; i++) {
    float num = (float)(rand() % bound);
    mat[i] = (num <= bound / 2) ? -num : num;
  }
}

/* Same as gemm.h's naive_sgemm: reference result and baseline for speedup. */
static void naive_sgemm(const float* A, const float* B, float* C, const int M,
                        const int N, const int K) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float sum = 0;
      for (int k = 0; k < K; k++)
        sum += (A[r * K + k] * B[k * N + c]);
      C[r * N + c] = sum;
    }
  }
}

static int parse_int(const char* arg, const char* name) {
  char* end;
  long value = strtol(arg, &end, 10);
  if (*arg == '\0' || *end != '\0' || value < 0) {
    fprintf(stderr, "[Error]: Invalid integer for %s.\n", name);
    exit(EXIT_FAILURE);
  }
  return (int)value;
}

static void help(void) {
  fprintf(stderr, "OpenBLAS reference for gemm.h (sgemm only)\n");
  fprintf(stderr, "  -h, --help             Print this help message\n");
  fprintf(stderr, "  -i, --iter=<num>       Number of iteration  Default: 1\n");
  fprintf(stderr,
          "  -w, --warmup=<num>     Number of untimed warm-up runs  Default: "
          "1\n");
  fprintf(stderr,
          "  -m, --M=<size>         Matrix size M for C(MxN) = A(MxK) X B(KxN) "
          "Default: 1024\n");
  fprintf(stderr,
          "  -k, --K=<size>         Matrix size K for C(MxN) = A(MxK) X B(KxN) "
          "Default: 1024\n");
  fprintf(stderr,
          "  -n, --N=<size>         Matrix size N for C(MxN) = A(MxK) X B(KxN) "
          "Default: 1024\n");
  fprintf(stderr,
          "  -b, --bound=<num>      Bound for generating random matrix value "
          "Default: 5\n");
  fprintf(stderr,
          "  -p, --print            Accepted for compatibility with run\n");
  fprintf(
      stderr,
      "  -C, --csv              Print results as CSV, same columns as run\n");
  fprintf(stderr,
          "  -c, --compare          Also report naive GEMM and speedup\n");
}

int main(int argc, char** argv) {
  int M = 1024, N = 1024, K = 1024;
  int niter = 1, warmup = 1, bound = 5;
  int compare_flag = 0, csv_flag = 0;

  static struct option long_options[] = {{"M", required_argument, 0, 'm'},
                                         {"N", required_argument, 0, 'n'},
                                         {"K", required_argument, 0, 'k'},
                                         {"iter", required_argument, 0, 'i'},
                                         {"warmup", required_argument, 0, 'w'},
                                         {"bound", required_argument, 0, 'b'},
                                         {"print", no_argument, 0, 'p'},
                                         {"compare", no_argument, 0, 'c'},
                                         {"csv", no_argument, 0, 'C'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:n:k:i:w:b:pcCh", long_options,
                            NULL)) != -1) {
    switch (opt) {
      case 'm':
        M = parse_int(optarg, "--M");
        break;
      case 'n':
        N = parse_int(optarg, "--N");
        break;
      case 'k':
        K = parse_int(optarg, "--K");
        break;
      case 'i':
        niter = parse_int(optarg, "--iter");
        break;
      case 'w':
        warmup = parse_int(optarg, "--warmup");
        break;
      case 'b':
        bound = parse_int(optarg, "--bound");
        break;
      case 'p':
        break;
      case 'c':
        compare_flag = 1;
        break;
      case 'C':
        csv_flag = 1;
        break;
      case 'h':
        help();
        return EXIT_SUCCESS;
      default:
        fprintf(stderr, "[Error]: Unknown option.\n");
        fprintf(stderr, "Use --help for usage.\n");
        return EXIT_FAILURE;
    }
  }
  if (M <= 0 || N <= 0 || K <= 0 || niter <= 0 || bound <= 0) {
    fprintf(stderr, "[Error]: M, N, K, --iter and --bound must be positive.\n");
    return EXIT_FAILURE;
  }

  float* A = malloc(sizeof(float) * (size_t)M * K);
  float* B = malloc(sizeof(float) * (size_t)K * N);
  float* C = malloc(sizeof(float) * (size_t)M * N);
  float* C_ref = malloc(sizeof(float) * (size_t)M * N);
  if (!A || !B || !C || !C_ref) {
    fprintf(stderr, "malloc failed\n");
    return EXIT_FAILURE;
  }

  srand((unsigned)time(NULL));
  rand_mat(M, K, A, bound);
  rand_mat(K, N, B, bound);

  TestResult res = {.lib = "openblas",
                    .name = "sgemm",
                    .isa = isa_of_core(openblas_get_corename()),
                    .threads = openblas_get_num_threads(),
                    .M = M,
                    .K = K,
                    .N = N,
                    .niter = niter,
                    .warmup = warmup,
                    .compare = compare_flag,
                    .exec_time = {0, 0, 1e30},
                    .gflops = {0, 0, 1e30}};

  const double flop = 2.0 * M * N * K;

  for (int it = 0; it < warmup + niter; it++) {
    memset(C, 0, sizeof(float) * (size_t)M * N); /* outside the timed region */

    uint64_t start = timer_ns();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K,
                B, N, 1.0f, C, N); /* C += A * B, same as gemm.h */
    uint64_t end = timer_ns();

    if (it < warmup)
      continue; /* warm-up: thread pool creation etc. */

    double ms = (end - start) * 1e-6;
    double gflops = flop / ((end - start) * 1e-9) / 1e9;
    res.exec_time[0] += ms;
    if (res.exec_time[1] < ms)
      res.exec_time[1] = ms;
    if (res.exec_time[2] > ms)
      res.exec_time[2] = ms;
    res.gflops[0] += gflops;
    if (res.gflops[1] < gflops)
      res.gflops[1] = gflops;
    if (res.gflops[2] > gflops)
      res.gflops[2] = gflops;

    /* Naive baseline: run once to validate and (optionally) benchmark. */
    if (it == warmup) {
      uint64_t nstart = timer_ns();
      naive_sgemm(A, B, C_ref, M, N, K);
      uint64_t nend = timer_ns();
      double nelapsed = (nend - nstart) * 1e-9;
      res.naive_time = nelapsed * 1000;
      res.naive_gflops = (flop / nelapsed) / 1e9;

      res.is_valid = 1;
      for (size_t idx = 0; idx < (size_t)M * N; idx++)
        if (C[idx] != C_ref[idx]) {
          res.is_valid = 0;
          break;
        }
    }
  }

  /* keep the result alive so the call cannot be optimized away */
  volatile float sink = C[0] + C[(size_t)M * N - 1];
  (void)sink;

  if (!res.is_valid)
    fprintf(stderr, "[Invalid GEMM!] %s M=%d N=%d K=%d\n", res.name, M, N, K);
  print_result(stdout, &res, csv_flag);
  if (res.is_valid)
    fprintf(stderr, "[Summary] 1/1 GEMM results valid\n");
  else
    fprintf(stderr, "[Summary] 1/1 GEMM results INVALID\n");

  free(A);
  free(B);
  free(C);
  free(C_ref);
  return res.is_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

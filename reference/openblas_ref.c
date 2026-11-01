/**********************************************************************************************
 * File   : openblas_ref.c
 *
 * Description:
 *      OpenBLAS cblas_sgemm / cblas_dgemm reference benchmark for gemm.h.
 *      Measures under the same conditions as gemm.h's tester (run):
 *        - same semantics: C += A * B  (beta = 1, C reset to 0 before every call)
 *        - same FLOP count: 2 * M * N * K
 *        - same timer: CLOCK_MONOTONIC_RAW around the gemm call only
 *        - warm-up calls are excluded from the statistics
 *        - with -v the result is checked against a naive GEMM after the timed
 *          runs, like run does
 *
 *      The output comes from ../report.c, which gemm.h's tester also uses, so
 *      both programs print the same text block / CSV row. Option names follow
 *      run too (only sgemm and dgemm are supported here).
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
 *      -t         : data type, s (sgemm, default) or d (dgemm)
 *      -m, -n, -k : matrix sizes (A: M x K, B: K x N, C: M x N, row-major)
 *      -i         : measured iterations (default 1)
 *      -w         : warm-up iterations, not measured (default 1)
 *      -b         : random value bound (default 5)
 *      -p         : accepted for compatibility with run (always printed here)
 *      -C, --csv  : print the result as a CSV row
 *      -v         : check the result against the naive GEMM (slow for large sizes)
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
void cblas_dgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA,
                 enum CBLAS_TRANSPOSE transB, int M, int N, int K, double alpha,
                 const double* A, int lda, const double* B, int ldb,
                 double beta, double* C, int ldc);
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

/* The matrices are float (is_double = 0) or double (is_double = 1) arrays. */
static double at(const void* mat, size_t i, int is_double) {
  return is_double ? ((const double*)mat)[i] : (double)((const float*)mat)[i];
}

static void set(void* mat, size_t i, double v, int is_double) {
  if (is_double)
    ((double*)mat)[i] = v;
  else
    ((float*)mat)[i] = (float)v;
}

/* Same distribution as gemm.h's fp32_get_rand_mat: integers in [-bound/2, bound) */
static void rand_mat(int row, int col, void* mat, int bound, int is_double) {
  for (size_t i = 0; i < (size_t)row * col; i++) {
    int num = rand() % bound;
    set(mat, i, (num <= bound / 2) ? -num : num, is_double);
  }
}

/* The reference result for validation. The values are small integers, so the
   double accumulation is exact for both types. */
static void naive_gemm(const void* A, const void* B, void* C, int M, int N,
                       int K, int is_double) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      double sum = 0;
      for (int k = 0; k < K; k++)
        sum += at(A, (size_t)r * K + k, is_double) *
               at(B, (size_t)k * N + c, is_double);
      set(C, (size_t)r * N + c, sum, is_double);
    }
  }
}

/* C += A * B */
static void openblas_gemm(const void* A, const void* B, void* C, int M, int N,
                          int K, int is_double) {
  if (is_double)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K,
                B, N, 1.0, C, N);
  else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K,
                B, N, 1.0f, C, N);
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
  fprintf(stderr, "OpenBLAS reference for gemm.h (sgemm, dgemm)\n");
  fprintf(stderr, "  -h, --help             Print this help message\n");
  fprintf(stderr,
          "  -t, --type=<type>      Data type: s (sgemm) or d (dgemm)  "
          "Default: s\n");
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
          "  -v, --verify           Check the result against the naive GEMM "
          "after the timed runs (slow for large sizes)\n");
}

int main(int argc, char** argv) {
  int M = 1024, N = 1024, K = 1024;
  int niter = 1, warmup = 1, bound = 5;
  int csv_flag = 0, verify_flag = 0, is_double = 0;

  static struct option long_options[] = {{"type", required_argument, 0, 't'},
                                         {"M", required_argument, 0, 'm'},
                                         {"N", required_argument, 0, 'n'},
                                         {"K", required_argument, 0, 'k'},
                                         {"iter", required_argument, 0, 'i'},
                                         {"warmup", required_argument, 0, 'w'},
                                         {"bound", required_argument, 0, 'b'},
                                         {"print", no_argument, 0, 'p'},
                                         {"csv", no_argument, 0, 'C'},
                                         {"verify", no_argument, 0, 'v'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "t:m:n:k:i:w:b:pCvh", long_options,
                            NULL)) != -1) {
    switch (opt) {
      case 't':
        if (!strcmp(optarg, "s") || !strcmp(optarg, "f") ||
            !strcmp(optarg, "float"))
          is_double = 0;
        else if (!strcmp(optarg, "d") || !strcmp(optarg, "double"))
          is_double = 1;
        else {
          fprintf(stderr,
                  "[Error]: Only s (sgemm) and d (dgemm) are supported.\n");
          return EXIT_FAILURE;
        }
        break;
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
      case 'C':
        csv_flag = 1;
        break;
      case 'v':
        verify_flag = 1;
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

  const size_t esz = is_double ? sizeof(double) : sizeof(float);
  void* A = aligned_alloc(64, esz * (size_t)M * K);
  void* B = aligned_alloc(64, esz * (size_t)K * N);
  void* C = aligned_alloc(64, esz * (size_t)M * N);
  if (!A || !B || !C) {
    fprintf(stderr, "malloc failed\n");
    return EXIT_FAILURE;
  }

  srand((unsigned)time(NULL));
  rand_mat(M, K, A, bound, is_double);
  rand_mat(K, N, B, bound, is_double);

  TestResult res = {.lib = "openblas",
                    .name = is_double ? "dgemm" : "sgemm",
                    .isa = isa_of_core(openblas_get_corename()),
                    .threads = openblas_get_num_threads(),
                    .M = M,
                    .K = K,
                    .N = N,
                    .niter = niter,
                    .warmup = warmup,
                    .verify = verify_flag,
                    .exec_time = {0, 0, 1e30},
                    .gflops = {0, 0, 1e30}};

  const double flop = 2.0 * M * N * K;

  for (int it = 0; it < warmup + niter; it++) {
    memset(C, 0, esz * (size_t)M * N); /* outside the timed region */

    uint64_t start = timer_ns();
    openblas_gemm(A, B, C, M, N, K, is_double);
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
  }

  /* keep the result alive so the call cannot be optimized away */
  volatile double sink =
      at(C, 0, is_double) + at(C, (size_t)M * N - 1, is_double);
  (void)sink;

  /* The naive GEMM is slow, so it runs only on request and after the timed
     runs (before them it would leave the caches and the cores cold). */
  if (verify_flag) {
    void* C_ref = malloc(esz * (size_t)M * N);
    if (!C_ref) {
      fprintf(stderr, "malloc failed\n");
      return EXIT_FAILURE;
    }

    memset(C, 0, esz * (size_t)M * N);
    openblas_gemm(A, B, C, M, N, K, is_double);
    naive_gemm(A, B, C_ref, M, N, K, is_double);

    res.is_valid = 1;
    for (size_t idx = 0; idx < (size_t)M * N; idx++)
      if (at(C, idx, is_double) != at(C_ref, idx, is_double)) {
        res.is_valid = 0;
        break;
      }
    free(C_ref);
  }

  const int invalid = res.verify && !res.is_valid;

  if (invalid)
    fprintf(stderr, "[Invalid GEMM!] %s M=%d N=%d K=%d\n", res.name, M, N, K);
  print_result(stdout, &res, csv_flag);
  if (verify_flag)
    fprintf(stderr, "[Summary] %s\n",
            invalid ? "1/1 GEMM results INVALID" : "1/1 GEMM results valid");

  free(A);
  free(B);
  free(C);
  return invalid ? EXIT_FAILURE : EXIT_SUCCESS;
}

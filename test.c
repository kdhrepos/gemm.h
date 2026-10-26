#include "test.h"

/* SIMD level the kernel actually runs: the build's INSTLEVEL, capped at the
   highest level the kernel has a dedicated path for (s/d/i: AVX512F,
   hq/q: AVX512BW). */
static const char* isa_name(const int kernel_level) {
  const int level = (INSTLEVEL < kernel_level) ? INSTLEVEL : kernel_level;

  if (level >= INSTLEVEL_AVX512VNNI)
    return "AVX512VNNI";
  if (level >= INSTLEVEL_AVX512BW)
    return "AVX512BW";
  if (level >= INSTLEVEL_AVX512F)
    return "AVX512F";
  if (level >= INSTLEVEL_AVX2)
    return "AVX2";
  if (level >= INSTLEVEL_AVX)
    return "AVX";
  return "SSE";
}

/* Print one result to the enabled outputs; returns 1 if the GEMM is invalid. */
static int report(const TestResult* res, FILE* file, BOOL console_flag,
                  BOOL csv_flag) {
  if (!res->is_valid)
    fprintf(stderr, "[Invalid GEMM!] %s M=%d N=%d K=%d\n", res->name, res->M,
            res->N, res->K);
  if (console_flag)
    print_result(stdout, res, csv_flag);
  if (file != NULL)
    print_result(file, res, csv_flag);
  return !res->is_valid;
}

int sgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL compare_flag, BOOL csv_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        TestResult res = {.lib = "gemm.h",
                          .name = "sgemm",
                          .isa = isa_name(INSTLEVEL_AVX512F),
                          .threads = omp_get_max_threads(),
                          .M = m,
                          .K = k,
                          .N = n,
                          .niter = niter,
                          .warmup = warmup,
                          .compare = compare_flag,
                          .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                          .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

        float* A = (float*)malloc(m * k * sizeof(float));
        float* B = (float*)malloc(k * n * sizeof(float));
        float* C = (float*)malloc(m * n * sizeof(float));
        float* C_ref = (float*)malloc(m * n * sizeof(float));

        /* untimed warm-up runs (page faults, caches, threads) */
        if (warmup > 0) {
          fp32_get_rand_mat(m, k, A, bound);
          fp32_get_rand_mat(k, n, B, bound);
          for (int w = 0; w < warmup; w++) {
            memset(C, 0, sizeof(float) * m * n);
            sgemm(A, B, C, m, n, k);
          }
        }

        for (int i = 0; i < niter; i++) {
          memset(A, 0, sizeof(float) * m * k);
          memset(B, 0, sizeof(float) * k * n);
          memset(C, 0, sizeof(float) * m * n);

          fp32_get_rand_mat(m, k, A, bound);
          fp32_get_rand_mat(k, n, B, bound);

          double FLOP = 2 * (double)m * n * k;

          uint64_t start = timer();
          sgemm(A, B, C, m, n, k);
          uint64_t end = timer();
          double elapsed = (end - start) * 1e-9;
          double FLOPS = FLOP / elapsed;

          /* Naive baseline: run once to validate and (optionally) benchmark. */
          if (i == 0) {
            uint64_t nstart = timer();
            naive_sgemm(A, B, C_ref, m, n, k);
            uint64_t nend = timer();
            double nelapsed = (nend - nstart) * 1e-9;
            res.naive_time = nelapsed * 1000;
            res.naive_gflops = (FLOP / nelapsed) / 1e9;

            res.is_valid = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                res.is_valid = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          res.exec_time[0] += (elapsed * 1000);
          res.exec_time[1] =
              (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[1]);
          res.exec_time[2] =
              (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[2]);

          res.gflops[0] += (FLOPS / 1e9);
          res.gflops[1] =
              (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
          res.gflops[2] =
              (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        error_num += report(&res, file, console_flag, csv_flag);
      }
    }
  }

  return error_num;
}

int dgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL compare_flag, BOOL csv_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        TestResult res = {.lib = "gemm.h",
                          .name = "dgemm",
                          .isa = isa_name(INSTLEVEL_AVX512F),
                          .threads = omp_get_max_threads(),
                          .M = m,
                          .K = k,
                          .N = n,
                          .niter = niter,
                          .warmup = warmup,
                          .compare = compare_flag,
                          .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                          .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

        double* A = (double*)malloc(m * k * sizeof(double));
        double* B = (double*)malloc(k * n * sizeof(double));
        double* C = (double*)malloc(m * n * sizeof(double));
        double* C_ref = (double*)malloc(m * n * sizeof(double));

        /* untimed warm-up runs (page faults, caches, threads) */
        if (warmup > 0) {
          fp64_get_rand_mat(m, k, A, bound);
          fp64_get_rand_mat(k, n, B, bound);
          for (int w = 0; w < warmup; w++) {
            memset(C, 0, sizeof(double) * m * n);
            dgemm(A, B, C, m, n, k);
          }
        }

        for (int i = 0; i < niter; i++) {
          memset(A, 0, sizeof(double) * m * k);
          memset(B, 0, sizeof(double) * k * n);
          memset(C, 0, sizeof(double) * m * n);

          fp64_get_rand_mat(m, k, A, bound);
          fp64_get_rand_mat(k, n, B, bound);

          double FLOP = 2 * (double)m * n * k;

          uint64_t start = timer();
          dgemm(A, B, C, m, n, k);
          uint64_t end = timer();
          double elapsed = (end - start) * 1e-9;
          double FLOPS = FLOP / elapsed;

          /* Naive baseline: run once to validate and (optionally) benchmark. */
          if (i == 0) {
            uint64_t nstart = timer();
            naive_dgemm(A, B, C_ref, m, n, k);
            uint64_t nend = timer();
            double nelapsed = (nend - nstart) * 1e-9;
            res.naive_time = nelapsed * 1000;
            res.naive_gflops = (FLOP / nelapsed) / 1e9;

            res.is_valid = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                res.is_valid = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          res.exec_time[0] += (elapsed * 1000);
          res.exec_time[1] =
              (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[1]);
          res.exec_time[2] =
              (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[2]);

          res.gflops[0] += (FLOPS / 1e9);
          res.gflops[1] =
              (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
          res.gflops[2] =
              (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        error_num += report(&res, file, console_flag, csv_flag);
      }
    }
  }

  return error_num;
}

int igemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL compare_flag, BOOL csv_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        TestResult res = {.lib = "gemm.h",
                          .name = "igemm",
                          .isa = isa_name(INSTLEVEL_AVX512F),
                          .threads = omp_get_max_threads(),
                          .M = m,
                          .K = k,
                          .N = n,
                          .niter = niter,
                          .warmup = warmup,
                          .compare = compare_flag,
                          .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                          .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

        int* A = (int*)malloc(m * k * sizeof(int));
        int* B = (int*)malloc(k * n * sizeof(int));
        int* C = (int*)malloc(m * n * sizeof(int));
        int* C_ref = (int*)malloc(m * n * sizeof(int));

        /* untimed warm-up runs (page faults, caches, threads) */
        if (warmup > 0) {
          int32_get_rand_mat(m, k, A, bound);
          int32_get_rand_mat(k, n, B, bound);
          for (int w = 0; w < warmup; w++) {
            memset(C, 0, sizeof(int) * m * n);
            igemm(A, B, C, m, n, k);
          }
        }

        for (int i = 0; i < niter; i++) {
          memset(A, 0, sizeof(int) * m * k);
          memset(B, 0, sizeof(int) * k * n);
          memset(C, 0, sizeof(int) * m * n);

          int32_get_rand_mat(m, k, A, bound);
          int32_get_rand_mat(k, n, B, bound);

          double FLOP = 2 * (double)m * n * k;

          uint64_t start = timer();
          igemm(A, B, C, m, n, k);
          uint64_t end = timer();
          double elapsed = (end - start) * 1e-9;
          double FLOPS = FLOP / elapsed;

          /* Naive baseline: run once to validate and (optionally) benchmark. */
          if (i == 0) {
            uint64_t nstart = timer();
            naive_igemm(A, B, C_ref, m, n, k);
            uint64_t nend = timer();
            double nelapsed = (nend - nstart) * 1e-9;
            res.naive_time = nelapsed * 1000;
            res.naive_gflops = (FLOP / nelapsed) / 1e9;

            res.is_valid = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                res.is_valid = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          res.exec_time[0] += (elapsed * 1000);
          res.exec_time[1] =
              (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[1]);
          res.exec_time[2] =
              (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[2]);

          res.gflops[0] += (FLOPS / 1e9);
          res.gflops[1] =
              (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
          res.gflops[2] =
              (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        error_num += report(&res, file, console_flag, csv_flag);
      }
    }
  }

  return error_num;
}

int hqgemm_test(const int M, const int N, const int K, const int niter,
                const int warmup, const int range, const int bound, FILE* file,
                BOOL console_flag, BOOL compare_flag, BOOL csv_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        TestResult res = {.lib = "gemm.h",
                          .name = "hqgemm",
                          .isa = isa_name(INSTLEVEL_AVX512BW),
                          .threads = omp_get_max_threads(),
                          .M = m,
                          .K = k,
                          .N = n,
                          .niter = niter,
                          .warmup = warmup,
                          .compare = compare_flag,
                          .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                          .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

        int16_t* A = (int16_t*)malloc(m * k * sizeof(int16_t));
        int16_t* B = (int16_t*)malloc(k * n * sizeof(int16_t));
        int16_t* C = (int16_t*)malloc(m * n * sizeof(int16_t));
        int16_t* C_ref = (int16_t*)malloc(m * n * sizeof(int16_t));

        /* untimed warm-up runs (page faults, caches, threads) */
        if (warmup > 0) {
          int16_get_rand_mat(m, k, A, bound);
          int16_get_rand_mat(k, n, B, bound);
          for (int w = 0; w < warmup; w++) {
            memset(C, 0, sizeof(int16_t) * m * n);
            hqgemm(A, B, C, m, n, k);
          }
        }

        for (int i = 0; i < niter; i++) {
          memset(A, 0, sizeof(int16_t) * m * k);
          memset(B, 0, sizeof(int16_t) * k * n);
          memset(C, 0, sizeof(int16_t) * m * n);

          int16_get_rand_mat(m, k, A, bound);
          int16_get_rand_mat(k, n, B, bound);

          double FLOP = 2 * (double)m * n * k;

          uint64_t start = timer();
          hqgemm(A, B, C, m, n, k);
          uint64_t end = timer();
          double elapsed = (end - start) * 1e-9;
          double FLOPS = FLOP / elapsed;

          /* Naive baseline: run once to validate and (optionally) benchmark. */
          if (i == 0) {
            uint64_t nstart = timer();
            naive_hqgemm(A, B, C_ref, m, n, k);
            uint64_t nend = timer();
            double nelapsed = (nend - nstart) * 1e-9;
            res.naive_time = nelapsed * 1000;
            res.naive_gflops = (FLOP / nelapsed) / 1e9;

            res.is_valid = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                res.is_valid = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          res.exec_time[0] += (elapsed * 1000);
          res.exec_time[1] =
              (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[1]);
          res.exec_time[2] =
              (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[2]);

          res.gflops[0] += (FLOPS / 1e9);
          res.gflops[1] =
              (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
          res.gflops[2] =
              (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        error_num += report(&res, file, console_flag, csv_flag);
      }
    }
  }

  return error_num;
}

int qgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL compare_flag, BOOL csv_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        TestResult res = {.lib = "gemm.h",
                          .name = "qgemm",
                          .isa = isa_name(INSTLEVEL_AVX512BW),
                          .threads = omp_get_max_threads(),
                          .M = m,
                          .K = k,
                          .N = n,
                          .niter = niter,
                          .warmup = warmup,
                          .compare = compare_flag,
                          .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                          .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

        int8_t* A = (int8_t*)malloc(m * k * sizeof(int8_t));
        int8_t* B = (int8_t*)malloc(k * n * sizeof(int8_t));
        int8_t* C = (int8_t*)malloc(m * n * sizeof(int8_t));
        int8_t* C_ref = (int8_t*)malloc(m * n * sizeof(int8_t));

        /* untimed warm-up runs (page faults, caches, threads) */
        if (warmup > 0) {
          int8_get_rand_mat(m, k, A, bound);
          int8_get_rand_mat(k, n, B, bound);
          for (int w = 0; w < warmup; w++) {
            memset(C, 0, sizeof(int8_t) * m * n);
            qgemm(A, B, C, m, n, k);
          }
        }

        for (int i = 0; i < niter; i++) {
          memset(A, 0, sizeof(int8_t) * m * k);
          memset(B, 0, sizeof(int8_t) * k * n);
          memset(C, 0, sizeof(int8_t) * m * n);

          int8_get_rand_mat(m, k, A, bound);
          int8_get_rand_mat(k, n, B, bound);

          double FLOP = 2 * (double)m * n * k;

          uint64_t start = timer();
          qgemm(A, B, C, m, n, k);
          uint64_t end = timer();
          double elapsed = (end - start) * 1e-9;
          double FLOPS = FLOP / elapsed;

          /* Naive baseline: run once to validate and (optionally) benchmark. */
          if (i == 0) {
            uint64_t nstart = timer();
            naive_qgemm(A, B, C_ref, m, n, k);
            uint64_t nend = timer();
            double nelapsed = (nend - nstart) * 1e-9;
            res.naive_time = nelapsed * 1000;
            res.naive_gflops = (FLOP / nelapsed) / 1e9;

            res.is_valid = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                res.is_valid = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          res.exec_time[0] += (elapsed * 1000);
          res.exec_time[1] =
              (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[1]);
          res.exec_time[2] =
              (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                   : res.exec_time[2]);

          res.gflops[0] += (FLOPS / 1e9);
          res.gflops[1] =
              (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
          res.gflops[2] =
              (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        error_num += report(&res, file, console_flag, csv_flag);
      }
    }
  }

  return error_num;
}

uint64_t timer() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

/********************************************************
 *                                                      
 *          Naive GEMM
 *                                                      
*********************************************************/
void naive_sgemm(const float* A, const float* B, float* C, const int M,
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

void naive_dgemm(const double* A, const double* B, double* C, const int M,
                 const int N, const int K) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      double sum = 0;
      for (int k = 0; k < K; k++)
        sum += (A[r * K + k] * B[k * N + c]);
      C[r * N + c] = sum;
    }
  }
}

void naive_igemm(const int* A, const int* B, int* C, const int M, const int N,
                 const int K) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      int sum = 0;
      for (int k = 0; k < K; k++)
        sum += (A[r * K + k] * B[k * N + c]);
      C[r * N + c] = sum;
    }
  }
}

void naive_hqgemm(const int16_t* A, const int16_t* B, int16_t* C, const int M,
                  const int N, const int K) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      int16_t sum = 0;
      for (int k = 0; k < K; k++)
        sum += (A[r * K + k] * B[k * N + c]);
      C[r * N + c] = sum;
    }
  }
}

void naive_qgemm(const int8_t* A, const int8_t* B, int8_t* C, const int M,
                 const int N, const int K) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      int8_t sum = 0;
      for (int k = 0; k < K; k++)
        sum += (A[r * K + k] * B[k * N + c]);
      C[r * N + c] = sum;
    }
  }
}

/********************************************************
 *                                                      
 *          Generate Random Matrix                                 
 *                                                      
*********************************************************/
void int32_get_rand_mat(int row, int col, int32_t* mat, int bound) {
  srand(time(NULL));
  for (int r = 0; r < row; r++)
    for (int c = 0; c < col; c++) {
      int32_t num = rand() % bound;
      mat[r * col + c] = num <= bound / 2 ? -num : num;
    }
}

void fp32_get_rand_mat(int row, int col, float* mat, int bound) {
  srand(time(NULL));
  for (int r = 0; r < row; r++)
    for (int c = 0; c < col; c++) {
      float num = rand() % bound;
      mat[r * col + c] = num <= bound / 2 ? -num : num;
    }
}

void fp64_get_rand_mat(int row, int col, double* mat, int bound) {
  srand(time(NULL));
  for (int r = 0; r < row; r++)
    for (int c = 0; c < col; c++) {
      double num = rand() % bound;
      mat[r * col + c] = num <= bound / 2 ? -num : num;
    }
}

void int16_get_rand_mat(int row, int col, int16_t* mat, int bound) {
  srand(time(NULL));
  for (int r = 0; r < row; r++)
    for (int c = 0; c < col; c++) {
      int16_t num = rand() % bound;
      mat[r * col + c] = num <= bound / 2 ? -num : num;
    }
}

void int8_get_rand_mat(int row, int col, int8_t* mat, int bound) {
  srand(time(NULL));
  for (int r = 0; r < row; r++)
    for (int c = 0; c < col; c++) {
      int8_t num = rand() % bound;
      mat[r * col + c] = num <= bound / 2 ? -num : num;
    }
}

/********************************************************
 *                                                      
 *          Matrix Print                                
 *                                                      
*********************************************************/
void int32_print(int row, int col, int32_t* mat) {
  printf("INT32 Matrix Print\n");
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++)
      printf("%3d ", mat[r * col + c]);
    printf("\n");
  }
  printf("\n");
}

void fp32_print(int row, int col, float* mat) {
  printf("FP32 Matrix Print\n");
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++)
      printf("%5.2f ", mat[r * col + c]);
    printf("\n");
  }
  printf("\n");
}

void fp64_print(int row, int col, double* mat) {
  printf("FP64 Matrix Print\n");
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++)
      printf("%5.2lf ", mat[r * col + c]);
    printf("\n");
  }
  printf("\n");
}

#include "test.h"

void sgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag,
                BOOL compare_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3] = {0, __FLT_MIN__, __FLT_MAX__}; /* avg max min */
        double gflops[3] = {0, __FLT_MIN__, __FLT_MAX__};     /* avg max min */
        double naive_time = 0, naive_gflops = 0;

        float* A = (float*)malloc(m * k * sizeof(float));
        float* B = (float*)malloc(k * n * sizeof(float));
        float* C = (float*)malloc(m * n * sizeof(float));
        float* C_ref = (float*)malloc(m * n * sizeof(float));

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
            naive_time = nelapsed * 1000;
            naive_gflops = (FLOP / nelapsed) / 1e9;

            is_valid_gemm = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                is_valid_gemm = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          exec_times[0] += (elapsed * 1000);
          exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[1]);
          exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[2]);

          gflops[0] += (FLOPS / 1e9);
          gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
          gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        if (!is_valid_gemm)
          error_num++;
        if (console_flag)
          print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm,
                        compare_flag, naive_time, naive_gflops);
        if (file != NULL)
          print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file,
                     compare_flag, naive_time, naive_gflops);
      }
    }
  }
}

void dgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag,
                BOOL compare_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3] = {0, __FLT_MIN__, __FLT_MAX__}; /* avg max min */
        double gflops[3] = {0, __FLT_MIN__, __FLT_MAX__};     /* avg max min */
        double naive_time = 0, naive_gflops = 0;

        double* A = (double*)malloc(m * k * sizeof(double));
        double* B = (double*)malloc(k * n * sizeof(double));
        double* C = (double*)malloc(m * n * sizeof(double));
        double* C_ref = (double*)malloc(m * n * sizeof(double));

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
            naive_time = nelapsed * 1000;
            naive_gflops = (FLOP / nelapsed) / 1e9;

            is_valid_gemm = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                is_valid_gemm = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          exec_times[0] += (elapsed * 1000);
          exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[1]);
          exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[2]);

          gflops[0] += (FLOPS / 1e9);
          gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
          gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        if (!is_valid_gemm)
          error_num++;
        if (console_flag)
          print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm,
                        compare_flag, naive_time, naive_gflops);
        if (file != NULL)
          print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file,
                     compare_flag, naive_time, naive_gflops);
      }
    }
  }
}

void igemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag,
                BOOL compare_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3] = {0, __FLT_MIN__, __FLT_MAX__}; /* avg max min */
        double gflops[3] = {0, __FLT_MIN__, __FLT_MAX__};     /* avg max min */
        double naive_time = 0, naive_gflops = 0;

        int* A = (int*)malloc(m * k * sizeof(int));
        int* B = (int*)malloc(k * n * sizeof(int));
        int* C = (int*)malloc(m * n * sizeof(int));
        int* C_ref = (int*)malloc(m * n * sizeof(int));

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
            naive_time = nelapsed * 1000;
            naive_gflops = (FLOP / nelapsed) / 1e9;

            is_valid_gemm = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                is_valid_gemm = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          exec_times[0] += (elapsed * 1000);
          exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[1]);
          exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[2]);

          gflops[0] += (FLOPS / 1e9);
          gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
          gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        if (!is_valid_gemm)
          error_num++;
        if (console_flag)
          print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm,
                        compare_flag, naive_time, naive_gflops);
        if (file != NULL)
          print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file,
                     compare_flag, naive_time, naive_gflops);
      }
    }
  }
}

void hqgemm_test(const int M, const int N, const int K, const int niter,
                 const int range, const int bound, FILE* file,
                 BOOL console_flag, BOOL compare_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3] = {0, __FLT_MIN__, __FLT_MAX__}; /* avg max min */
        double gflops[3] = {0, __FLT_MIN__, __FLT_MAX__};     /* avg max min */
        double naive_time = 0, naive_gflops = 0;

        int16_t* A = (int16_t*)malloc(m * k * sizeof(int16_t));
        int16_t* B = (int16_t*)malloc(k * n * sizeof(int16_t));
        int16_t* C = (int16_t*)malloc(m * n * sizeof(int16_t));
        int16_t* C_ref = (int16_t*)malloc(m * n * sizeof(int16_t));

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
            naive_time = nelapsed * 1000;
            naive_gflops = (FLOP / nelapsed) / 1e9;

            is_valid_gemm = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                is_valid_gemm = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          exec_times[0] += (elapsed * 1000);
          exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[1]);
          exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[2]);

          gflops[0] += (FLOPS / 1e9);
          gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
          gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        if (!is_valid_gemm)
          error_num++;
        if (console_flag)
          print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm,
                        compare_flag, naive_time, naive_gflops);
        if (file != NULL)
          print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file,
                     compare_flag, naive_time, naive_gflops);
      }
    }
  }
}

void qgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag,
                BOOL compare_flag) {
  int error_num = 0;

  for (int m = M; m < (M + range); m++) {
    for (int n = N; n < (N + range); n++) {
      for (int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3] = {0, __FLT_MIN__, __FLT_MAX__}; /* avg max min */
        double gflops[3] = {0, __FLT_MIN__, __FLT_MAX__};     /* avg max min */
        double naive_time = 0, naive_gflops = 0;

        int8_t* A = (int8_t*)malloc(m * k * sizeof(int8_t));
        int8_t* B = (int8_t*)malloc(k * n * sizeof(int8_t));
        int8_t* C = (int8_t*)malloc(m * n * sizeof(int8_t));
        int8_t* C_ref = (int8_t*)malloc(m * n * sizeof(int8_t));

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
            naive_time = nelapsed * 1000;
            naive_gflops = (FLOP / nelapsed) / 1e9;

            is_valid_gemm = TRUE;
            for (int idx = 0; idx < m * n; idx++)
              if (C[idx] != C_ref[idx]) {
                is_valid_gemm = FALSE;
                break;
              }
          }

          // if range is not 0, don't print each results
          if (range == 0) {
            printf("Exec. time = %.3lfms\n", elapsed * 1000);
            printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
          }

          exec_times[0] += (elapsed * 1000);
          exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[1]);
          exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : exec_times[2]);

          gflops[0] += (FLOPS / 1e9);
          gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
          gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);
        free(C_ref);

        if (!is_valid_gemm)
          error_num++;
        if (console_flag)
          print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm,
                        compare_flag, naive_time, naive_gflops);
        if (file != NULL)
          print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file,
                     compare_flag, naive_time, naive_gflops);
      }
    }
  }
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

void print_console(const int M, const int K, const int N, const int niter,
                   const double* exec_time, const double* gflops,
                   const BOOL is_valid_gemm, const BOOL compare_flag,
                   const double naive_time, const double naive_gflops) {
  printf("════════════════════════════════════════════════\n\n");
  printf("A: %dx%d B: %dx%d C: %dx%d\n\n", M, K, K, N, M, N);
  printf("GEMM Test on %d Iterations\n", niter);
  printf("%s\n\n",
         (is_valid_gemm == TRUE) ? "[Valid GEMM]" : "[Invalid GEMM!]");

  printf("AVG Exec. Time: %5.3lfms\n", exec_time[0] / niter);
  printf("MAX Exec. Time: %5.3lfms\n", exec_time[1]);
  printf("MIN Exec. Time: %5.3lfms\n\n", exec_time[2]);

  printf("AVG GFLOPS : %5.3lf\n", gflops[0] / niter);
  printf("MAX GFLOPS : %5.3lf\n", gflops[1]);
  printf("MIN GFLOPS : %5.3lf\n\n", gflops[2]);

  if (compare_flag) {
    double simd_avg = exec_time[0] / niter;
    printf("── vs Naive GEMM ──\n");
    printf("Naive Exec. Time: %5.3lfms\n", naive_time);
    printf("Naive GFLOPS : %5.3lf\n", naive_gflops);
    printf("Speedup (naive / simd avg): %5.3lfx\n\n",
           (simd_avg > 0) ? (naive_time / simd_avg) : 0);
  }
  printf("════════════════════════════════════════════════\n");
}

void print_file(const int M, const int K, const int N, const int niter,
                const double* exec_time, const double* gflops,
                const BOOL is_valid_gemm, FILE* file, const BOOL compare_flag,
                const double naive_time, const double naive_gflops) {
  fprintf(file, "════════════════════════════════════════════════\n\n");
  fprintf(file, "A: %dx%d B: %dx%d C: %dx%d\n\n", M, K, K, N, M, N);
  fprintf(file, "GEMM Test on %d Iterations\n", niter);
  fprintf(file, "%s\n\n",
          (is_valid_gemm == TRUE) ? "[Valid GEMM]" : "[Invalid GEMM!]");

  fprintf(file, "AVG Exec. Time: %5.3lfms\n", exec_time[0] / niter);
  fprintf(file, "MAX Exec. Time: %5.3lfms\n", exec_time[1]);
  fprintf(file, "MIN Exec. Time: %5.3lfms\n\n", exec_time[2]);

  fprintf(file, "AVG GFLOPS : %5.3lf\n", gflops[0] / niter);
  fprintf(file, "MAX GFLOPS : %5.3lf\n", gflops[1]);
  fprintf(file, "MIN GFLOPS : %5.3lf\n\n", gflops[2]);

  if (compare_flag) {
    double simd_avg = exec_time[0] / niter;
    fprintf(file, "── vs Naive GEMM ──\n");
    fprintf(file, "Naive Exec. Time: %5.3lfms\n", naive_time);
    fprintf(file, "Naive GFLOPS : %5.3lf\n", naive_gflops);
    fprintf(file, "Speedup (naive / simd avg): %5.3lfx\n\n",
            (simd_avg > 0) ? (naive_time / simd_avg) : 0);
  }
  fprintf(file, "════════════════════════════════════════════════\n");
}
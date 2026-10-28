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
  const int invalid = res->verify && !res->is_valid;

  if (invalid)
    fprintf(stderr, "[Invalid GEMM!] %s M=%d N=%d K=%d\n", res->name, res->M,
            res->N, res->K);
  if (console_flag)
    print_result(stdout, res, csv_flag);
  if (file != NULL)
    print_result(file, res, csv_flag);
  return invalid;
}

/* 64-byte aligned matrix. A malloc'd block starts 16 bytes past a cache line,
   so every vector load in packing and in the C update splits two lines. */
static void* alloc_mat(size_t bytes) {
  return aligned_alloc(MEM_ALIGN,
                       (bytes + MEM_ALIGN - 1) / MEM_ALIGN * MEM_ALIGN);
}

int sgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL csv_flag, BOOL verify_flag) {
  float* A = alloc_mat(sizeof(float) * (size_t)M * K);
  float* B = alloc_mat(sizeof(float) * (size_t)K * N);
  float* C = alloc_mat(sizeof(float) * (size_t)M * N);

  fp32_get_rand_mat(M, K, A, bound);
  fp32_get_rand_mat(K, N, B, bound);
  memset(C, 0, sizeof(float) * (size_t)M * N);

  const double FLOP = 2.0 * M * N * K;

  TestResult res = {.lib = "gemm.h",
                    .name = "sgemm",
                    .isa = isa_name(INSTLEVEL_AVX512F),
                    .threads = omp_get_max_threads(),
                    .M = M,
                    .K = K,
                    .N = N,
                    .niter = niter,
                    .warmup = warmup,
                    .verify = verify_flag,
                    .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                    .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

  /* warm-up runs (page faults, caches, threads) are not measured */
  for (int w = 0; w < warmup; w++)
    sgemm(A, B, C, M, N, K);

  for (int i = 0; i < niter; i++) {
    /* C += A * B, so C is reset before every run (outside the timer) */
    memset(C, 0, sizeof(float) * (size_t)M * N);

    uint64_t start = timer();
    sgemm(A, B, C, M, N, K);
    uint64_t end = timer();

    double elapsed = (end - start) * 1e-9;
    double FLOPS = FLOP / elapsed;

    /* if range is not 0, don't print each results */
    if (range == 0) {
      printf("Exec. time = %.3lfms\n", elapsed * 1000);
      printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
    }

    res.exec_time[0] += (elapsed * 1000);
    res.exec_time[1] = (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[1]);
    res.exec_time[2] = (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[2]);

    res.gflops[0] += (FLOPS / 1e9);
    res.gflops[1] =
        (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
    res.gflops[2] =
        (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
  }

  /* The naive GEMM is slow, so it runs only on request and after the timed
     runs (before them it would leave the caches and the cores cold). */
  if (verify_flag) {
    float* C_ref = alloc_mat(sizeof(float) * (size_t)M * N);

    memset(C, 0, sizeof(float) * (size_t)M * N);
    sgemm(A, B, C, M, N, K);
    naive_sgemm(A, B, C_ref, M, N, K);

    res.is_valid = TRUE;
    for (size_t idx = 0; idx < (size_t)M * N; idx++)
      if (C[idx] != C_ref[idx]) {
        res.is_valid = FALSE;
        break;
      }
    free(C_ref);
  }

  free(A);
  free(B);
  free(C);

  return report(&res, file, console_flag, csv_flag);
}

int dgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL csv_flag, BOOL verify_flag) {
  double* A = alloc_mat(sizeof(double) * (size_t)M * K);
  double* B = alloc_mat(sizeof(double) * (size_t)K * N);
  double* C = alloc_mat(sizeof(double) * (size_t)M * N);

  fp64_get_rand_mat(M, K, A, bound);
  fp64_get_rand_mat(K, N, B, bound);
  memset(C, 0, sizeof(double) * (size_t)M * N);

  const double FLOP = 2.0 * M * N * K;

  TestResult res = {.lib = "gemm.h",
                    .name = "dgemm",
                    .isa = isa_name(INSTLEVEL_AVX512F),
                    .threads = omp_get_max_threads(),
                    .M = M,
                    .K = K,
                    .N = N,
                    .niter = niter,
                    .warmup = warmup,
                    .verify = verify_flag,
                    .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                    .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

  /* warm-up runs (page faults, caches, threads) are not measured */
  for (int w = 0; w < warmup; w++)
    dgemm(A, B, C, M, N, K);

  for (int i = 0; i < niter; i++) {
    /* C += A * B, so C is reset before every run (outside the timer) */
    memset(C, 0, sizeof(double) * (size_t)M * N);

    uint64_t start = timer();
    dgemm(A, B, C, M, N, K);
    uint64_t end = timer();

    double elapsed = (end - start) * 1e-9;
    double FLOPS = FLOP / elapsed;

    /* if range is not 0, don't print each results */
    if (range == 0) {
      printf("Exec. time = %.3lfms\n", elapsed * 1000);
      printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
    }

    res.exec_time[0] += (elapsed * 1000);
    res.exec_time[1] = (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[1]);
    res.exec_time[2] = (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[2]);

    res.gflops[0] += (FLOPS / 1e9);
    res.gflops[1] =
        (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
    res.gflops[2] =
        (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
  }

  /* The naive GEMM is slow, so it runs only on request and after the timed
     runs (before them it would leave the caches and the cores cold). */
  if (verify_flag) {
    double* C_ref = alloc_mat(sizeof(double) * (size_t)M * N);

    memset(C, 0, sizeof(double) * (size_t)M * N);
    dgemm(A, B, C, M, N, K);
    naive_dgemm(A, B, C_ref, M, N, K);

    res.is_valid = TRUE;
    for (size_t idx = 0; idx < (size_t)M * N; idx++)
      if (C[idx] != C_ref[idx]) {
        res.is_valid = FALSE;
        break;
      }
    free(C_ref);
  }

  free(A);
  free(B);
  free(C);

  return report(&res, file, console_flag, csv_flag);
}

int igemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL csv_flag, BOOL verify_flag) {
  int* A = alloc_mat(sizeof(int) * (size_t)M * K);
  int* B = alloc_mat(sizeof(int) * (size_t)K * N);
  int* C = alloc_mat(sizeof(int) * (size_t)M * N);

  int32_get_rand_mat(M, K, A, bound);
  int32_get_rand_mat(K, N, B, bound);
  memset(C, 0, sizeof(int) * (size_t)M * N);

  const double FLOP = 2.0 * M * N * K;

  TestResult res = {.lib = "gemm.h",
                    .name = "igemm",
                    .isa = isa_name(INSTLEVEL_AVX512F),
                    .threads = omp_get_max_threads(),
                    .M = M,
                    .K = K,
                    .N = N,
                    .niter = niter,
                    .warmup = warmup,
                    .verify = verify_flag,
                    .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                    .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

  /* warm-up runs (page faults, caches, threads) are not measured */
  for (int w = 0; w < warmup; w++)
    igemm(A, B, C, M, N, K);

  for (int i = 0; i < niter; i++) {
    /* C += A * B, so C is reset before every run (outside the timer) */
    memset(C, 0, sizeof(int) * (size_t)M * N);

    uint64_t start = timer();
    igemm(A, B, C, M, N, K);
    uint64_t end = timer();

    double elapsed = (end - start) * 1e-9;
    double FLOPS = FLOP / elapsed;

    /* if range is not 0, don't print each results */
    if (range == 0) {
      printf("Exec. time = %.3lfms\n", elapsed * 1000);
      printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
    }

    res.exec_time[0] += (elapsed * 1000);
    res.exec_time[1] = (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[1]);
    res.exec_time[2] = (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[2]);

    res.gflops[0] += (FLOPS / 1e9);
    res.gflops[1] =
        (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
    res.gflops[2] =
        (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
  }

  /* The naive GEMM is slow, so it runs only on request and after the timed
     runs (before them it would leave the caches and the cores cold). */
  if (verify_flag) {
    int* C_ref = alloc_mat(sizeof(int) * (size_t)M * N);

    memset(C, 0, sizeof(int) * (size_t)M * N);
    igemm(A, B, C, M, N, K);
    naive_igemm(A, B, C_ref, M, N, K);

    res.is_valid = TRUE;
    for (size_t idx = 0; idx < (size_t)M * N; idx++)
      if (C[idx] != C_ref[idx]) {
        res.is_valid = FALSE;
        break;
      }
    free(C_ref);
  }

  free(A);
  free(B);
  free(C);

  return report(&res, file, console_flag, csv_flag);
}

int hqgemm_test(const int M, const int N, const int K, const int niter,
                const int warmup, const int range, const int bound, FILE* file,
                BOOL console_flag, BOOL csv_flag, BOOL verify_flag) {
  int16_t* A = alloc_mat(sizeof(int16_t) * (size_t)M * K);
  int16_t* B = alloc_mat(sizeof(int16_t) * (size_t)K * N);
  int16_t* C = alloc_mat(sizeof(int16_t) * (size_t)M * N);

  int16_get_rand_mat(M, K, A, bound);
  int16_get_rand_mat(K, N, B, bound);
  memset(C, 0, sizeof(int16_t) * (size_t)M * N);

  const double FLOP = 2.0 * M * N * K;

  TestResult res = {.lib = "gemm.h",
                    .name = "hqgemm",
                    .isa = isa_name(INSTLEVEL_AVX512BW),
                    .threads = omp_get_max_threads(),
                    .M = M,
                    .K = K,
                    .N = N,
                    .niter = niter,
                    .warmup = warmup,
                    .verify = verify_flag,
                    .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                    .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

  /* warm-up runs (page faults, caches, threads) are not measured */
  for (int w = 0; w < warmup; w++)
    hqgemm(A, B, C, M, N, K);

  for (int i = 0; i < niter; i++) {
    /* C += A * B, so C is reset before every run (outside the timer) */
    memset(C, 0, sizeof(int16_t) * (size_t)M * N);

    uint64_t start = timer();
    hqgemm(A, B, C, M, N, K);
    uint64_t end = timer();

    double elapsed = (end - start) * 1e-9;
    double FLOPS = FLOP / elapsed;

    /* if range is not 0, don't print each results */
    if (range == 0) {
      printf("Exec. time = %.3lfms\n", elapsed * 1000);
      printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
    }

    res.exec_time[0] += (elapsed * 1000);
    res.exec_time[1] = (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[1]);
    res.exec_time[2] = (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[2]);

    res.gflops[0] += (FLOPS / 1e9);
    res.gflops[1] =
        (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
    res.gflops[2] =
        (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
  }

  /* The naive GEMM is slow, so it runs only on request and after the timed
     runs (before them it would leave the caches and the cores cold). */
  if (verify_flag) {
    int16_t* C_ref = alloc_mat(sizeof(int16_t) * (size_t)M * N);

    memset(C, 0, sizeof(int16_t) * (size_t)M * N);
    hqgemm(A, B, C, M, N, K);
    naive_hqgemm(A, B, C_ref, M, N, K);

    res.is_valid = TRUE;
    for (size_t idx = 0; idx < (size_t)M * N; idx++)
      if (C[idx] != C_ref[idx]) {
        res.is_valid = FALSE;
        break;
      }
    free(C_ref);
  }

  free(A);
  free(B);
  free(C);

  return report(&res, file, console_flag, csv_flag);
}

int qgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL csv_flag, BOOL verify_flag) {
  int8_t* A = alloc_mat(sizeof(int8_t) * (size_t)M * K);
  int8_t* B = alloc_mat(sizeof(int8_t) * (size_t)K * N);
  int8_t* C = alloc_mat(sizeof(int8_t) * (size_t)M * N);

  int8_get_rand_mat(M, K, A, bound);
  int8_get_rand_mat(K, N, B, bound);
  memset(C, 0, sizeof(int8_t) * (size_t)M * N);

  const double FLOP = 2.0 * M * N * K;

  TestResult res = {.lib = "gemm.h",
                    .name = "qgemm",
                    .isa = isa_name(INSTLEVEL_AVX512BW),
                    .threads = omp_get_max_threads(),
                    .M = M,
                    .K = K,
                    .N = N,
                    .niter = niter,
                    .warmup = warmup,
                    .verify = verify_flag,
                    .exec_time = {0, __FLT_MIN__, __FLT_MAX__},
                    .gflops = {0, __FLT_MIN__, __FLT_MAX__}};

  /* warm-up runs (page faults, caches, threads) are not measured */
  for (int w = 0; w < warmup; w++)
    qgemm(A, B, C, M, N, K);

  for (int i = 0; i < niter; i++) {
    /* C += A * B, so C is reset before every run (outside the timer) */
    memset(C, 0, sizeof(int8_t) * (size_t)M * N);

    uint64_t start = timer();
    qgemm(A, B, C, M, N, K);
    uint64_t end = timer();

    double elapsed = (end - start) * 1e-9;
    double FLOPS = FLOP / elapsed;

    /* if range is not 0, don't print each results */
    if (range == 0) {
      printf("Exec. time = %.3lfms\n", elapsed * 1000);
      printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
    }

    res.exec_time[0] += (elapsed * 1000);
    res.exec_time[1] = (res.exec_time[1] < (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[1]);
    res.exec_time[2] = (res.exec_time[2] > (elapsed * 1000) ? (elapsed * 1000)
                                                            : res.exec_time[2]);

    res.gflops[0] += (FLOPS / 1e9);
    res.gflops[1] =
        (res.gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[1]);
    res.gflops[2] =
        (res.gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : res.gflops[2]);
  }

  /* The naive GEMM is slow, so it runs only on request and after the timed
     runs (before them it would leave the caches and the cores cold). */
  if (verify_flag) {
    int8_t* C_ref = alloc_mat(sizeof(int8_t) * (size_t)M * N);

    memset(C, 0, sizeof(int8_t) * (size_t)M * N);
    qgemm(A, B, C, M, N, K);
    naive_qgemm(A, B, C_ref, M, N, K);

    res.is_valid = TRUE;
    for (size_t idx = 0; idx < (size_t)M * N; idx++)
      if (C[idx] != C_ref[idx]) {
        res.is_valid = FALSE;
        break;
      }
    free(C_ref);
  }

  free(A);
  free(B);
  free(C);

  return report(&res, file, console_flag, csv_flag);
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

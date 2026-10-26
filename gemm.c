/**********************************************************************************************
 * File   : gemm.c
 * Author : kdh     
 * Github : https://github.com/kdhrepos/gemm.h
 * 
 * Description: 
 *      Definitions for main gemm functions. These functions have most important algorithms
 *      for one gemm operation. 
 * 
 *      Those are also optimized with hardware implemented in [opt.c] and they use 
 *      kernels in [kernel.c] and pack functions in [pack.c]
 *      
 *      The letter in front of gemm means a data type. For example, "s" in "sgemm" means
 *      32-bit floating point data type.
 * 
 * Reference:
 *      https://github.com/salykova/matmul.c
 *                                                    
**********************************************************************************************/

#include "gemm.h"

static void* buf_A = NULL;
static size_t cap_A = 0;
static void* buf_B = NULL;
static size_t cap_B = 0;

static void* gemm_buffer(void** buf, size_t* cap, size_t bytes) {
  if (*cap < bytes) {
    free(*buf);
    size_t sz =
        (bytes + MEM_ALIGN - 1) / MEM_ALIGN * MEM_ALIGN;  // 64B aligning
    *buf = aligned_alloc(MEM_ALIGN, sz);
    *cap = (*buf != NULL) ? sz : 0;
  }
  return *buf;
}

void sgemm(const float* A, const float* B, float* C, const int M, const int N,
           const int K) {
#if INSTLEVEL >= 8 /* AVX512F */
  int MR = 14, NR = 32;
#elif INSTLEVEL >= 6 /* AVX, AVX2 */
  int MR = 6, NR = 16;
#endif

  int MC, KC, NC;
  int NTHREADS = omp_get_max_threads();
  cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_FP32);

  /* packing for TLB efficiency */
  float* packed_A = gemm_buffer(&buf_A, &cap_A, sizeof(float) * MC * KC);
  float* packed_B = gemm_buffer(&buf_B, &cap_B, sizeof(float) * KC * NC);

  for (int Bm_col = 0; Bm_col < N; Bm_col += NC) { /* 5th loop */
    const int nc = min(NC, N - Bm_col);
    for (int k = 0; k < K; k += KC) { /* 4th loop */
      const int kc = min(KC, K - k);
      spack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, N, kc, NTHREADS);
      for (int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
        const int mc = min(MC, M - Am_row);
        spack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K,
                     NTHREADS);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
        for (int Bb_col = 0; Bb_col < nc; Bb_col += NR) {   /* 2nd loop */
          for (int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* 1st loop */
            const int nr = min(NR, nc - Bb_col);
            const int mr = min(MR, mc - Ab_row);
            skernel(&packed_A[Ab_row * kc], &packed_B[Bb_col * kc],
                    &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, MR, kc,
                    KC, nr, NR, N);
          }
        }
      }
    }
  }
}

void dgemm(const double* A, const double* B, double* C, const int M,
           const int N, const int K) {
#if INSTLEVEL >= 8 /* AVX512F */
  int MR = 6, NR = 16;
#elif INSTLEVEL >= 6 /* AVX, AVX2 */
  int MR = 6, NR = 8;
#endif

  int MC, KC, NC;
  int NTHREADS = omp_get_max_threads();
  cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_FP64);

  /* packing for TLB efficiency */
  double* packed_A = gemm_buffer(&buf_A, &cap_A, sizeof(double) * MC * KC);
  double* packed_B = gemm_buffer(&buf_B, &cap_B, sizeof(double) * KC * NC);

  for (int Bm_col = 0; Bm_col < N; Bm_col += NC) { /* 5th loop */
    const int nc = min(NC, N - Bm_col);
    for (int k = 0; k < K; k += KC) { /* 4th loop */
      const int kc = min(KC, K - k);
      dpack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, N, kc, NTHREADS);
      for (int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
        const int mc = min(MC, M - Am_row);
        dpack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K,
                     NTHREADS);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
        for (int Bb_col = 0; Bb_col < nc; Bb_col += NR) {   /* 2nd loop */
          for (int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* 1st loop */
            const int nr = min(NR, nc - Bb_col);
            const int mr = min(MR, mc - Ab_row);
            dkernel(&packed_A[Ab_row * kc], &packed_B[Bb_col * kc],
                    &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, MR, kc,
                    KC, nr, NR, N);
          }
        }
      }
    }
  }
}

void igemm(const int* A, const int* B, int* C, const int M, const int N,
           const int K) {
#if INSTLEVEL >= 8 /* AVX512F */
  int MR = 14, NR = 32;
#elif INSTLEVEL >= 6 /* AVX, AVX2 */
  int MR = 6, NR = 16;
#endif

  int MC, KC, NC;
  int NTHREADS = omp_get_max_threads();
  cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_INT32);

  /* packing for TLB efficiency */
  int* packed_A = gemm_buffer(&buf_A, &cap_A, sizeof(int) * MC * KC);
  int* packed_B = gemm_buffer(&buf_B, &cap_B, sizeof(int) * KC * NC);

  for (int Bm_col = 0; Bm_col < N; Bm_col += NC) { /* 5th loop */
    const int nc = min(NC, N - Bm_col);
    for (int k = 0; k < K; k += KC) { /* 4th loop */
      const int kc = min(KC, K - k);
      ipack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, N, kc, NTHREADS);
      for (int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
        const int mc = min(MC, M - Am_row);
        ipack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K,
                     NTHREADS);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
        for (int Bb_col = 0; Bb_col < nc; Bb_col += NR) {   /* 2nd loop */
          for (int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* 1st loop */
            const int nr = min(NR, nc - Bb_col);
            const int mr = min(MR, mc - Ab_row);
            ikernel(&packed_A[Ab_row * kc], &packed_B[Bb_col * kc],
                    &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, MR, kc,
                    KC, nr, NR, N);
          }
        }
      }
    }
  }
}

void hqgemm(const int16_t* A, const int16_t* B, int16_t* C, const int M,
            const int N, const int K) {
#if INSTLEVEL >= 9 /* AVX512BW */
  int MR = 30, NR = 32;
#elif INSTLEVEL >= 7 /* AVX2 */
  int MR = 6, NR = 16;
#elif INSTLEVEL >= 6
  int MR = 6, NR = 16;
#endif

  int MC, KC, NC;
  int NTHREADS = omp_get_max_threads();
  cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_INT16);

  /* packing for TLB efficiency */
  int16_t* packed_A = gemm_buffer(&buf_A, &cap_A, sizeof(int16_t) * MC * KC);
  int16_t* packed_B = gemm_buffer(&buf_B, &cap_B, sizeof(int16_t) * KC * NC);

  for (int Bm_col = 0; Bm_col < N; Bm_col += NC) { /* 5th loop */
    const int nc = min(NC, N - Bm_col);
    for (int k = 0; k < K; k += KC) { /* 4th loop */
      const int kc = min(KC, K - k);
      hqpack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, N, kc, NTHREADS);
      for (int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
        const int mc = min(MC, M - Am_row);
        hqpack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K,
                      NTHREADS);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
        for (int Bb_col = 0; Bb_col < nc; Bb_col += NR) {   /* 2nd loop */
          for (int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* 1st loop */
            const int nr = min(NR, nc - Bb_col);
            const int mr = min(MR, mc - Ab_row);
            hqkernel(&packed_A[Ab_row * kc], &packed_B[Bb_col * kc],
                     &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, MR,
                     kc, KC, nr, NR, N);
          }
        }
      }
    }
  }
}

void qgemm(const int8_t* A, const int8_t* B, int8_t* C, const int M,
           const int N, const int K) {
#if INSTLEVEL >= 9 /* AVX512BW */
  int MR = 30, NR = 64;
#elif INSTLEVEL >= 7 /* AVX2 */
  int MR = 6, NR = 16;
#elif INSTLEVEL >= 6
  int MR = 6, NR = 16;
#endif

  int MC, KC, NC;
  int NTHREADS = omp_get_max_threads();
  cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_INT8);

  /* packing for TLB efficiency */
  int8_t* packed_A = gemm_buffer(&buf_A, &cap_A, sizeof(int8_t) * MC * KC);
  int8_t* packed_B = gemm_buffer(&buf_B, &cap_B, sizeof(int8_t) * KC * NC);

  for (int Bm_col = 0; Bm_col < N; Bm_col += NC) { /* 5th loop */
    const int nc = min(NC, N - Bm_col);
    for (int k = 0; k < K; k += KC) { /* 4th loop */
      const int kc = min(KC, K - k);
      qpack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, N, kc, NTHREADS);
      for (int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
        const int mc = min(MC, M - Am_row);
        qpack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K,
                     NTHREADS);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
        for (int Bb_col = 0; Bb_col < nc; Bb_col += NR) {   /* 2nd loop */
          for (int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* 1st loop */
            const int nr = min(NR, nc - Bb_col);
            const int mr = min(MR, mc - Ab_row);
            qkernel(&packed_A[Ab_row * kc], &packed_B[Bb_col * kc],
                    &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, MR, kc,
                    KC, nr, NR, N);
          }
        }
      }
    }
  }
}

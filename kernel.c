/**********************************************************************************************
 * File   : kernel.c
 * Author : kdh     
 * Github : https://github.com/kdhrepos/gemm.h
 * 
 * Description: 
 *      Our gemm basically based on divide and conquer approach, which follows GotoBLAS.
 *      These kernels are the smallest ones that do matrix multiplication.
 * 
 *      They are all implemented with Intel SIMD intrisic, based on INSTLEVEL macro in
 *      [sse.h]
 *      
 *      The letter in front of kernel means a data type. For example, "s" in "skernel" means
 *      32-bit floating point data type.
 * Reference:
 *      https://github.com/salykova/matmul.c    
 *        
**********************************************************************************************/

#include "gemm.h"
#include "ops.h"

void skernel(const float* packed_blockA, const float* packed_blockB, float* C,
             const int m, const int MR, const int kc, const int KC, const int n,
             const int NR, const int N) {
#if INSTLEVEL >= 8 /* AVX512F */ /* 14x32 kernel */
  __m512 packed_C[14][2];        /* 14x32 */
  __m512 a_blockA, b0_blockB, b1_blockB;
  __mmask16 packed_mask_0 = (n < 16) ? 0xFFFF >> (16 - n) : 0xFFFF;
  __mmask16 packed_mask_1 = (n >= 16) ? 0xFFFF >> (32 - n) : 0x0000;

  for (int r = 0; r < m; r++) {
    packed_C[r][0] = _mm512_maskz_loadu_ps(packed_mask_0, &C[r * N + 0]);
    packed_C[r][1] = _mm512_maskz_loadu_ps(packed_mask_1, &C[r * N + 16]);
  }
  for (int k = 0; k < kc; k++) {
    b0_blockB = _mm512_load_ps(packed_blockB + 0);
    b1_blockB = _mm512_load_ps(packed_blockB + 16);

    a_blockA = _mm512_set1_ps(packed_blockA[0]);
    packed_C[0][0] = sfma(a_blockA, b0_blockB, packed_C[0][0]);
    packed_C[0][1] = sfma(a_blockA, b1_blockB, packed_C[0][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[1]);
    packed_C[1][0] = sfma(a_blockA, b0_blockB, packed_C[1][0]);
    packed_C[1][1] = sfma(a_blockA, b1_blockB, packed_C[1][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[2]);
    packed_C[2][0] = sfma(a_blockA, b0_blockB, packed_C[2][0]);
    packed_C[2][1] = sfma(a_blockA, b1_blockB, packed_C[2][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[3]);
    packed_C[3][0] = sfma(a_blockA, b0_blockB, packed_C[3][0]);
    packed_C[3][1] = sfma(a_blockA, b1_blockB, packed_C[3][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[4]);
    packed_C[4][0] = sfma(a_blockA, b0_blockB, packed_C[4][0]);
    packed_C[4][1] = sfma(a_blockA, b1_blockB, packed_C[4][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[5]);
    packed_C[5][0] = sfma(a_blockA, b0_blockB, packed_C[5][0]);
    packed_C[5][1] = sfma(a_blockA, b1_blockB, packed_C[5][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[6]);
    packed_C[6][0] = sfma(a_blockA, b0_blockB, packed_C[6][0]);
    packed_C[6][1] = sfma(a_blockA, b1_blockB, packed_C[6][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[7]);
    packed_C[7][0] = sfma(a_blockA, b0_blockB, packed_C[7][0]);
    packed_C[7][1] = sfma(a_blockA, b1_blockB, packed_C[7][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[8]);
    packed_C[8][0] = sfma(a_blockA, b0_blockB, packed_C[8][0]);
    packed_C[8][1] = sfma(a_blockA, b1_blockB, packed_C[8][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[9]);
    packed_C[9][0] = sfma(a_blockA, b0_blockB, packed_C[9][0]);
    packed_C[9][1] = sfma(a_blockA, b1_blockB, packed_C[9][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[10]);
    packed_C[10][0] = sfma(a_blockA, b0_blockB, packed_C[10][0]);
    packed_C[10][1] = sfma(a_blockA, b1_blockB, packed_C[10][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[11]);
    packed_C[11][0] = sfma(a_blockA, b0_blockB, packed_C[11][0]);
    packed_C[11][1] = sfma(a_blockA, b1_blockB, packed_C[11][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[12]);
    packed_C[12][0] = sfma(a_blockA, b0_blockB, packed_C[12][0]);
    packed_C[12][1] = sfma(a_blockA, b1_blockB, packed_C[12][1]);

    a_blockA = _mm512_set1_ps(packed_blockA[13]);
    packed_C[13][0] = sfma(a_blockA, b0_blockB, packed_C[13][0]);
    packed_C[13][1] = sfma(a_blockA, b1_blockB, packed_C[13][1]);

    packed_blockA += MR; /* next column */
    packed_blockB += NR; /* next 32 elements*/
  }
  for (int r = 0; r < m; r++) {
    _mm512_mask_storeu_ps(&C[r * N + 0], packed_mask_0, packed_C[r][0]);
    _mm512_mask_storeu_ps(&C[r * N + 16], packed_mask_1, packed_C[r][1]);
  }
#elif INSTLEVEL >= 6 /* AVX, AVX2 */ /* 6x16 kernel */
  __m256 packed_C[6][2]; /* 6x16 */
  __m256 a_blockA, b0_blockB, b1_blockB;
  __m256i packed_mask[2];

  static int32_t mask[32] __attribute__((aligned(32))) = {
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

  packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 0]);
  packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 8]);

  for (int r = 0; r < m; r++) {
    packed_C[r][0] = _mm256_maskload_ps(&C[r * N + 0], packed_mask[0]);
    packed_C[r][1] = _mm256_maskload_ps(&C[r * N + 8], packed_mask[1]);
  }
  for (int k = 0; k < kc; k++) {
    b0_blockB = _mm256_loadu_ps(packed_blockB + 0);
    b1_blockB = _mm256_loadu_ps(packed_blockB + 8);

    a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 0));
    packed_C[0][0] = sfma(a_blockA, b0_blockB, packed_C[0][0]);
    packed_C[0][1] = sfma(a_blockA, b1_blockB, packed_C[0][1]);

    a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 1));
    packed_C[1][0] = sfma(a_blockA, b0_blockB, packed_C[1][0]);
    packed_C[1][1] = sfma(a_blockA, b1_blockB, packed_C[1][1]);

    a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 2));
    packed_C[2][0] = sfma(a_blockA, b0_blockB, packed_C[2][0]);
    packed_C[2][1] = sfma(a_blockA, b1_blockB, packed_C[2][1]);

    a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 3));
    packed_C[3][0] = sfma(a_blockA, b0_blockB, packed_C[3][0]);
    packed_C[3][1] = sfma(a_blockA, b1_blockB, packed_C[3][1]);

    a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 4));
    packed_C[4][0] = sfma(a_blockA, b0_blockB, packed_C[4][0]);
    packed_C[4][1] = sfma(a_blockA, b1_blockB, packed_C[4][1]);

    a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 5));
    packed_C[5][0] = sfma(a_blockA, b0_blockB, packed_C[5][0]);
    packed_C[5][1] = sfma(a_blockA, b1_blockB, packed_C[5][1]);

    packed_blockA += 1;  /* next column */
    packed_blockB += NC; /* next 16 elements*/
  }
  for (int r = 0; r < m; r++) {
    _mm256_maskstore_ps(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
    _mm256_maskstore_ps(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
  }
#endif                               // skernel
}

void dkernel(const double* packed_blockA, const double* packed_blockB,
             double* C, const int m, const int kc, const int KC, const int n,
             const int NC, const int N) {
#if INSTLEVEL >= 8 /* AVX512F */ /* 6x16 kernel */
  __m512d packed_C[6][4];        /* 6x16 */
  __m512d a_blockA, b0_blockB, b1_blockB;
  __mmask16 packed_mask_0 = (n < 8) ? 0xFFFF >> (8 - n) : 0xFFFF;
  __mmask16 packed_mask_1 = (n >= 8) ? 0xFFFF >> (16 - n) : 0x0000;

  for (int r = 0; r < m; r++) {
    packed_C[r][0] = _mm512_maskz_loadu_pd(packed_mask_0, &C[r * N + 0]);
    packed_C[r][1] = _mm512_maskz_loadu_pd(packed_mask_1, &C[r * N + 8]);
  }
  for (int k = 0; k < kc; k++) {
    b0_blockB = _mm512_load_pd(packed_blockB + 0);
    b1_blockB = _mm512_load_pd(packed_blockB + 8);

    a_blockA = _mm512_set1_pd(packed_blockA[KC * 0]);
    packed_C[0][0] = dfma(a_blockA, b0_blockB, packed_C[0][0]);
    packed_C[0][1] = dfma(a_blockA, b1_blockB, packed_C[0][1]);

    a_blockA = _mm512_set1_pd(packed_blockA[KC * 1]);
    packed_C[1][0] = dfma(a_blockA, b0_blockB, packed_C[1][0]);
    packed_C[1][1] = dfma(a_blockA, b1_blockB, packed_C[1][1]);

    a_blockA = _mm512_set1_pd(packed_blockA[KC * 2]);
    packed_C[2][0] = dfma(a_blockA, b0_blockB, packed_C[2][0]);
    packed_C[2][1] = dfma(a_blockA, b1_blockB, packed_C[2][1]);

    a_blockA = _mm512_set1_pd(packed_blockA[KC * 3]);
    packed_C[3][0] = dfma(a_blockA, b0_blockB, packed_C[3][0]);
    packed_C[3][1] = dfma(a_blockA, b1_blockB, packed_C[3][1]);

    a_blockA = _mm512_set1_pd(packed_blockA[KC * 4]);
    packed_C[4][0] = dfma(a_blockA, b0_blockB, packed_C[4][0]);
    packed_C[4][1] = dfma(a_blockA, b1_blockB, packed_C[4][1]);

    a_blockA = _mm512_set1_pd(packed_blockA[KC * 5]);
    packed_C[5][0] = dfma(a_blockA, b0_blockB, packed_C[5][0]);
    packed_C[5][1] = dfma(a_blockA, b1_blockB, packed_C[5][1]);

    packed_blockA += 1;  /* next column */
    packed_blockB += NC; /* next 16 elements*/
  }
  for (int r = 0; r < m; r++) {
    _mm512_mask_storeu_pd(&C[r * N + 0], packed_mask_0, packed_C[r][0]);
    _mm512_mask_storeu_pd(&C[r * N + 8], packed_mask_1, packed_C[r][1]);
  }
#elif INSTLEVEL >= 6 /* AVX, AVX2 */ /* 6x8 kernel */
  __m256d packed_C[6][2]; /* 6x8 */
  __m256d a_blockA, b0_blockB, b1_blockB;
  __m256i packed_mask[2];

  static int64_t mask[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                             0,  0,  0,  0,  0,  0,  0,  0};

  packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 0]);
  packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 4]);
  for (int r = 0; r < m; r++) {
    packed_C[r][0] = _mm256_maskload_pd(&C[r * N + 0], packed_mask[0]);
    packed_C[r][1] = _mm256_maskload_pd(&C[r * N + 4], packed_mask[1]);
  }
  for (int k = 0; k < kc; k++) {
    b0_blockB = _mm256_loadu_pd(packed_blockB + 0);
    b1_blockB = _mm256_loadu_pd(packed_blockB + 4);

    a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 0));
    packed_C[0][0] = dfma(a_blockA, b0_blockB, packed_C[0][0]);
    packed_C[0][1] = dfma(a_blockA, b1_blockB, packed_C[0][1]);

    a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 1));
    packed_C[1][0] = dfma(a_blockA, b0_blockB, packed_C[1][0]);
    packed_C[1][1] = dfma(a_blockA, b1_blockB, packed_C[1][1]);

    a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 2));
    packed_C[2][0] = dfma(a_blockA, b0_blockB, packed_C[2][0]);
    packed_C[2][1] = dfma(a_blockA, b1_blockB, packed_C[2][1]);

    a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 3));
    packed_C[3][0] = dfma(a_blockA, b0_blockB, packed_C[3][0]);
    packed_C[3][1] = dfma(a_blockA, b1_blockB, packed_C[3][1]);

    a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 4));
    packed_C[4][0] = dfma(a_blockA, b0_blockB, packed_C[4][0]);
    packed_C[4][1] = dfma(a_blockA, b1_blockB, packed_C[4][1]);

    a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 5));
    packed_C[5][0] = dfma(a_blockA, b0_blockB, packed_C[5][0]);
    packed_C[5][1] = dfma(a_blockA, b1_blockB, packed_C[5][1]);

    packed_blockA += 1;  /* next column */
    packed_blockB += NC; /* next 16 elements*/
  }
  for (int r = 0; r < m; r++) {
    _mm256_maskstore_pd(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
    _mm256_maskstore_pd(&C[r * N + 4], packed_mask[1], packed_C[r][1]);
  }
#endif                               // dkernel
}

void ikernel(const int* packed_blockA, const int* packed_blockB, int* C,
             const int m, const int kc, const int KC, const int n, const int NC,
             const int N) {
#if INSTLEVEL >= 8 /* AVX512F */ /* 14x32 kernel */
  __m512i packed_C[14][2];       /* 14x32 */
  __m512i a_blockA, b0_blockB, b1_blockB;
  __mmask16 packed_mask_0 = (n < 16) ? 0xFFFF >> (16 - n) : 0xFFFF;
  __mmask16 packed_mask_1 = (n >= 16) ? 0xFFFF >> (32 - n) : 0x0000;

  for (int r = 0; r < m; r++) {
    packed_C[r][0] = _mm512_maskz_loadu_epi32(packed_mask_0, &C[r * N + 0]);
    packed_C[r][1] = _mm512_maskz_loadu_epi32(packed_mask_1, &C[r * N + 16]);
  }
  for (int k = 0; k < kc; k++) {
    b0_blockB = _mm512_load_epi32(packed_blockB + 0);
    b1_blockB = _mm512_load_epi32(packed_blockB + 16);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 0]);
    packed_C[0][0] = ifma(a_blockA, b0_blockB, packed_C[0][0]);
    packed_C[0][1] = ifma(a_blockA, b1_blockB, packed_C[0][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 1]);
    packed_C[1][0] = ifma(a_blockA, b0_blockB, packed_C[1][0]);
    packed_C[1][1] = ifma(a_blockA, b1_blockB, packed_C[1][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 2]);
    packed_C[2][0] = ifma(a_blockA, b0_blockB, packed_C[2][0]);
    packed_C[2][1] = ifma(a_blockA, b1_blockB, packed_C[2][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 3]);
    packed_C[3][0] = ifma(a_blockA, b0_blockB, packed_C[3][0]);
    packed_C[3][1] = ifma(a_blockA, b1_blockB, packed_C[3][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 4]);
    packed_C[4][0] = ifma(a_blockA, b0_blockB, packed_C[4][0]);
    packed_C[4][1] = ifma(a_blockA, b1_blockB, packed_C[4][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 5]);
    packed_C[5][0] = ifma(a_blockA, b0_blockB, packed_C[5][0]);
    packed_C[5][1] = ifma(a_blockA, b1_blockB, packed_C[5][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 6]);
    packed_C[6][0] = ifma(a_blockA, b0_blockB, packed_C[6][0]);
    packed_C[6][1] = ifma(a_blockA, b1_blockB, packed_C[6][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 7]);
    packed_C[7][0] = ifma(a_blockA, b0_blockB, packed_C[7][0]);
    packed_C[7][1] = ifma(a_blockA, b1_blockB, packed_C[7][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 8]);
    packed_C[8][0] = ifma(a_blockA, b0_blockB, packed_C[8][0]);
    packed_C[8][1] = ifma(a_blockA, b1_blockB, packed_C[8][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 9]);
    packed_C[9][0] = ifma(a_blockA, b0_blockB, packed_C[9][0]);
    packed_C[9][1] = ifma(a_blockA, b1_blockB, packed_C[9][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 10]);
    packed_C[10][0] = ifma(a_blockA, b0_blockB, packed_C[10][0]);
    packed_C[10][1] = ifma(a_blockA, b1_blockB, packed_C[10][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 11]);
    packed_C[11][0] = ifma(a_blockA, b0_blockB, packed_C[11][0]);
    packed_C[11][1] = ifma(a_blockA, b1_blockB, packed_C[11][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 12]);
    packed_C[12][0] = ifma(a_blockA, b0_blockB, packed_C[12][0]);
    packed_C[12][1] = ifma(a_blockA, b1_blockB, packed_C[12][1]);

    a_blockA = _mm512_set1_epi32(packed_blockA[KC * 13]);
    packed_C[13][0] = ifma(a_blockA, b0_blockB, packed_C[13][0]);
    packed_C[13][1] = ifma(a_blockA, b1_blockB, packed_C[13][1]);

    packed_blockA += 1;  /* next column */
    packed_blockB += NC; /* next 32 elements*/
  }
  for (int r = 0; r < m; r++) {
    _mm512_mask_storeu_epi32(&C[r * N + 0], packed_mask_0, packed_C[r][0]);
    _mm512_mask_storeu_epi32(&C[r * N + 16], packed_mask_1, packed_C[r][1]);
  }
#elif INSTLEVEL >= 7 /* AVX2 */ /* 6x16 kernel */
  __m256i packed_C[6][2]; /* 6x16 */
  __m256i a_blockA, b0_blockB, b1_blockB;
  __m256i packed_mask[2];

  static int32_t mask[32] __attribute__((aligned(32))) = {
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

  packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 0]);
  packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 8]);

  for (int r = 0; r < m; r++) {
    packed_C[r][0] = _mm256_maskload_epi32(&C[r * N + 0], packed_mask[0]);
    packed_C[r][1] = _mm256_maskload_epi32(&C[r * N + 8], packed_mask[1]);
  }
  for (int k = 0; k < kc; k++) {
    b0_blockB = _mm256_loadu_si256((__m256i_u*)(packed_blockB + 0));
    b1_blockB = _mm256_loadu_si256((__m256i_u*)(packed_blockB + 8));

    a_blockA = _mm256_set1_epi32(packed_blockA[KC * 0]);
    packed_C[0][0] = ifma(a_blockA, b0_blockB, packed_C[0][0]);
    packed_C[0][1] = ifma(a_blockA, b1_blockB, packed_C[0][1]);

    a_blockA = _mm256_set1_epi32(packed_blockA[KC * 1]);
    packed_C[1][0] = ifma(a_blockA, b0_blockB, packed_C[1][0]);
    packed_C[1][1] = ifma(a_blockA, b1_blockB, packed_C[1][1]);

    a_blockA = _mm256_set1_epi32(packed_blockA[KC * 2]);
    packed_C[2][0] = ifma(a_blockA, b0_blockB, packed_C[2][0]);
    packed_C[2][1] = ifma(a_blockA, b1_blockB, packed_C[2][1]);

    a_blockA = _mm256_set1_epi32(packed_blockA[KC * 3]);
    packed_C[3][0] = ifma(a_blockA, b0_blockB, packed_C[3][0]);
    packed_C[3][1] = ifma(a_blockA, b1_blockB, packed_C[3][1]);

    a_blockA = _mm256_set1_epi32(packed_blockA[KC * 4]);
    packed_C[4][0] = ifma(a_blockA, b0_blockB, packed_C[4][0]);
    packed_C[4][1] = ifma(a_blockA, b1_blockB, packed_C[4][1]);

    a_blockA = _mm256_set1_epi32(packed_blockA[KC * 5]);
    packed_C[5][0] = ifma(a_blockA, b0_blockB, packed_C[5][0]);
    packed_C[5][1] = ifma(a_blockA, b1_blockB, packed_C[5][1]);

    packed_blockA += 1;  /* next column */
    packed_blockB += NC; /* next 16 elements*/
  }
  for (int r = 0; r < m; r++) {
    _mm256_maskstore_epi32(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
    _mm256_maskstore_epi32(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
  }
#elif INSTLEVEL >= 6            /* AVX */
  __m128i packed_C[6][4]; /* 6x16 */
  __m128i a_blockA;
  __m128i b0_blockB, b1_blockB, b2_blockB, b3_blockB;
  int8_t mask0, mask1, mask2, mask3;
  mask0 = (n < 4) ? 0xF >> (4 - n) : 0xF;
  mask1 = (n >= 4) ? (n >= 8) ? 0xF : 0xF >> (8 - n) : 0x0;
  mask2 = (n >= 8) ? (n >= 12) ? 0xF : 0xF >> (12 - n) : 0x0;
  mask3 = (n >= 12) ? 0xF >> (16 - n) : 0x0;

  for (int r = 0; r < m; r++) {
    packed_C[r][0] = maskload(&C[r * N + 0], mask0);
    packed_C[r][1] = maskload(&C[r * N + 4], mask1);
    packed_C[r][2] = maskload(&C[r * N + 8], mask2);
    packed_C[r][3] = maskload(&C[r * N + 12], mask3);
  }
  for (int k = 0; k < kc; k++) {
    b0_blockB = _mm_load_si128((__m128i_u*)(packed_blockB + 0));
    b1_blockB = _mm_load_si128((__m128i_u*)(packed_blockB + 4));
    b2_blockB = _mm_load_si128((__m128i_u*)(packed_blockB + 8));
    b3_blockB = _mm_load_si128((__m128i_u*)(packed_blockB + 12));

    a_blockA = _mm_set1_epi32(packed_blockA[KC * 0]);
    packed_C[0][0] = ifma(a_blockA, b0_blockB, packed_C[0][0]);
    packed_C[0][1] = ifma(a_blockA, b1_blockB, packed_C[0][1]);
    packed_C[0][2] = ifma(a_blockA, b2_blockB, packed_C[0][2]);
    packed_C[0][3] = ifma(a_blockA, b3_blockB, packed_C[0][3]);

    a_blockA = _mm_set1_epi32(packed_blockA[KC * 1]);
    packed_C[1][0] = ifma(a_blockA, b0_blockB, packed_C[1][0]);
    packed_C[1][1] = ifma(a_blockA, b1_blockB, packed_C[1][1]);
    packed_C[1][2] = ifma(a_blockA, b2_blockB, packed_C[1][2]);
    packed_C[1][3] = ifma(a_blockA, b3_blockB, packed_C[1][3]);

    a_blockA = _mm_set1_epi32(packed_blockA[KC * 2]);
    packed_C[2][0] = ifma(a_blockA, b0_blockB, packed_C[2][0]);
    packed_C[2][1] = ifma(a_blockA, b1_blockB, packed_C[2][1]);
    packed_C[2][2] = ifma(a_blockA, b2_blockB, packed_C[2][2]);
    packed_C[2][3] = ifma(a_blockA, b3_blockB, packed_C[2][3]);

    a_blockA = _mm_set1_epi32(packed_blockA[KC * 3]);
    packed_C[3][0] = ifma(a_blockA, b0_blockB, packed_C[3][0]);
    packed_C[3][1] = ifma(a_blockA, b1_blockB, packed_C[3][1]);
    packed_C[3][2] = ifma(a_blockA, b2_blockB, packed_C[3][2]);
    packed_C[3][3] = ifma(a_blockA, b3_blockB, packed_C[3][3]);

    a_blockA = _mm_set1_epi32(packed_blockA[KC * 4]);
    packed_C[4][0] = ifma(a_blockA, b0_blockB, packed_C[4][0]);
    packed_C[4][1] = ifma(a_blockA, b1_blockB, packed_C[4][1]);
    packed_C[4][2] = ifma(a_blockA, b2_blockB, packed_C[4][2]);
    packed_C[4][3] = ifma(a_blockA, b3_blockB, packed_C[4][3]);

    a_blockA = _mm_set1_epi32(packed_blockA[KC * 5]);
    packed_C[5][0] = ifma(a_blockA, b0_blockB, packed_C[5][0]);
    packed_C[5][1] = ifma(a_blockA, b1_blockB, packed_C[5][1]);
    packed_C[5][2] = ifma(a_blockA, b2_blockB, packed_C[5][2]);
    packed_C[5][3] = ifma(a_blockA, b3_blockB, packed_C[5][3]);

    packed_blockA += 1;  /* next column */
    packed_blockB += NC; /* next 16 elements*/
  }
  for (int r = 0; r < m; r++) {
    maskstore(&C[r * N + 0], mask0, packed_C[r][0]);
    maskstore(&C[r * N + 4], mask1, packed_C[r][1]);
    maskstore(&C[r * N + 8], mask2, packed_C[r][2]);
    maskstore(&C[r * N + 12], mask3, packed_C[r][3]);
  }
#endif                          // ikernel
}

void hqkernel(const int16_t* packed_blockA, const int16_t* packed_blockB,
              int16_t* C, const int m, const int kc, const int KC, const int n,
              const int NC, const int N) {
#if INSTLEVEL >= 9      /* AVX512BW */
  __m512i packed_C[30]; /* 30x32 */
  __m512i a_blockA, b_blockB;
  __mmask32 packed_mask = 0xFFFFFFFF >> (32 - n);

  for (int r = 0; r < m; r++)
    packed_C[r] = _mm512_maskz_loadu_epi16(packed_mask, &C[r * N]);
  for (int k = 0; k < kc; k++) {
    b_blockB = _mm512_loadu_epi16(packed_blockB);

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 0]);
    packed_C[0] =
        _mm512_add_epi16(packed_C[0], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 1]);
    packed_C[1] =
        _mm512_add_epi16(packed_C[1], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 2]);
    packed_C[2] =
        _mm512_add_epi16(packed_C[2], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 3]);
    packed_C[3] =
        _mm512_add_epi16(packed_C[3], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 4]);
    packed_C[4] =
        _mm512_add_epi16(packed_C[4], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 5]);
    packed_C[5] =
        _mm512_add_epi16(packed_C[5], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 6]);
    packed_C[6] =
        _mm512_add_epi16(packed_C[6], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 7]);
    packed_C[7] =
        _mm512_add_epi16(packed_C[7], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 8]);
    packed_C[8] =
        _mm512_add_epi16(packed_C[8], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 9]);
    packed_C[9] =
        _mm512_add_epi16(packed_C[9], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 10]);
    packed_C[10] =
        _mm512_add_epi16(packed_C[10], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 11]);
    packed_C[11] =
        _mm512_add_epi16(packed_C[11], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 12]);
    packed_C[12] =
        _mm512_add_epi16(packed_C[12], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 13]);
    packed_C[13] =
        _mm512_add_epi16(packed_C[13], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 14]);
    packed_C[14] =
        _mm512_add_epi16(packed_C[14], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 15]);
    packed_C[15] =
        _mm512_add_epi16(packed_C[15], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 16]);
    packed_C[16] =
        _mm512_add_epi16(packed_C[16], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 17]);
    packed_C[17] =
        _mm512_add_epi16(packed_C[17], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 18]);
    packed_C[18] =
        _mm512_add_epi16(packed_C[18], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 19]);
    packed_C[19] =
        _mm512_add_epi16(packed_C[19], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 20]);
    packed_C[20] =
        _mm512_add_epi16(packed_C[20], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 21]);
    packed_C[21] =
        _mm512_add_epi16(packed_C[21], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 22]);
    packed_C[22] =
        _mm512_add_epi16(packed_C[22], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 23]);
    packed_C[23] =
        _mm512_add_epi16(packed_C[23], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 24]);
    packed_C[24] =
        _mm512_add_epi16(packed_C[24], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 25]);
    packed_C[25] =
        _mm512_add_epi16(packed_C[25], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 26]);
    packed_C[26] =
        _mm512_add_epi16(packed_C[26], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 27]);
    packed_C[27] =
        _mm512_add_epi16(packed_C[27], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 28]);
    packed_C[28] =
        _mm512_add_epi16(packed_C[28], _mm512_mullo_epi16(b_blockB, a_blockA));

    a_blockA = _mm512_set1_epi16(packed_blockA[KC * 29]);
    packed_C[29] =
        _mm512_add_epi16(packed_C[29], _mm512_mullo_epi16(b_blockB, a_blockA));

    packed_blockA += 1;  /* next column */
    packed_blockB += NC; /* next 32 elements*/
  }
  for (int r = 0; r < m; r++)
    _mm512_mask_storeu_epi16(&C[r * N], packed_mask, packed_C[r]);
#elif INSTLEVEL >= 7 /* AVX2 */
// TODO: implement
#endif               // hqkernel
}

void qkernel(const int8_t* packed_blockA, const int8_t* packed_blockB,
             int8_t* C, const int m, const int kc, const int KC, const int n,
             const int NC, const int N) {
#if INSTLEVEL >= 9      /* AVX512BW */
  __m512i packed_C[30]; /* 30x64 */
  __m512i a_blockA, b_blockB;
  __mmask64 packed_mask = 0xFFFFFFFFFFFFFFFF >> (64 - n);

  for (int r = 0; r < m; r++)
    packed_C[r] = _mm512_maskz_loadu_epi8(packed_mask, &C[r * N]);
  for (int k = 0; k < kc; k++) {
    b_blockB = _mm512_loadu_epi8(packed_blockB);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 0]);
    packed_C[0] = qfma(a_blockA, b_blockB, packed_C[0]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 1]);
    packed_C[1] = qfma(a_blockA, b_blockB, packed_C[1]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 2]);
    packed_C[2] = qfma(a_blockA, b_blockB, packed_C[2]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 3]);
    packed_C[3] = qfma(a_blockA, b_blockB, packed_C[3]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 4]);
    packed_C[4] = qfma(a_blockA, b_blockB, packed_C[4]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 5]);
    packed_C[5] = qfma(a_blockA, b_blockB, packed_C[5]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 6]);
    packed_C[6] = qfma(a_blockA, b_blockB, packed_C[6]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 7]);
    packed_C[7] = qfma(a_blockA, b_blockB, packed_C[7]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 8]);
    packed_C[8] = qfma(a_blockA, b_blockB, packed_C[8]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 9]);
    packed_C[9] = qfma(a_blockA, b_blockB, packed_C[9]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 10]);
    packed_C[10] = qfma(a_blockA, b_blockB, packed_C[10]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 11]);
    packed_C[11] = qfma(a_blockA, b_blockB, packed_C[11]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 12]);
    packed_C[12] = qfma(a_blockA, b_blockB, packed_C[12]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 13]);
    packed_C[13] = qfma(a_blockA, b_blockB, packed_C[13]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 14]);
    packed_C[14] = qfma(a_blockA, b_blockB, packed_C[14]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 15]);
    packed_C[15] = qfma(a_blockA, b_blockB, packed_C[15]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 16]);
    packed_C[16] = qfma(a_blockA, b_blockB, packed_C[16]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 17]);
    packed_C[17] = qfma(a_blockA, b_blockB, packed_C[17]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 18]);
    packed_C[18] = qfma(a_blockA, b_blockB, packed_C[18]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 19]);
    packed_C[19] = qfma(a_blockA, b_blockB, packed_C[19]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 20]);
    packed_C[20] = qfma(a_blockA, b_blockB, packed_C[20]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 21]);
    packed_C[21] = qfma(a_blockA, b_blockB, packed_C[21]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 22]);
    packed_C[22] = qfma(a_blockA, b_blockB, packed_C[22]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 23]);
    packed_C[23] = qfma(a_blockA, b_blockB, packed_C[23]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 24]);
    packed_C[24] = qfma(a_blockA, b_blockB, packed_C[24]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 25]);
    packed_C[25] = qfma(a_blockA, b_blockB, packed_C[25]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 26]);
    packed_C[26] = qfma(a_blockA, b_blockB, packed_C[26]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 27]);
    packed_C[27] = qfma(a_blockA, b_blockB, packed_C[27]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 28]);
    packed_C[28] = qfma(a_blockA, b_blockB, packed_C[28]);

    a_blockA = _mm512_set1_epi8(packed_blockA[KC * 29]);
    packed_C[29] = qfma(a_blockA, b_blockB, packed_C[29]);

    packed_blockA += 1;  /* next column */
    packed_blockB += NC; /* next 32 elements*/
  }
  for (int r = 0; r < m; r++)
    _mm512_mask_storeu_epi8(&C[r * N], packed_mask, packed_C[r]);
#elif INSTLEVEL >= 7 /* AVX2 */
// TODO: implement
#endif               // qkernel
}
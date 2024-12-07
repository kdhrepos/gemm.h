#include "gemm.h"

/**
 * Micro kernel for GEMM, implemented with Intel SSE intrinsic
 * 
 * INSTLEVEL >= 8 for AVX512F
 *  14x32 kernel vs 31x16 kernel
 *  32 ZMM registers
 *  Use FMA
 * INSTLEVEL >= 7 for AVX2
 *  6x16 kernel
 *  16 YMM registers
 *  Use FMA
 * INSTLEVEL >= 6 for AVX
 *  6x16 kernel
 *  16 YMM registers
 *  No FMA
 */
void s_kernel(const float* packed_blockA, const float* packed_blockB, float* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 8 /* AVX512F */ /* 14x32 kernel */
    __m512 packed_C[14][2]; /* 14x32 */
    __m512 a_blockA, b0_blockB, b1_blockB;
    __mmask16 packed_mask_0 = (n < 16)  ? 0xFFFF >> (16 - n) : 0xFFFF;
    __mmask16 packed_mask_1 = (n >= 16) ? 0xFFFF >> (32 - n) : 0x0000;

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm512_maskz_loadu_ps(packed_mask_0, &C[r * N + 0]);
        packed_C[r][1] = _mm512_maskz_loadu_ps(packed_mask_1, &C[r * N + 16]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm512_load_ps(packed_blockB + 0);
        b1_blockB = _mm512_load_ps(packed_blockB + 16);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 0]); 
        packed_C[0][0] = sfma(a_blockA, b0_blockB, packed_C[0][0]);
        packed_C[0][1] = sfma(a_blockA, b1_blockB, packed_C[0][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 1]); 
        packed_C[1][0] = sfma(a_blockA, b0_blockB, packed_C[1][0]);
        packed_C[1][1] = sfma(a_blockA, b1_blockB, packed_C[1][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 2]); 
        packed_C[2][0] = sfma(a_blockA, b0_blockB, packed_C[2][0]);
        packed_C[2][1] = sfma(a_blockA, b1_blockB, packed_C[2][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 3]); 
        packed_C[3][0] = sfma(a_blockA, b0_blockB, packed_C[3][0]);
        packed_C[3][1] = sfma(a_blockA, b1_blockB, packed_C[3][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 4]); 
        packed_C[4][0] = sfma(a_blockA, b0_blockB, packed_C[4][0]);
        packed_C[4][1] = sfma(a_blockA, b1_blockB, packed_C[4][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 5]); 
        packed_C[5][0] = sfma(a_blockA, b0_blockB, packed_C[5][0]);
        packed_C[5][1] = sfma(a_blockA, b1_blockB, packed_C[5][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 6]); 
        packed_C[6][0] = sfma(a_blockA, b0_blockB, packed_C[6][0]);
        packed_C[6][1] = sfma(a_blockA, b1_blockB, packed_C[6][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 7]); 
        packed_C[7][0] = sfma(a_blockA, b0_blockB, packed_C[7][0]);
        packed_C[7][1] = sfma(a_blockA, b1_blockB, packed_C[7][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 8]); 
        packed_C[8][0] = sfma(a_blockA, b0_blockB, packed_C[8][0]);
        packed_C[8][1] = sfma(a_blockA, b1_blockB, packed_C[8][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 9]); 
        packed_C[9][0] = sfma(a_blockA, b0_blockB, packed_C[9][0]);
        packed_C[9][1] = sfma(a_blockA, b1_blockB, packed_C[9][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 10]); 
        packed_C[10][0] = sfma(a_blockA, b0_blockB, packed_C[10][0]);
        packed_C[10][1] = sfma(a_blockA, b1_blockB, packed_C[10][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 11]); 
        packed_C[11][0] = sfma(a_blockA, b0_blockB, packed_C[11][0]);
        packed_C[11][1] = sfma(a_blockA, b1_blockB, packed_C[11][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 12]); 
        packed_C[12][0] = sfma(a_blockA, b0_blockB, packed_C[12][0]);
        packed_C[12][1] = sfma(a_blockA, b1_blockB, packed_C[12][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 13]); 
        packed_C[13][0] = sfma(a_blockA, b0_blockB, packed_C[13][0]);
        packed_C[13][1] = sfma(a_blockA, b1_blockB, packed_C[13][1]);

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 32 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm512_mask_storeu_ps(&C[r * N + 0], packed_mask_0, packed_C[r][0]);
        _mm512_mask_storeu_ps(&C[r * N + 16], packed_mask_1, packed_C[r][1]);
    }
#elif INSTLEVEL >= 7 /* AVX2 */ /* 6x16 kernel */
    __m256 packed_C[6][2]; /* 6x16 */
    __m256 b0_blockB, b1_blockB, a_blockA;
    __m256i packed_mask[2];

    static int32_t mask[32] __attribute__((aligned(32))) = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    };
    
    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 8]);
    
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_ps(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_ps(&C[r * N + 8], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
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

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_ps(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_ps(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
    }
#elif INSTLEVEL >= 6 /* AVX */ /* 6x16 kernel */
    __m256 packed_C[6][2]; /* 6x16 */
    __m256 b0_blockB, b1_blockB, a_blockA;
    __m256i packed_mask[2];

    static int32_t mask[32] __attribute__((aligned(32))) = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    };

    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 8]);
    
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_ps(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_ps(&C[r * N + 8], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
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

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_ps(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_ps(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
    }
#endif
}

void d_kernel(const double* packed_blockA, const double* packed_blockB, double* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 8 /* AVX512F */ /* 6x16 kernel */
    __m512d packed_C[6][4]; /* 6x16 */
    __m512d a_blockA, b0_blockB, b1_blockB;
    __mmask16 packed_mask_0 = (n < 8)  ? 0xFFFF >> (8 - n)  : 0xFFFF;
    __mmask16 packed_mask_1 = (n >= 8) ? 0xFFFF >> (16 - n) : 0x0000;

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm512_maskz_loadu_pd(packed_mask_0, &C[r * N + 0]);
        packed_C[r][1] = _mm512_maskz_loadu_pd(packed_mask_1, &C[r * N + 8]);
    }
    for(int k = 0; k < kc; k++) {
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
    for(int r = 0; r < m; r++) {
        _mm512_mask_storeu_pd(&C[r * N + 0], packed_mask_0, packed_C[r][0]);
        _mm512_mask_storeu_pd(&C[r * N + 8], packed_mask_1, packed_C[r][1]);
    }
#elif INSTLEVEL >= 7 /* AVX2 */ /* 6x8 kernel */
    __m256d packed_C[6][2]; /* 6x8 */
    __m256d a_blockA, b0_blockB, b1_blockB;
    __m256i packed_mask[2];

    static int64_t mask[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 
        0,  0,  0,  0,  0,  0,  0,  0
    };

    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 4]);

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_pd(&C[r * N + 0],  packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_pd(&C[r * N + 4],  packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_load_pd(packed_blockB + 0);
        b1_blockB = _mm256_load_pd(packed_blockB + 4);

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
        packed_blockB += NC; /* next 8 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_pd(&C[r * N + 0],  packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_pd(&C[r * N + 4],  packed_mask[1], packed_C[r][1]);
    }
#elif INSTLEVEL >= 6 /* AVX */ /* 6x8 kernel */
    __m256d packed_C[6][2]; /* 6x8 */
    __m256d a_blockA, b0_blockB, b1_blockB;
    __m256i packed_mask[2];

    static int64_t mask[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 
        0,  0,  0,  0,  0,  0,  0,  0
    };

    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 4]);
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_pd(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_pd(&C[r * N + 4], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
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

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_pd(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_pd(&C[r * N + 4], packed_mask[1], packed_C[r][1]);
    }
#endif // d_kernel
}

void i_kernel(const int* packed_blockA, const int* packed_blockB, int* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 8      /* AVX512F */   /* 14x32 kernel */
    __m512i packed_C[14][2]; /* 14x32 */
    __m512i a_blockA, b0_blockB, b1_blockB;
    __mmask16 packed_mask_0 = (n < 16)  ? 0xFFFF >> (16 - n) : 0xFFFF;
    __mmask16 packed_mask_1 = (n >= 16) ? 0xFFFF >> (32 - n) : 0x0000;

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm512_maskz_loadu_epi32(packed_mask_0, &C[r * N + 0]);
        packed_C[r][1] = _mm512_maskz_loadu_epi32(packed_mask_1, &C[r * N + 16]);
    }
    for(int k = 0; k < kc; k++) {
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
        packed_C[8][0] = ifma(a_blockA, b0_blockB,packed_C[8][0]);
        packed_C[8][1] = ifma(a_blockA, b1_blockB,packed_C[8][1]);

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 9]); 
        packed_C[9][0] = ifma(a_blockA, b0_blockB,packed_C[9][0]);
        packed_C[9][1] = ifma(a_blockA, b1_blockB,packed_C[9][1]);

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 10]); 
        packed_C[10][0] = ifma(a_blockA, b0_blockB,packed_C[10][0]);
        packed_C[10][1] = ifma(a_blockA, b1_blockB,packed_C[10][1]);

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 11]); 
        packed_C[11][0] = ifma(a_blockA, b0_blockB,packed_C[11][0]);
        packed_C[11][1] = ifma(a_blockA, b1_blockB,packed_C[11][1]);

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 12]); 
        packed_C[12][0] = ifma(a_blockA, b0_blockB,packed_C[12][0]);
        packed_C[12][1] = ifma(a_blockA, b1_blockB,packed_C[12][1]);

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 13]); 
        packed_C[13][0] = ifma(a_blockA, b0_blockB,packed_C[13][0]);
        packed_C[13][1] = ifma(a_blockA, b1_blockB,packed_C[13][1]);

        packed_blockA += 1;     /* next column */
        packed_blockB += NC;    /* next 32 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm512_mask_storeu_epi32(&C[r * N + 0],  packed_mask_0, packed_C[r][0]);
        _mm512_mask_storeu_epi32(&C[r * N + 16], packed_mask_1, packed_C[r][1]);
    }
#elif INSTLEVEL >= 7    /* AVX2 */      /* 6x16 kernel */
    __m256i packed_C[6][2]; /* 6x16 */
    __m256i a_blockA, b0_blockB, b1_blockB;
    __m256i packed_mask[2];

    static int32_t mask[32] __attribute__((aligned(32))) = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    };
    
    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 8]);
    
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_epi32(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_epi32(&C[r * N + 8], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_loadu_si256((__m256i_u*)(packed_blockB + 0));
        b1_blockB = _mm256_loadu_si256((__m256i_u*)(packed_blockB + 8));

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 0]);
        packed_C[0][0] = ifma(a_blockA, b0_blockB, packed_C[0][0]); /* FMA */
        packed_C[0][1] = ifma(a_blockA, b1_blockB, packed_C[0][1]); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 1]);
        packed_C[1][0] = ifma(a_blockA, b0_blockB, packed_C[1][0]); /* FMA */
        packed_C[1][1] = ifma(a_blockA, b1_blockB, packed_C[1][1]); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 2]);
        packed_C[2][0] = ifma(a_blockA, b0_blockB, packed_C[2][0]); /* FMA */
        packed_C[2][1] = ifma(a_blockA, b1_blockB, packed_C[2][1]); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 3]);
        packed_C[3][0] = ifma(a_blockA, b0_blockB, packed_C[3][0]); /* FMA */
        packed_C[3][1] = ifma(a_blockA, b1_blockB, packed_C[3][1]); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 4]);
        packed_C[4][0] = ifma(a_blockA, b0_blockB, packed_C[4][0]); /* FMA */
        packed_C[4][1] = ifma(a_blockA, b1_blockB, packed_C[4][1]); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 5]);
        packed_C[5][0] = ifma(a_blockA, b0_blockB, packed_C[5][0]); /* FMA */
        packed_C[5][1] = ifma(a_blockA, b1_blockB, packed_C[5][1]); /* FMA */

        packed_blockA += 1;     /* next column */
        packed_blockB += NC;    /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_epi32(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_epi32(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
    }
#endif // i_kernel
}

void hq_kernel(const int16_t* packed_blockA, const int16_t* packed_blockB, int16_t* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 9      /* AVX512BW */
    __m512i packed_C[30]; /* 30x32 */
    __m512i a_blockA, b_blockB;
    __mmask32 packed_mask = 0xFFFFFFFF >> (32 - n);

    for (int r = 0; r < m; r++)
        packed_C[r] = _mm512_maskz_loadu_epi16(packed_mask, &C[r * N]);
    for(int k = 0; k < kc; k++) {
        b_blockB = _mm512_loadu_epi16(packed_blockB);

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 0]); 
        packed_C[0] = _mm512_add_epi16(packed_C[0], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 1]); 
        packed_C[1] = _mm512_add_epi16(packed_C[1], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 2]); 
        packed_C[2] = _mm512_add_epi16(packed_C[2], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 3]); 
        packed_C[3] = _mm512_add_epi16(packed_C[3], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 4]); 
        packed_C[4] = _mm512_add_epi16(packed_C[4], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 5]); 
        packed_C[5] = _mm512_add_epi16(packed_C[5], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 6]); 
        packed_C[6] = _mm512_add_epi16(packed_C[6], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 7]); 
        packed_C[7] = _mm512_add_epi16(packed_C[7], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 8]); 
        packed_C[8] = _mm512_add_epi16(packed_C[8], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 9]); 
        packed_C[9] = _mm512_add_epi16(packed_C[9], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 10]); 
        packed_C[10] = _mm512_add_epi16(packed_C[10], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 11]); 
        packed_C[11] = _mm512_add_epi16(packed_C[11], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 12]); 
        packed_C[12] = _mm512_add_epi16(packed_C[12], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 13]); 
        packed_C[13] = _mm512_add_epi16(packed_C[13], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 14]); 
        packed_C[14] = _mm512_add_epi16(packed_C[14], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 15]); 
        packed_C[15] = _mm512_add_epi16(packed_C[15], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 16]); 
        packed_C[16] = _mm512_add_epi16(packed_C[16], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 17]); 
        packed_C[17] = _mm512_add_epi16(packed_C[17], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 18]); 
        packed_C[18] = _mm512_add_epi16(packed_C[18], _mm512_mullo_epi16(b_blockB, a_blockA));
        
        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 19]); 
        packed_C[19] = _mm512_add_epi16(packed_C[19], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 20]); 
        packed_C[20] = _mm512_add_epi16(packed_C[20], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 21]); 
        packed_C[21] = _mm512_add_epi16(packed_C[21], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 22]); 
        packed_C[22] = _mm512_add_epi16(packed_C[22], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 23]); 
        packed_C[23] = _mm512_add_epi16(packed_C[23], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 24]); 
        packed_C[24] = _mm512_add_epi16(packed_C[24], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 25]); 
        packed_C[25] = _mm512_add_epi16(packed_C[25], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 26]); 
        packed_C[26] = _mm512_add_epi16(packed_C[26], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 27]); 
        packed_C[27] = _mm512_add_epi16(packed_C[27], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 28]); 
        packed_C[28] = _mm512_add_epi16(packed_C[28], _mm512_mullo_epi16(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi16(packed_blockA[KC * 29]); 
        packed_C[29] = _mm512_add_epi16(packed_C[29], _mm512_mullo_epi16(b_blockB, a_blockA));

        packed_blockA += 1;     /* next column */
        packed_blockB += NC;    /* next 32 elements*/
    }
    for(int r = 0; r < m; r++)
        _mm512_mask_storeu_epi16(&C[r * N],  packed_mask, packed_C[r]);
#elif INSTLEVEL >= 7 /* AVX2 */
#endif // q_kernel
}

void q_kernel(const int8_t* packed_blockA, const int8_t* packed_blockB, int8_t* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 9      /* AVX512BW */
    __m512i packed_C[30]; /* 30x64 */
    __m512i a_blockA, b_blockB;
    __mmask64 packed_mask = 0xFFFFFFFFFFFFFFFF >> (64 - n);

    for (int r = 0; r < m; r++)
        packed_C[r] = _mm512_maskz_loadu_epi8(packed_mask, &C[r * N]);
    for(int k = 0; k < kc; k++) {
        b_blockB = _mm512_loadu_epi8(packed_blockB);

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 0]); 
        packed_C[0] = _mm512_add_epi8(packed_C[0], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 1]); 
        packed_C[1] = _mm512_add_epi8(packed_C[1], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 2]); 
        packed_C[2] = _mm512_add_epi8(packed_C[2], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 3]); 
        packed_C[3] = _mm512_add_epi8(packed_C[3], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 4]); 
        packed_C[4] = _mm512_add_epi8(packed_C[4], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 5]); 
        packed_C[5] = _mm512_add_epi8(packed_C[5], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 6]); 
        packed_C[6] = _mm512_add_epi8(packed_C[6], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 7]); 
        packed_C[7] = _mm512_add_epi8(packed_C[7], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 8]); 
        packed_C[8] = _mm512_add_epi8(packed_C[8], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 9]); 
        packed_C[9] = _mm512_add_epi8(packed_C[9], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 10]); 
        packed_C[10] = _mm512_add_epi8(packed_C[10], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 11]); 
        packed_C[11] = _mm512_add_epi8(packed_C[11], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 12]); 
        packed_C[12] = _mm512_add_epi8(packed_C[12], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 13]); 
        packed_C[13] = _mm512_add_epi8(packed_C[13], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 14]); 
        packed_C[14] = _mm512_add_epi8(packed_C[14], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 15]); 
        packed_C[15] = _mm512_add_epi8(packed_C[15], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 16]); 
        packed_C[16] = _mm512_add_epi8(packed_C[16], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 17]); 
        packed_C[17] = _mm512_add_epi8(packed_C[17], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 18]); 
        packed_C[18] = _mm512_add_epi8(packed_C[18], int8_mul(b_blockB, a_blockA));
        
        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 19]); 
        packed_C[19] = _mm512_add_epi8(packed_C[19], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 20]); 
        packed_C[20] = _mm512_add_epi8(packed_C[20], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 21]); 
        packed_C[21] = _mm512_add_epi8(packed_C[21], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 22]); 
        packed_C[22] = _mm512_add_epi8(packed_C[22], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 23]); 
        packed_C[23] = _mm512_add_epi8(packed_C[23], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 24]); 
        packed_C[24] = _mm512_add_epi8(packed_C[24], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 25]); 
        packed_C[25] = _mm512_add_epi8(packed_C[25], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 26]); 
        packed_C[26] = _mm512_add_epi8(packed_C[26], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 27]); 
        packed_C[27] = _mm512_add_epi8(packed_C[27], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 28]); 
        packed_C[28] = _mm512_add_epi8(packed_C[28], int8_mul(b_blockB, a_blockA));

        a_blockA = _mm512_set1_epi8(packed_blockA[KC * 29]); 
        packed_C[29] = _mm512_add_epi8(packed_C[29], int8_mul(b_blockB, a_blockA));

        packed_blockA += 1;     /* next column */
        packed_blockB += NC;    /* next 32 elements*/
    }
    for(int r = 0; r < m; r++)
        _mm512_mask_storeu_epi8(&C[r * N],  packed_mask, packed_C[r]);
#elif INSTLEVEL >= 7 /* AVX2 */
#endif // q_kernel
}
#ifndef GEMM_H
#define GEMM_H 1

#pragma once 

#include "util.h"
#include "sse.h"

#define MEM_ALIGN 64

/********************************************************
 *                                                      
 *          GEMM                              
 *                                                      
*********************************************************/

void sgemm(const float* A, const float* B, float* C,
           const int M, const int N, const int K);
void dgemm(const double* A, const double* B, double* C,
           const int M, const int N, const int K);
void igemm(const int* A, const int* B, int* C,
           const int M, const int N, const int K);
void hqgemm(const int16_t* A, const int16_t* B, int16_t* C,
           const int M, const int N, const int K);
void qgemm(const int8_t* A, const int8_t* B, int8_t* C,
           const int M, const int N, const int K);

/********************************************************
 *                                                      
 *          Kernel
 *                                                      
*********************************************************/

void s_kernel(const float* packed_blockA, const float* packed_blockB, float* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void d_kernel(const double* packed_blockA, const double* packed_blockB, double* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void i_kernel(const int* packed_blockA, const int* packed_blockB, int* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void hq_kernel(const int16_t* packed_blockA, const int16_t* packed_blockB, int16_t* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void q_kernel(const int8_t* packed_blockA, const int8_t* packed_blockB, int8_t* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);

/********************************************************
 *                                                      
 *          Vector Operation
 *                                                      
*********************************************************/
#if INSTLEVEL >= 9 /* AVX512BW */
#if (defined (__AVX512__) || defined (__AVX512F__))
inline __m512i qmul(__m512i a, __m512i b) {
    __m512i a_even = a;
    __m512i a_odd_shifted = _mm512_and_si512(a, _mm512_set1_epi16(0xff00)); // save odd 
    __m512i b_even = b;
    __m512i b_odd_lo = _mm512_srli_epi16(b, 8);

    __m512i mul_even = _mm512_mullo_epi16(a_even, b_even);
    __m512i mul_odd  = _mm512_mullo_epi16(a_odd_shifted, b_odd_lo);
    
    // blend using the same vector constant we already needed, instead of a k mask
    // first source operand is a variable not needed later so it can be overwritten
    __m512i result = _mm512_ternarylogic_epi32(mul_even, _mm512_set1_epi16(0xff00), mul_odd, 0xB8); // 0xB8: B ? C : A
    return result;
}
#endif // AVX512F
#endif          /* INSTLEVEL */

/* FMA */
#if INSTLEVEL >= 9   /* AVX512BW */
#if defined (__AVX512BW__) && (defined (__AVX512__) || defined (__AVX512F__))
inline __m512  sfma(__m512 a, __m512 b, __m512 c)    { return _mm512_fmadd_ps(a, b, c); }
inline __m512d dfma(__m512d a, __m512d b, __m512d c) { return _mm512_fmadd_pd(a, b, c); }
inline __m512i ifma(__m512i a, __m512i b, __m512i c) { return _mm512_add_epi32(c, _mm512_mullo_epi32(a, b)); }
inline __m512i qfma(__m512i a, __m512i b, __m512i c) { return _mm512_add_epi8(c, qmul(a, b)); }
#endif
#elif INSTLEVEL >= 8 /* AVX512 */
#if defined (__AVX512__) || defined (__AVX512F__)
inline __m512  sfma(__m512 a, __m512 b, __m512 c)    { return _mm512_fmadd_ps(a, b, c); }
inline __m512d dfma(__m512d a, __m512d b, __m512d c) { return _mm512_fmadd_pd(a, b, c); }
inline __m512i ifma(__m512i a, __m512i b, __m512i c) { return _mm512_add_epi32(c, _mm512_mullo_epi32(a, b)); }
#endif // AVX512F
#elif INSTLEVEL >= 6 /* AVX */
#if defined (__FMA__)
inline __m256  sfma(__m256 a, __m256 b, __m256 c)    { return _mm256_fmadd_ps(a, b, c); }
inline __m256d dfma(__m256d a, __m256d b, __m256d c) { return _mm256_fmadd_pd(a, b, c); }
inline __m256i ifma(__m256i a, __m256i b, __m256i c) { return _mm256_add_epi32(c, _mm256_mullo_epi32(a, b)); }
#else  // No FMA
inline __m256  sfma(__m256 a, __m256 b, __m256 c)    { return _mm256_add_ps(c, _mm256_mul_ps(a, b)); }
inline __m256d dfma(__m256d a, __m256d b, __m256d c) { return _mm256_add_pd(c, _mm256_mul_pd(a, b)); }
inline __m256i ifma(__m256i a, __m256i b, __m256i c) { return _mm256_add_epi32(c, _mm256_mullo_epi32(a, b)); }
#endif // AVX, FMA
#endif              /* INSTLEVEL */

/********************************************************
 *                                                      
 *          Matrix Pack
 *                                                      
*********************************************************/

void spack_blockB(const float* B, float* packed_B, const int NR, 
                  const int nc, const int NC, const int N, 
                  const int kc, const int NTHREADS);
void spack_blockA(const float* A, float* packed_A, const int MR,
                  const int mc, const int kc, const int KC, 
                  const int K, const int NTHREADS);
void spack_panelB(const float* B, float* packed_B, 
                  const int nr, const int NC, const int N, const int kc);
void spack_panelA(const float* A, float* packed_A, 
                  const int mr, const int kc, const int KC, const int K);

void dpack_blockB(const double* B, double* packed_B, const int NR, 
                  const int nc, const int NC, const int N, 
                  const int kc, const int NTHREADS);
void dpack_blockA(const double* A, double* packed_A, const int MR,
                  const int mc, const int kc, const int KC, 
                  const int K, const int NTHREADS);
void dpack_panelB(const double* B, double* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void dpack_panelA(const double* A, double* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

void ipack_blockB(const int* B, int* packed_B, const int NR, 
                  const int nc, const int NC, const int N, 
                  const int kc, const int NTHREADS);
void ipack_blockA(const int* A, int* packed_A, const int MR,
                  const int mc, const int kc, const int KC, 
                  const int K, const int NTHREADS);
void ipack_panelB(const int* B, int* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void ipack_panelA(const int* A, int* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

void hqpack_blockB(const int16_t* B, int16_t* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void hqpack_blockA(const int16_t* A, int16_t* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void hqpack_panelB(const int16_t* B, int16_t* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void hqpack_panelA(const int16_t* A, int16_t* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

void qpack_blockB(const int8_t* B, int8_t* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void qpack_blockA(const int8_t* A, int8_t* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void qpack_panelB(const int8_t* B, int8_t* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void qpack_panelA(const int8_t* A, int8_t* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

/********************************************************
 *                                                      
 *          Hardware Optimization
 *                                                      
*********************************************************/

/**
 * Get cache information especially cache size.
 * This only cares about data cache and also L1 to L3.
 */
void show_cache(size_t* cache_size);
void get_cache_size(size_t* cache_size);
void set_block_size(size_t* cache_size, const int NTHREADS,
                    const int MR, const int NR,
                    int* MC, int* KC, int* NC, D_TYPE d_type);
void cache_opt(const int NTHREADS, const int MR, const int NR,
               int* MC, int* KC, int* NC, D_TYPE d_type);
int get_core_num();

#endif // GEMM_H
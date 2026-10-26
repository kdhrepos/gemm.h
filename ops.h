/**********************************************************************************************
 * File   : ops.c
 * Author : kdh     
 * Github : https://github.com/kdhrepos/gemm.h
 * 
 * Description: 
 *  Implementations that define basic arithmetic operations such as int8 multiplications, 
 *  and also FMAs for each of data types.
 * 
 * Notice:
 *  - Int8 multiplication does not well-supported by Intel, so it's implemented with
 *    int16 operations.
 *  - Integer FMA also does not well-supported by Intel. 
 *  - Those problems might be solved by AVX-VNNI and AVX-IFMA extensions.
 * 
**********************************************************************************************/

#ifndef OPS_H
#define OPS_H

#include "sse.h"

/* Basic operations */
#if INSTLEVEL >= 9 /* AVX512BW */
#if (defined (__AVX512__) || defined (__AVX512F__))
inline __m512i qmul(__m512i a, __m512i b) {
    __m512i mask = _mm512_set1_epi16(0xff00);

    __m512i a_odd_hi = _mm512_and_si512(a, mask);

    // Multiply vectors in INT16
    __m512i mul_even = _mm512_mullo_epi16(a, b);       // with high garbage

    __m512i mul_odd  = _mm512_maddubs_epi16(b, a_odd_hi);  // at the bottom of i16 elements, unlike previous version

    __m512i mul_odd_shifted = _mm512_slli_epi16(mul_odd, 0x8);

    __m512i result = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, mul_even, mul_odd_shifted);
    return result;
}
#elif INSTLEVEL >= 7 /* AVX2 */
inline __m256i qmul(__m256i a, __m256i b) {
    __m256i a_odd_hi = _mm256_and_si256(a, _mm256_set1_epi16(0xff00));

    // Multiply vectors in INT16
    __m256i mul_even = _mm256_mullo_epi16(a, b);
    __m256i mul_odd  = _mm256_maddubs_epi16(a_odd_hi, b);
    __m256i mul_odd_shifted = _mm256_slli_epi16(mul_odd, 0x8);

    __m256i mask = _mm256_set1_epi16(0xF0);
    __m256i result = _mm256_blendv_epi8(mul_odd_shifted, mul_even, mask);
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
#elif INSTLEVEL >= 7 /* AVX2 */
#if defined (__FMA__)
inline __m256  sfma(__m256 a, __m256 b, __m256 c)    { return _mm256_fmadd_ps(a, b, c); }
inline __m256d dfma(__m256d a, __m256d b, __m256d c) { return _mm256_fmadd_pd(a, b, c); }
inline __m256i ifma(__m256i a, __m256i b, __m256i c) { return _mm256_add_epi32(c, _mm256_mullo_epi32(a, b)); }
#else  // No FMA
inline __m256  sfma(__m256 a, __m256 b, __m256 c)    { return _mm256_add_ps(c, _mm256_mul_ps(a, b)); }
inline __m256d dfma(__m256d a, __m256d b, __m256d c) { return _mm256_add_pd(c, _mm256_mul_pd(a, b)); }
inline __m256i ifma(__m256i a, __m256i b, __m256i c) { return _mm256_add_epi32(c, _mm256_mullo_epi32(a, b)); }
#endif // AVX2, FMA
#elif INSTLEVEL >= 6 /* AVX */
#if defined (__FMA__)
inline __m256  sfma(__m256 a, __m256 b, __m256 c)    { return _mm256_fmadd_ps(a, b, c); }
inline __m256d dfma(__m256d a, __m256d b, __m256d c) { return _mm256_fmadd_pd(a, b, c); }
inline __m128i ifma(__m128i a, __m128i b, __m128i c) { return _mm_add_epi32(c, _mm_mullo_epi32(a, b)); }
#else  // No FMA
inline __m256  sfma(__m256 a, __m256 b, __m256 c)    { return _mm256_add_ps(c, _mm256_mul_ps(a, b)); }
inline __m256d dfma(__m256d a, __m256d b, __m256d c) { return _mm256_add_pd(c, _mm256_mul_pd(a, b)); }
inline __m128i ifma(__m128i a, __m128i b, __m128i c) { return _mm_add_epi32(c, _mm_mullo_epi32(a, b)); }
#endif // AVX, FMA
#endif              /* INSTLEVEL */

/* Memory operations */
#if INSTLEVEL >= 9   /* AVX512BW */
#elif INSTLEVEL >= 8 /* AVX512 */
#elif INSTLEVEL >= 7 /* AVX2 */
#elif INSTLEVEL >= 6 /* AVX*/
inline __m128i maskload(int* C, int8_t mask) {
    return _mm_set_epi32(
        ((mask & 0x08) == 0x08) ? C[3] : 0,
        ((mask & 0x04) == 0x04) ? C[2] : 0,
        ((mask & 0x02) == 0x02) ? C[1] : 0,
        ((mask & 0x01) == 0x01) ? C[0] : 0
    );
}
inline void maskstore(int* C, int8_t mask, __m128i packed_C) {
    if((mask & 0x08) == 0x08) C[3]=_mm_extract_epi32(packed_C, 3);
    if((mask & 0x04) == 0x04) C[2]=_mm_extract_epi32(packed_C, 2);
    if((mask & 0x02) == 0x02) C[1]=_mm_extract_epi32(packed_C, 1);
    if((mask & 0x01) == 0x01) C[0]=_mm_extract_epi32(packed_C, 0);
    return;
}
#endif              /* INSTLEVEL */

#endif
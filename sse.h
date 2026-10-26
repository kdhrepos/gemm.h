/**********************************************************************************************
 * File   : sse.h
 * Author : kdh     
 * Github : https://github.com/kdhrepos/gemm.h
 * 
 * Description: 
 *      This header file determines whether SIMD extensions can be used or not.
 *      If a certain extension can be used, then INSTLEVEL is added up.
 *      All the necessary Intel SIMD instrinsics depend on [INSTLEVEL] macro.
 * 
 * Reference:
 *    https://github.com/vectorclass/version2/blob/master/instrset.h
 *                                                    
**********************************************************************************************/

#ifndef SSE_H
#define SSE_H

#include <immintrin.h>
#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#endif

// The following values of INSTLEVEL are currently defined.
// Use the names instead of the numbers in C code (macros, so #if works too).
#define INSTLEVEL_NONE 0
#define INSTLEVEL_SSE 1
#define INSTLEVEL_SSE2 2
#define INSTLEVEL_SSE3 3
#define INSTLEVEL_SSE4_1 4
#define INSTLEVEL_SSE4_2 5
#define INSTLEVEL_AVX 6
#define INSTLEVEL_AVX2 7
#define INSTLEVEL_AVX512F 8
#define INSTLEVEL_AVX512BW 9
#define INSTLEVEL_AVX512VNNI 10

// TODO: Define another macros (or not...?)
#ifndef INSTLEVEL
#if defined(__AVX512VNNI__)
#define INSTLEVEL INSTLEVEL_AVX512VNNI
#elif defined(__AVX512BW__) /* && defined (__AVX512VL__) */
#define INSTLEVEL INSTLEVEL_AVX512BW
#elif defined(__AVX512F__) || defined(__AVX512__)
#define INSTLEVEL INSTLEVEL_AVX512F
#elif defined(__AVX2__)
#define INSTLEVEL INSTLEVEL_AVX2
#elif defined(__AVX__)
#define INSTLEVEL INSTLEVEL_AVX
#elif defined(__SSE4_2__)
#define INSTLEVEL INSTLEVEL_SSE4_2
#elif defined(__SSE4_1__)
#define INSTLEVEL INSTLEVEL_SSE4_1
#elif defined(__SSE3__)
#define INSTLEVEL INSTLEVEL_SSE3
#elif defined(__SSE2__)
#define INSTLEVEL INSTLEVEL_SSE2
#elif defined(__SSE__)
#define INSTLEVEL INSTLEVEL_SSE
#else
#define INSTLEVEL INSTLEVEL_NONE
#endif  // INSTLEVEL <LEVEL>
#endif  // INSTLEVEL
#endif  // SSE_H
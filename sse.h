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

// The following values of INSTRSET are currently defined:
// 2:  SSE2
// 3:  SSE3
// 4:  SSE4.1
// 5:  SSE4.2
// 6:  AVX
// 7:  AVX2
// 8:  AVX512F
// 9:  AVX512BW
// 10: AVX512VNNI

// TODO: Define another macros (or not...?)
 #ifndef INSTLEVEL
  #if defined (__AVX512VNNI__)
     #define INSTLEVEL 10
  #elif defined (__AVX512BW__) /* && defined (__AVX512VL__) */
     #define INSTLEVEL 9
  #elif defined (__AVX512F__) || defined (__AVX512__)
     #define INSTLEVEL 8
  #elif defined (__AVX2__)
     #define INSTLEVEL 7
  #elif defined (__AVX__)
     #define INSTLEVEL 6
  #elif defined (__SSE4_2__)
     #define INSTLEVEL 5
  #elif defined (__SSE4_1__)
     #define INSTLEVEL 4
  #elif defined (__SSE3__)
     #define INSTLEVEL 3
  #elif defined (__SSE2__)
     #define INSTLEVEL 2
  #elif defined (__SSE__)
     #define INSTLEVEL 1
  #else 
     #define INSTLEVEL 0
  #endif // INSTLEVEL <LEVEL>
 #endif // INSTLEVEL
#endif // SSE_H
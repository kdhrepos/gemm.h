/**********************************************************************************************
 * File   : opt.c
 * Author : kdh     
 * Github : https://github.com/kdhrepos/gemm.h
 * 
 * Description: 
 *      This file contains implementations of functions to make GEMM to be 
 *      well-optimized on hardware such as the number of CPU threads and cache size
 *                                                    
**********************************************************************************************/

#include "gemm.h"

/**
 * Get cache information especially cache size.
 * This only cares about data cache and also L1 to L3.
 */
void get_cache_size(size_t* cache_size) { 
    memset(cache_size, 0, sizeof(size_t) * 32);

    for (int cache = 0; cache < 32; cache++) {
        uint32_t eax, ebx, ecx, edx;

        eax = 4;     // get cache info
        ecx = cache; // cache id

        /* CPUID instruction */
        __asm__ (
            "cpuid" 
            : "+a" (eax) , "=b" (ebx) , "+c" (ecx) , "=d" (edx)
        );

        int cache_type = eax & 0x1F; // bits 04-00: cache type field
        if (cache_type == 0)        // no more cache
            break;
        if (cache_type == 2)        // pass instruction cache
            continue;

        int cache_level = (eax >>= 5) & 0x7; // bits 07-50: cache level, starts at 1

        uint32_t cache_sets = ecx + 1;                                          // bits 31-00
        uint32_t cache_coherency_line_size = (ebx & 0xFFF) + 1;                 // bits 11-00
        uint32_t cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;   // bits 21-12
        uint32_t cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;      // bits 31-22

        size_t total_cache_size = cache_ways_of_associativity * cache_physical_line_partitions 
                            * cache_coherency_line_size * cache_sets;

        cache_size[cache_level] = total_cache_size; // store only d-cache size
    }
}

void set_block_size(size_t* cache_size, const int NTHREADS,
                    const int MR, const int NR,
                    int* MC, int* KC, int* NC, D_TYPE d_type) {
    float MC_f = (*MC), NC_f = (*NC);
    int d_size = 0;
    if(d_type == D_FP32)        d_size = sizeof(float);
    else if(d_type == D_FP64)   d_size = sizeof(double);
    else if(d_type == D_INT32)  d_size = sizeof(int32_t);
    else if(d_type == D_INT16)  d_size = sizeof(int16_t);
    else if(d_type == D_INT8)   d_size = sizeof(int8_t);

    if(cache_size[1] != 0) {
        (*KC) = cache_size[1] / (NR * d_size);      // L1 = KC * NR
    }
    if(cache_size[2] != 0) {
        MC_f = cache_size[2] / ((*KC) * d_size);   // L2 = MC * KC
        MC_f /= (MR * NTHREADS);
        (*MC) = round(MC_f) * MR * NTHREADS;
    }
    if(cache_size[3] != 0) {
        NC_f = cache_size[3] / ((*MC) * d_size);   // L3 = NC * KC
        NC_f /= (NR * NTHREADS);
        (*NC) = round(NC_f) * NR * NTHREADS;
    }

#if DEBUG
    printf("MC : %d\n", (* MC));
    printf("KC : %d\n", (* KC));
    printf("NC : %d\n", (* NC));
#endif
}

void cache_opt(const int NTHREADS, const int MR, const int NR,
               int* MC, int* KC, int* NC, 
               D_TYPE d_type) {
    static size_t cache_size[32];      // CPUID 결과는 프로그램 전체에서 한 번만 읽는다
    static int cache_ready = 0;
    if (!cache_ready) {
        get_cache_size(cache_size);
        cache_ready = 1;
    }

    /* default value */
    *KC = 256;
    *MC = MR * NTHREADS;
    *NC = NR * NTHREADS;

    set_block_size(cache_size, NTHREADS, MR, NR, MC, KC, NC, d_type);

    if (*MC < MR * NTHREADS) *MC = MR * NTHREADS;
    if (*NC < NR * NTHREADS) *NC = NR * NTHREADS;

#if DEBUG
    printf("L1 size: %ld bytes\n", cache_size[1]);
    printf("L2 size: %ld bytes\n", cache_size[2]);
    printf("L3 size: %ld bytes\n", cache_size[3]);
#endif
}

int get_core_num() {
    uint32_t eax=0x1, ebx=0, ecx=0, edx=0;

    __asm__ (
      "cpuid" 
      : "+a" (eax) , "=b" (ebx) , "+c" (ecx) , "=d" (edx)
    );

#if DEBUG
    printf("NTHREADS: %d\n", ((ebx >> 16) & 0xFF));
#endif
    return ((ebx >> 16) & 0xFF); // the number of logical processors
}
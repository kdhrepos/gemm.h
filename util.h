/**********************************************************************************************
 * File   : util.h
 * Author : kdh     
 * Github : https://github.com/kdhrepos/gemm.h
 * 
 * Description: 
 *      Header file used for some util functions and inclusion of libraries.
 * 
**********************************************************************************************/

#ifndef UTIL_H
#define UTIL_H

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <omp.h>

typedef short BOOL;
#define TRUE 1
#define FALSE 0

#define min(a, b) ((a) < (b) ? (a) : (b))

typedef enum { D_ALL, D_FP32, D_FP64, D_INT32, D_INT8, D_INT16 } D_TYPE;

#define DEBUG TRUE

#endif  // UTIL_H
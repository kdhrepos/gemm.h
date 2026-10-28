#ifndef TEST_H
#define TEST_H

#include "gemm.h"
#include "report.h"

#define GEMM_H_TESTER_VERSION "1.0.0"

int sgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL csv_flag, BOOL verify_flag);
int dgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL csv_flag, BOOL verify_flag);
int igemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL csv_flag, BOOL verify_flag);
int hqgemm_test(const int M, const int N, const int K, const int niter,
                const int warmup, const int range, const int bound, FILE* file,
                BOOL console_flag, BOOL csv_flag, BOOL verify_flag);
int qgemm_test(const int M, const int N, const int K, const int niter,
               const int warmup, const int range, const int bound, FILE* file,
               BOOL console_flag, BOOL csv_flag, BOOL verify_flag);

uint64_t timer();

/* Naive GEMM writes its reference result into C, used as the
   correctness reference. */
void naive_sgemm(const float* A, const float* B, float* C, const int M,
                 const int N, const int K);
void naive_dgemm(const double* A, const double* B, double* C, const int M,
                 const int N, const int K);
void naive_igemm(const int* A, const int* B, int* C, const int M, const int N,
                 const int K);
void naive_hqgemm(const int16_t* A, const int16_t* B, int16_t* C, const int M,
                  const int N, const int K);
void naive_qgemm(const int8_t* A, const int8_t* B, int8_t* C, const int M,
                 const int N, const int K);

void fp32_get_rand_mat(int row, int col, float* mat, int bound);
void fp64_get_rand_mat(int row, int col, double* mat, int bound);
void int32_get_rand_mat(int row, int col, int32_t* mat, int bound);
void int16_get_rand_mat(int row, int col, int16_t* mat, int bound);
void int8_get_rand_mat(int row, int col, int8_t* mat, int bound);

/********************************************************
 *                                                      
 *          Print                                
 *                                                      
*********************************************************/
void fp32_print(int row, int col, float* mat);
void fp64_print(int row, int col, double* mat);
void int32_print(int row, int col, int32_t* mat);

#endif  // TEST_H

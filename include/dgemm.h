#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string.h>
#include <cblas.h>
#include <omp.h>
#include <openblas_config.h> 
#include <math.h>

#define M 4096
#define K 8192
#define N 4096

#define NUM_THREADS 16

#define NC 256
#define KC 256
#define MC 128

#define MR 4
#define NR 8

#define A(i, j) A[(j) * (lda) + (i)]
#define B(i, j) B[(j) * (ldb) + (i)]
#define C(i, j) C[(i) * (ldc) + (j)]
#define C_ref(i, j) C_ref[(j) * (ldc) + (i)]

#define a(i, j) a[(j) * (MR) + (i)]
#define b(i, j) b[(i) * (NR) + (j)]
#define c(i, j) c[(i) * (ldc) + (j)]

#define min(a, b) ((a) < (b) ? (a) : (b))


void dgemm(int, int, int, double*, int, double*, int, double*, int);
void macro_kernel(int, int, int, double*, double*, double*, int);
void micro_kernel(int, double*, double*, double*,int);

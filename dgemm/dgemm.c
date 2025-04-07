#include "dgemm.h"
#include <omp.h>

inline void packA_mcxkc_d(int m, int k, double *A, int ldA, int offseta, double *packA) {
    double *a_pntr[MR];

    for (int i = 0; i < m; i++) {
        a_pntr[i] = A + (offseta + i);
    }

    for (int i = m; i < MR; i++) {
        a_pntr[i] = A + (offseta + 0);
    }

    for (int p = 0; p < k; p++) {
        for (int i = 0; i < MR; i++) {
            *packA = *a_pntr[i];
            packA++;
            a_pntr[i] = a_pntr[i] + ldA;
        }
    }
}

inline void packB_kcxnc_d(int n, int k, double *B, int ldB, int offsetb, double *packB) {
    double *b_pntr[NR];

    for (int j = 0; j < n; j++) {
        b_pntr[j] = B + ldB * (offsetb + j);
    }

    for (int j = n; j < NR; j++) {
        b_pntr[j] = B + ldB * (offsetb + 0);
    }

    for (int p = 0; p < k; p++) {
        for (int j = 0; j < NR; j++) {
            *packB = *b_pntr[j];
            packB++;
            b_pntr[j]++;
        }
    }
}

void dgemm(int m, int n, int k, double *A, int lda, 
           double *B, int ldb, double *C, int ldc) {
    
    // 每个线程有自己的 packed_A 和 packed_B
    int num_threads = omp_get_max_threads();
    double** packed_A_per_thread = (double**)malloc(num_threads * sizeof(double*));
    double** packed_B_per_thread = (double**)malloc(num_threads * sizeof(double*));
    
    for (int t = 0; t < num_threads; t++) {
        packed_A_per_thread[t] = (double*)aligned_alloc(64, sizeof(double) * (MC + 1) * (KC + 1));
        packed_B_per_thread[t] = (double*)aligned_alloc(64, sizeof(double) * (NC + 1) * (KC + 1));
    }

    // 仅并行化 nc 循环，确保不同线程处理不同的列
    #pragma omp parallel for schedule(dynamic)
    for (int nc = 0; nc < n; nc += NC) {
        int thread_id = omp_get_thread_num();
        double* packed_A = packed_A_per_thread[thread_id];
        double* packed_B = packed_B_per_thread[thread_id];

        for (int kc = 0; kc < k; kc += KC) {
            int curr_nc = min(n - nc, NC);
            int curr_kc = min(k - kc, KC);

            // 打包 B
            for (int nr = 0; nr < curr_nc; nr += NR) {
                int curr_nr = min(curr_nc - nr, NR);
                packB_kcxnc_d(curr_nr, curr_kc, &B[kc], k, nc + nr, &packed_B[nr * curr_kc]);
            }

            // 处理 A 和计算 C
            for (int mc = 0; mc < m; mc += MC) {
                int curr_mc = min(m - mc, MC);
                
                // 打包 A
                for (int mr = 0; mr < curr_mc; mr += MR) {
                    int curr_mr = min(curr_mc - mr, MR);
                    packA_mcxkc_d(curr_mr, curr_kc, &A[kc * lda], m, mc + mr, &packed_A[mr * curr_kc]);
                }
                
                // 调用微内核（不同线程的 C 区域不重叠，无需保护）
                macro_kernel(curr_mc, curr_nc, curr_kc, packed_A, packed_B, &C(mc, nc), ldc);
            }
        }
    }

    // 释放资源
    for (int t = 0; t < num_threads; t++) {
        free(packed_A_per_thread[t]);
        free(packed_B_per_thread[t]);
    }
    free(packed_A_per_thread);
    free(packed_B_per_thread);
}
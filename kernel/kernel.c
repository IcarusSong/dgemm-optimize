#include "dgemm.h"

// void AddDot_int_4x8(int k, double *a, double *b, double *c, int ldc)
// {
//     __m512d c_temp[4];
//     for (int i = 0; i < 4; i++) {
//         c_temp[i] = _mm512_setzero_pd();
//     }
    
//     __m512d a_temp[4], b_temp;
//     for (int p = 0; p < k; p++) {

//         b_temp = _mm512_load_pd(&b(p, 0));

//         a_temp[0] = _mm512_set1_pd(a(0, p));
//         a_temp[1] = _mm512_set1_pd(a(1, p));
//         a_temp[2] = _mm512_set1_pd(a(2, p));
//         a_temp[3] = _mm512_set1_pd(a(3, p));

//         c_temp[0] = _mm512_fmadd_pd(a_temp[0], b_temp, c_temp[0]);
//         c_temp[1] = _mm512_fmadd_pd(a_temp[1], b_temp, c_temp[1]);
//         c_temp[2] = _mm512_fmadd_pd(a_temp[2], b_temp, c_temp[2]);
//         c_temp[3] = _mm512_fmadd_pd(a_temp[3], b_temp, c_temp[3]);
//     }
//     __m512d c0 = _mm512_load_pd(&c[0 * ldc]);
//     __m512d c1 = _mm512_load_pd(&c[1 * ldc]);
//     __m512d c2 = _mm512_load_pd(&c[2 * ldc]);
//     __m512d c3 = _mm512_load_pd(&c[3 * ldc]);

//     c0 = _mm512_add_pd(c0, c_temp[0]);
//     c1 = _mm512_add_pd(c1, c_temp[1]);
//     c2 = _mm512_add_pd(c2, c_temp[2]);
//     c3 = _mm512_add_pd(c3, c_temp[3]);
    
//     _mm512_storeu_pd(&c(0, 0), c0);
//     _mm512_storeu_pd(&c(1, 0), c1);
//     _mm512_storeu_pd(&c(2, 0), c2);
//     _mm512_storeu_pd(&c(3, 0), c3);
// }


// void AddDot_int_4x8(int k, double *a, double *b, double *c, int ldc) {
//     __m512d c_temp[4] = {0};
//     for (int p = 0; p < k; p++) {
//         // 只预取B的下一个块（最可能不连续的部分）
//         if (p + 1 < k) {
//             // _mm_prefetch((const char*)&b(p+1, 0), _MM_HINT_T0);
//         }

//         __m512d b_val = _mm512_load_pd(&b(p, 0));
//         c_temp[0] = _mm512_fmadd_pd(_mm512_set1_pd(a(0, p)), b_val, c_temp[0]);
//         c_temp[1] = _mm512_fmadd_pd(_mm512_set1_pd(a(1, p)), b_val, c_temp[1]);
//         c_temp[2] = _mm512_fmadd_pd(_mm512_set1_pd(a(2, p)), b_val, c_temp[2]);
//         c_temp[3] = _mm512_fmadd_pd(_mm512_set1_pd(a(3, p)), b_val, c_temp[3]);
//     }

//     // 直接累加到C，不预取（假设硬件能处理连续写入）
//     _mm512_storeu_pd(&c(0, 0), _mm512_add_pd(_mm512_load_pd(&c(0, 0)), c_temp[0]));
//     _mm512_storeu_pd(&c(1, 0), _mm512_add_pd(_mm512_load_pd(&c(1, 0)), c_temp[1]));
//     _mm512_storeu_pd(&c(2, 0), _mm512_add_pd(_mm512_load_pd(&c(2, 0)), c_temp[2]));
//     _mm512_storeu_pd(&c(3, 0), _mm512_add_pd(_mm512_load_pd(&c(3, 0)), c_temp[3]));
// }


void AddDot_int_4x8(int k, double *a, double *b, double *c, int ldc) {
    // 初始化累加寄存器
    __m512d c00 = _mm512_setzero_pd();
    __m512d c10 = _mm512_setzero_pd();
    __m512d c20 = _mm512_setzero_pd();
    __m512d c30 = _mm512_setzero_pd();

    // 2×循环展开：每次处理2个k的迭代
    int p;
    for (p = 0; p < k - 1; p += 2) {
        // 预取下下个B块（提前预取，避免延迟）
        _mm_prefetch((const char*)&b(p+2, 0), _MM_HINT_T0);
        _mm_prefetch((const char*)&b(p+3, 0), _MM_HINT_T0);
        _mm_prefetch((const char*)&a(0, p+4), _MM_HINT_T0);
        _mm_prefetch((const char*)&a(1, p+4), _MM_HINT_T0);
        // _mm_prefetch((const char*)&b(p+2, 0), _MM_HINT_T0); // 预取B[p+2]
        // _mm_prefetch((const char*)&a(0, p+4), _MM_HINT_T1); // 预取A列数据到L2
        // _mm_prefetch((const char*)&a(0, p+4) + 64, _MM_HINT_T1); // 预取相邻缓存行
        
        
        // 加载当前和下一个B块
        __m512d b0 = _mm512_load_pd(&b(p,   0));
        __m512d b1 = _mm512_load_pd(&b(p+1, 0));

        // 加载A的当前和下一个值，并广播
        __m512d a00 = _mm512_set1_pd(a(0, p));
        __m512d a01 = _mm512_set1_pd(a(0, p+1));
        __m512d a10 = _mm512_set1_pd(a(1, p));
        __m512d a11 = _mm512_set1_pd(a(1, p+1));
        __m512d a20 = _mm512_set1_pd(a(2, p));
        __m512d a21 = _mm512_set1_pd(a(2, p+1));
        __m512d a30 = _mm512_set1_pd(a(3, p));
        __m512d a31 = _mm512_set1_pd(a(3, p+1));

        // 双FMA累加
        c00 = _mm512_fmadd_pd(a00, b0, c00);
        c00 = _mm512_fmadd_pd(a01, b1, c00);
        c10 = _mm512_fmadd_pd(a10, b0, c10);
        c10 = _mm512_fmadd_pd(a11, b1, c10);
        c20 = _mm512_fmadd_pd(a20, b0, c20);
        c20 = _mm512_fmadd_pd(a21, b1, c20);
        c30 = _mm512_fmadd_pd(a30, b0, c30);
        c30 = _mm512_fmadd_pd(a31, b1, c30);
    }

    // 处理剩余奇数迭代（如果k是奇数）
    if (p < k) {
        __m512d b0 = _mm512_load_pd(&b(p, 0));
        c00 = _mm512_fmadd_pd(_mm512_set1_pd(a(0, p)), b0, c00);
        c10 = _mm512_fmadd_pd(_mm512_set1_pd(a(1, p)), b0, c10);
        c20 = _mm512_fmadd_pd(_mm512_set1_pd(a(2, p)), b0, c20);
        c30 = _mm512_fmadd_pd(_mm512_set1_pd(a(3, p)), b0, c30);
    }

    // 累加到C矩阵（使用对齐存储）
    _mm512_storeu_pd(&c(0, 0), _mm512_add_pd(_mm512_load_pd(&c(0, 0)), c00));
    _mm512_storeu_pd(&c(1, 0), _mm512_add_pd(_mm512_load_pd(&c(1, 0)), c10));
    _mm512_storeu_pd(&c(2, 0), _mm512_add_pd(_mm512_load_pd(&c(2, 0)), c20));
    _mm512_storeu_pd(&c(3, 0), _mm512_add_pd(_mm512_load_pd(&c(3, 0)), c30));
}


void macro_kernel(int m, int n, int k, double* packed_A, double *packed_B, double *C, int ldc){
    for (int j = 0; j < n; j += NR) {
        for (int i = 0; i < m; i += MR) {
            micro_kernel(k, &packed_A[i * k], &packed_B[j * k], &C(i, j), ldc);
        }
    }
}

void micro_kernel(int k, double *a, double *b, double *c,int ldc) {
    AddDot_int_4x8(k, a, b, c, ldc);
}

#include "dgemm.h"

void AddDot_int_4x8(int k, double *a, double *b, double *c, int ldc)
{
    __m512d c_temp[4];
    for (int i = 0; i < 4; i++) {
        c_temp[i] = _mm512_setzero_pd();
    }
    
    __m512d a_temp[4], b_temp;
    for (int p = 0; p < k; p++) {

        b_temp = _mm512_load_pd(&b(p, 0));

        a_temp[0] = _mm512_set1_pd(a(0, p));
        a_temp[1] = _mm512_set1_pd(a(1, p));
        a_temp[2] = _mm512_set1_pd(a(2, p));
        a_temp[3] = _mm512_set1_pd(a(3, p));

        c_temp[0] = _mm512_fmadd_pd(a_temp[0], b_temp, c_temp[0]);
        c_temp[1] = _mm512_fmadd_pd(a_temp[1], b_temp, c_temp[1]);
        c_temp[2] = _mm512_fmadd_pd(a_temp[2], b_temp, c_temp[2]);
        c_temp[3] = _mm512_fmadd_pd(a_temp[3], b_temp, c_temp[3]);
    }
    __m512d c0 = _mm512_load_pd(&c[0 * ldc]);
    __m512d c1 = _mm512_load_pd(&c[1 * ldc]);
    __m512d c2 = _mm512_load_pd(&c[2 * ldc]);
    __m512d c3 = _mm512_load_pd(&c[3 * ldc]);

    c0 = _mm512_add_pd(c0, c_temp[0]);
    c1 = _mm512_add_pd(c1, c_temp[1]);
    c2 = _mm512_add_pd(c2, c_temp[2]);
    c3 = _mm512_add_pd(c3, c_temp[3]);
    
    _mm512_storeu_pd(&c(0, 0), c0);
    _mm512_storeu_pd(&c(1, 0), c1);
    _mm512_storeu_pd(&c(2, 0), c2);
    _mm512_storeu_pd(&c(3, 0), c3);
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

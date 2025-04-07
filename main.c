#include "dgemm.h"


int main(int argc, char *argv[]) {
    // 设置默认线程数
    int num_threads = 1;  // 默认使用4线程
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    openblas_set_num_threads(num_threads);
    printf("Using %d OpenBLAS threads\n", num_threads);

    // 内存分配（64字节对齐）
    double* A = (double*)aligned_alloc(64, sizeof(double) * M * K);
    double* B = (double*)aligned_alloc(64, sizeof(double) * K * N);
    double* C = (double*)aligned_alloc(64, sizeof(double) * M * N);
    double* C_ref = (double*)aligned_alloc(64, sizeof(double) * M * N);

    int lda = M, ldb = K, ldc = N;  // A、B列主序布局，C行主序

    // 初始化矩阵（使用随机值）
    srand48(time(NULL));
    for (int j = 0; j < K; j++) {
        for (int i = 0; i < M; i++) {
            A(i, j) = drand48();
        }
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < K; i++) {
            B(i, j) = drand48();
        }
    }
    memset(C, 0, sizeof(double) * M * N);
    memset(C_ref, 0, sizeof(double) * M * N);

    // 计算理论浮点运算次数（2*M*N*K）
    double flops = 2.0 * M * N * K;

    // 1. 运行OpenBLAS的dgemm
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0, A, lda, B, ldb, 0.0, C_ref, ldc);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double blas_time = (end.tv_sec - start.tv_sec) + 
                      (end.tv_nsec - start.tv_nsec) / 1e9;
    double blas_gflops = (flops / 1e9) / blas_time;

    // clock_gettime(CLOCK_MONOTONIC, &start);
    
    // // 这里调用你的dgemm实现
    // dgemm(M, N, K, A, lda, B, ldb, C, ldc);
    
    // clock_gettime(CLOCK_MONOTONIC, &end);

    // double my_time = (end.tv_sec - start.tv_sec) + 
    //                   (end.tv_nsec - start.tv_nsec) / 1e9;
    // double my_gflops = (flops / 1e9) / my_time;

    // // 3. 验证正确性
    // double max_error = 0.0;
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         double err = fabs(C(i, j) - C_ref(i, j));
    //         if (err > max_error) max_error = err;            
    //     }
    // }
    printf("Matrix size: %d x %d x %d\n", M, K, N);
    printf("---------------------------------\n");
    printf("OpenBLAS time: %.4f sec, GFLOPS: %.2f\n", blas_time, blas_gflops);
    // printf("My Gemm time:    %.4f sec, GFLOPS: %.2f\n", my_time, my_gflops);
    // printf("Max error: %e\n", max_error);

    // 释放内存
    free(A); free(B); free(C); free(C_ref);
    return 0;
}

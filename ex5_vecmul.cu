#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 10000000
#define THREADS_PER_BLOCK 512

__global__ void vecMul(const int *a, const int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

int main(void) {
    size_t size = (size_t)N * sizeof(int);

    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c = (int*)malloc(size);

    if (!a || !b || !c) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    srand(0);

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
    }

    for (int i = 0; i < N; i++) {
        b[i] = rand() % 100;
    }

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Launching %d blocks x %d threads\n",
           blocks, THREADS_PER_BLOCK);

    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the last 1000 results
    for (int i = N - 1000; i < N; i++) {
        printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);
    }

    // Verify all 10 million results
    int mismatches = 0;

    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] * b[i]) {
            mismatches++;
        }
    }

    printf("Verification: %d mismatches out of %d\n",
           mismatches, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}

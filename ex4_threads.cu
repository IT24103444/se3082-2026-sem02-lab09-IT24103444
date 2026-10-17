#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 512

__global__ void addT(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(void) {
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;

    srand(0);

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    size_t size = N * sizeof(int);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 1 block, 512 threads
    addT<<<1, N>>>(d_a, d_b, d_c);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    int mismatches = 0;

    for (int i = 0; i < N; i++) {
        printf("%d) %d + %d = %d\n", i, a[i], b[i], c[i]);

        if (c[i] != a[i] + b[i]) {
            mismatches++;
        }
    }

    printf("Verification: %d mismatches out of %d\n", mismatches, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

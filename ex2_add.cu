#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main(void) {
    int a = 2, b = 7, c = 0;
    int *d_a, *d_b, *d_c;

    // Allocate GPU memory
    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    // Copy values from CPU to GPU
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // Run CUDA kernel
    add<<<1, 1>>>(d_a, d_b, d_c);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result from GPU to CPU
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result is %d\n", c);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

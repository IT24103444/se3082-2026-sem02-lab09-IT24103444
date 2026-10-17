#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ROWS 10000
#define COLS 10000

__global__ void mulElem(const int *a, const int *b,
                        int *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
}

int main(void) {
    size_t count = (size_t)ROWS * COLS;
    size_t size = count * sizeof(int);

    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c = (int*)malloc(size);

    if (!a || !b || !c) {
        printf("Host memory allocation failed\n");
        free(a); free(b); free(c);
        return 1;
    }

    srand(0);

    for (size_t i = 0; i < count; i++)
        a[i] = rand() % 100;

    for (size_t i = 0; i < count; i++)
        b[i] = rand() % 100;

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaError_t err;
    err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc((void**)&d_c, size);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    {
        dim3 threadsPerBlock(32, 16);
        dim3 numBlocks(
            (COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y
        );

        printf("Grid: %u x %u blocks\n",
               numBlocks.x, numBlocks.y);

        mulElem<<<numBlocks, threadsPerBlock>>>(
            d_a, d_b, d_c, ROWS, COLS
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    err = cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;

    // Print the last 1000 matrix elements
    for (size_t i = count - 1000; i < count; i++) {
        int row = (int)(i / COLS);
        int col = (int)(i % COLS);

        printf("C[%d][%d] %d x %d = %d\n",
               row, col, a[i], b[i], c[i]);
    }

    {
        size_t mismatches = 0;

        for (size_t i = 0; i < count; i++) {
            if (c[i] != a[i] * b[i])
                mismatches++;
        }

        printf("Verification: %zu mismatches out of %zu\n",
               mismatches, count);
    }

cleanup:
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return err == cudaSuccess ? 0 : 1;
}

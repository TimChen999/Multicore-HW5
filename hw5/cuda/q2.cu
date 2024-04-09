#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void compute(int *arrayA, int size, int *min, int *arrayB) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = threadIdx.x;

    // Copy data to shared memory
    if (tid < size)
        sdata[idx] = arrayA[tid];
    else
        sdata[idx] = INT_MAX;  // Set to maximum integer value for threads without data

    __syncthreads();

    // Use reduction technique to find the minimum value
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            if (sdata[idx + stride] < sdata[idx]) {
                sdata[idx] = sdata[idx + stride];
            }
        }
        __syncthreads();
    }

    // Write the minimum value to global memory
    if (idx == 0) {
        min[blockIdx.x] = sdata[0];
    }

    __syncthreads();

    // Compute the last digit of each element in array A and store it in array B
    if (tid < size)
        arrayB[tid] = arrayA[tid] % 10;
}

int main() {
    FILE *inputFile = fopen("inp.txt", "r");
    if (inputFile == NULL) {
        printf("Error opening inp.txt\n");
        return 1;
    }

    int arrayA[10000]; 
    int size = 0;
    int value;
    while (fscanf(inputFile, "%d,", &value) != EOF) {
        arrayA[size++] = value;
    }
    fclose(inputFile);

    int *d_arrayA;
    int *d_min;
    int *d_arrayB;
    int *h_min = (int*)malloc(sizeof(int) * (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    int arrayB[10000]; 

    cudaMalloc(&d_arrayA, sizeof(int) * size);
    cudaMalloc(&d_min, sizeof(int) * (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    cudaMalloc(&d_arrayB, sizeof(int) * size);

    cudaMemcpy(d_arrayA, arrayA, sizeof(int) * size, cudaMemcpyHostToDevice);

    compute<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_arrayA, size, d_min, d_arrayB);

    cudaMemcpy(h_min, d_min, sizeof(int) * (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
    cudaMemcpy(arrayB, d_arrayB, sizeof(int) * size, cudaMemcpyDeviceToHost);

    cudaFree(d_arrayA);
    cudaFree(d_min);
    cudaFree(d_arrayB);

    int minA = h_min[0];
    for (int i = 1; i < (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; i++) {
        if (h_min[i] < minA) {
            minA = h_min[i];
        }
    }

    FILE *outputFileA = fopen("q2a.txt", "w");
    if (outputFileA == NULL) {
        printf("Error opening q2a.txt\n");
        return 1;
    }
    fprintf(outputFileA, "%d", minA);
    fclose(outputFileA);

    FILE *outputFileB = fopen("q2b.txt", "w");
    if (outputFileB == NULL) {
        printf("Error opening q2b.txt\n");
        return 1;
    }
    for (int i = 0; i < size; i++) {
        fprintf(outputFileB, "%d", arrayB[i]);
        if (i != size - 1) {
            fprintf(outputFileB, ", "); // Add a comma if it's not the last value
        }
    }
    fclose(outputFileB);

    free(h_min);

    return 0;
}

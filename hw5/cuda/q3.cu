#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void computeGlobal(int *arrayA, int size, int *arrayB) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        int rangeIndex = arrayA[tid] / 100;
        atomicAdd(&arrayB[rangeIndex], 1);
    }
}

__global__ void computeShared(int *arrayA, int size, int *arrayB) {
    __shared__ int localB[10];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int rangeIndex;
    if (threadIdx.x < 10) {
        localB[threadIdx.x] = 0;
    }
    __syncthreads();

    if (tid < size) {
        rangeIndex = arrayA[tid] / 100;
        atomicAdd(&localB[rangeIndex], 1);
    }
    __syncthreads();

    if (threadIdx.x < 10) {
        atomicAdd(&arrayB[threadIdx.x], localB[threadIdx.x]);
    }
}

int main(int argc, char **argv) {
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

    // part A

    int *d_arrayA;
    int *d_arrayB;
    int arrayB[10] = {0};

    cudaMalloc(&d_arrayA, sizeof(int) * size);
    cudaMalloc(&d_arrayB, sizeof(int) * 10);

    cudaMemcpy(d_arrayA, arrayA, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrayB, arrayB, sizeof(int) * 10, cudaMemcpyHostToDevice);

    computeGlobal<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_arrayA, size, d_arrayB);

    cudaMemcpy(arrayB, d_arrayB, sizeof(int) * 10, cudaMemcpyDeviceToHost);

    cudaFree(d_arrayA);
    cudaFree(d_arrayB);

    FILE *outputFile = fopen("q3a.txt", "w");
    if (outputFile == NULL) {
        printf("Error opening q3a.txt\n");
        return 1;
    }
    for (int i = 0; i < 10; i++) {
        fprintf(outputFile, "%d", arrayB[i]);
        if (i < 9){
            fprintf(outputFile, ", ");
        }
    }
    fclose(outputFile);

    // part B
    memset(arrayB, 0, 80);

    cudaMalloc(&d_arrayA, sizeof(int) * size);
    cudaMalloc(&d_arrayB, sizeof(int) * 10);

    cudaMemcpy(d_arrayA, arrayA, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrayB, arrayB, sizeof(int) * 10, cudaMemcpyHostToDevice);

    computeShared<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_arrayA, size, d_arrayB);

    cudaMemcpy(arrayB, d_arrayB, sizeof(int) * 10, cudaMemcpyDeviceToHost);

    cudaFree(d_arrayA);
    cudaFree(d_arrayB);

    FILE *outputFileB = fopen("q3b.txt", "w");
    if (outputFileB == NULL) {
        printf("Error opening q3b.txt\n");
        return 1;
    }
    for (int i = 0; i < 10; i++) {
        fprintf(outputFileB, "%d", arrayB[i]);
        if (i < 9){
            fprintf(outputFileB, ", ");
        }
    }
    fclose(outputFileB);

    return 0;
}

// int main(int argc, char **argv)
// {
//     // Implement your solution for question 3. The input file is inp.txt
//     // and contains an array A (range of values is 0-999).
//     // Running this program should output three files:
//     //  (1) q3a.txt which contains an array B of size 10 that keeps a count of
//     //      the entries in each of the ranges: [0, 99], [100, 199], [200, 299], ..., [900, 999].
//     //      For this part, the array B should reside in global GPU memory during computation.
//     //  (2) q3b.txt which contains the same array B as in the previous part. However,
//     //      you must use shared memory to represent a local copy of B in each block, and
//     //      combine all local copies at the end to get a global copy of B.
//     //  (3) q3c.txt which contains an array C of size 10 that keeps a count of
//     //      the entries in each of the ranges: [0, 99], [0, 199], [0, 299], ..., [0, 999].
//     //      You should only use array B for this part (do not use the original input array A).
//     return 0;
// }

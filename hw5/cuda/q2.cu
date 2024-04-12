#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void compute(int *A, int size, int *minA, int *B) {
    __shared__ int sdata[THREADS_PER_BLOCK]; 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_minA = 1000;

    // Load element into shared memory
    if (tid < size) {
        sdata[threadIdx.x] = A[tid];
    } else {
        sdata[threadIdx.x] = 1000; 
    }
    __syncthreads();

    // Perform parallel reduction to compute minA
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Write block's minA to global memory
    if (threadIdx.x == 0) {
        atomicMin(minA, sdata[0]);
    }

    // Compute array B
    if (tid < size) {
        B[tid] = A[tid] % 10;
    }
}

int main(int argc, char **argv) {
    // Implement your solution for question 2. The input file is inp.txt
    // and contains an array A.
    // Running this program should output two files:
    //  (1) q2a.txt which contains the minimum value in the input array
    //  (2) q2b.txt which contains an array B (in the same format as inp.txt)
    //      where B[i] = the last digit of A[i]

    FILE *inputFile = fopen("inp2.txt", "r");
    if (inputFile == NULL) {
        printf("Error opening inp.txt\n");
        return 1;
    }

    int arrayA[100000]; 
    int size = 0;
    int value;
    while (fscanf(inputFile, "%d,", &value) != EOF) {
        arrayA[size++] = value;
    }

    fclose(inputFile);

    // Allocate memory on the device
    int *d_A, *d_B, *d_minA;
    cudaMalloc(&d_A, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_minA, sizeof(int));

    // Copy input array from host to device
    cudaMemcpy(d_A, arrayA, size * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize minA
    int initialMin = 1000;
    cudaMemcpy(d_minA, &initialMin, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel to compute minA and generate array B
    int numBlocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, size, d_minA, d_B);

    // Copy minA from device to host
    int minA;
    cudaMemcpy(&minA, d_minA, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy array B from device to host
    int arrayB[100000];
    cudaMemcpy(arrayB, d_B, size * sizeof(int), cudaMemcpyDeviceToHost);

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
            fprintf(outputFileB, ", ");
        }
    }
    fclose(outputFileB);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_minA);
    return 0;
}

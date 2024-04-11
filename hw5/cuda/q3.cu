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

__global__ void computeC(int *arrayB, int *arrayC, int N) {
    // Get C from B
    int tid = threadIdx.x;

    // Compute array C using array B
    arrayC[tid] = arrayB[tid+1];
    if(tid == 9){
        arrayC[tid] = N;
    }
}

__global__ void inclusiveScan(int *input, int *output, int size) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int offset = 1;

    // Load input data into shared memory
    int index = 2 * blockIdx.x * blockDim.x + tid;
    if (index < size) {
        temp[tid] = input[index];
    } else {
        temp[tid] = 0;  // Pad with zeros if out of bounds
    }
    if (index + blockDim.x < size) {
        temp[tid + blockDim.x] = input[index + blockDim.x];
    } else {
        temp[tid + blockDim.x] = 0;  // Pad with zeros if out of bounds
    }

    // Perform reduction phase
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element to zero
    if (tid == 0) {
        temp[blockDim.x * 2 - 1] = 0;
    }

    // Perform down-sweep phase
    for (int d = 1; d < blockDim.x * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    // Write the results to output array
    __syncthreads();
    if (index < size) {
        output[index] = temp[tid];
    }
    if (index + blockDim.x < size) {
        output[index + blockDim.x] = temp[tid + blockDim.x];
    }
}

int main(int argc, char **argv) {

    // Implement your solution for question 3. The input file is inp.txt
    // and contains an array A (range of values is 0-999).
    // Running this program should output three files:
    //  (1) q3a.txt which contains an array B of size 10 that keeps a count of
    //      the entries in each of the ranges: [0, 99], [100, 199], [200, 299], ..., [900, 999].
    //      For this part, the array B should reside in global GPU memory during computation.
    //  (2) q3b.txt which contains the same array B as in the previous part. However,
    //      you must use shared memory to represent a local copy of B in each block, and
    //      combine all local copies at the end to get a global copy of B.
    //  (3) q3c.txt which contains an array C of size 10 that keeps a count of
    //      the entries in each of the ranges: [0, 99], [0, 199], [0, 299], ..., [0, 999].
    //      You should only use array B for this part (do not use the original input array A).


    FILE *inputFile = fopen("inp.txt", "r");
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
    memset(arrayB, 0, sizeof(arrayB));

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

    // part C

    int arrayC[10];
    int *d_arrayC;

    cudaMalloc(&d_arrayB, sizeof(int) * 10);
    cudaMalloc(&d_arrayC, sizeof(int) * 10);

    cudaMemcpy(d_arrayB, arrayB, sizeof(int) * 10, cudaMemcpyHostToDevice);

    inclusiveScan<<<1, THREADS_PER_BLOCK, sizeof(int) * THREADS_PER_BLOCK * 2>>>(d_arrayB, d_arrayC, 10);

    computeC<<<1, 10>>>(d_arrayC, d_arrayC, size);

    cudaMemcpy(arrayC, d_arrayC, sizeof(int) * 10, cudaMemcpyDeviceToHost);

    FILE *outputFileC = fopen("q3c.txt", "w");
    if (outputFileC == NULL) {
        printf("Error opening q3c.txt\n");
        return 1;
    }
    for (int i = 0; i < 10; i++) {
        fprintf(outputFileC, "%d", arrayC[i]);
        if (i < 9){
            fprintf(outputFileC, ", ");
        }
    }
    fclose(outputFileC);
    
    cudaFree(d_arrayB);
    cudaFree(d_arrayC);


    return 0;
}


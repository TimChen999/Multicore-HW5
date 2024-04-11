#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void findOddNumbers(int *input, int *output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        if (input[tid] % 2 != 0) {
            output[tid] = input[tid];
        } else {
            output[tid] = -123;
        }
    }
}

int main() {

    // Implement your solution for question 4. The input file is inp.txt
    // and contains an array A.
    // Running this program should output one file:
    //  (1) q4.txt which contains an array D such that D contains only the odd
    //      numbers from the input array. You should preserve the order of the
    //      numbers as they are in the input array.

    FILE *inputFile = fopen("inp.txt", "r");
    if (inputFile == NULL) {
        printf("Error opening inp.txt\n");
        return 1;
    }

    int arrayA[100000];
    int arrayD[100000];
    int size = 0;
    int value;
    while (fscanf(inputFile, "%d,", &value) != EOF) {
        arrayA[size++] = value;
    }
    fclose(inputFile);

    int *d_input, *d_output;

    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    cudaMemcpy(d_input, arrayA, size * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    findOddNumbers<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, size);

    cudaMemcpy(arrayD, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *outputFile = fopen("q4.txt", "w");
    if (outputFile == NULL) {
        printf("Error opening q4.txt\n");
        return 1;
    }
    int written = 0;
    for (int i = 0; i < size; i++) {
        if (arrayD[i] != -123) {
            if (!written) {
                fprintf(outputFile, "%d", arrayD[i]);
                written = 1;
            }else{  
                fprintf(outputFile, ", %d", arrayD[i]);
            }
        }
    }
    fclose(outputFile);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

// int main(int argc, char **argv)
// {

//     return 0;
// }

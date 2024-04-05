
int main(int argc, char **argv)
{
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
    return 0;
}

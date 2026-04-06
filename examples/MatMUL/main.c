#include <stdio.h>
#include <stdlib.h>

// MLIR-generated function (memref descriptor signature)
void matmul(float*, float*, long, long, long, long, long,  // A: ptr, ptr, offset, size0, size1, stride0, stride1
            float*, float*, long, long, long, long, long,  // B
            float*, float*, long, long, long, long, long); // C

void print_matrix(const char* name, float mat[4][4]) {
    printf("%s:\n", name);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    float A[4][4] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    
    float B[4][4] = {
        {2, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 0, 2, 0},
        {0, 0, 0, 2}
    };
    
    float C[4][4] = {0};
    
    // Call MLIR function
    // Memref descriptor: allocated_ptr, aligned_ptr, offset, size0, size1, stride0, stride1
    matmul((float*)A, (float*)A, 0, 4, 4, 4, 1,
           (float*)B, (float*)B, 0, 4, 4, 4, 1,
           (float*)C, (float*)C, 0, 4, 4, 4, 1);
    
    printf("Identity × 2I = 2I (expected)\n\n");
    print_matrix("Result C", C);
    
    return 0;
}
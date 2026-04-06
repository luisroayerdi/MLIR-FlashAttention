#include <stdio.h>
#include <stdlib.h>

// Declare the MLIR-generated function (expanded memref signature)
void parallel_vector_add(float*, float*, long, long, long,
                         float*, float*, long, long, long,
                         float*, float*, long, long, long);

int main() {
    float a[1024], b[1024], c[1024];
    
    // Initialize arrays
    for (int i = 0; i < 1024; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 3.0f;
    }
    
    // Call MLIR function (memref descriptor: ptr, ptr, offset, size, stride)
    parallel_vector_add(a, a, 0, 1024, 1,
                       b, b, 0, 1024, 1,
                       c, c, 0, 1024, 1);
    
    // Verify
    printf("c[0] = %f (expected 0)\n", c[0]);
    printf("c[10] = %f (expected 30)\n", c[10]);

    for (int i = 0; i < 1024; i++) {
        printf("%f \n",c[i]);
    }
    
    return 0;
}
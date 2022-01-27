#include <immintrin.h>
#include <string.h>

/* const char* dgemm_desc = "Simple blocked dgemm."; */
const char* dgemm_desc = "Simple blocked dgemm. blocking";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

// add -funroll-loops -> 2 percent
static void do_block_fixsize(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // Put into contiguous array
    double A_cpy[BLOCK_SIZE * BLOCK_SIZE] = {0};
    double B_cpy[BLOCK_SIZE * BLOCK_SIZE] = {0};

    // col-major
    for(int j = 0; j < K; ++j){
        for(int i = 0; i < M; ++i){
            // M x K
            /* A_cpy[i + j * BLOCK_SIZE]= A[i + j * lda]; */
            // transpose
            A_cpy[j + i * BLOCK_SIZE]= A[i + j * lda];
        }
    }
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < K; ++i){
            // K x N
            B_cpy[i + j * BLOCK_SIZE] = B[i + j * lda];
        }
    }

    // For each row i of A
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        // For each column j of B
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                cij += A_cpy[k + i * BLOCK_SIZE] * B_cpy[k + j * BLOCK_SIZE];
            }
            C[i + j * lda] = cij;
        }
    }
}


static double* pack(int lda, double *A, int block_size){
    // padding
    int griddim = (lda + block_size - 1) / block_size;
    double *packed = (double *) _mm_malloc(griddim * griddim * block_size * block_size * sizeof(double), 64);

    memset(packed, 0, griddim * griddim * block_size * block_size * sizeof(double));

    for(int j = 0; j < lda; ++j){
        for(int i = 0; i < lda; ++i){
            // reindex
            int block_i = i / block_size;
            int block_j = j / block_size;
            int ii = i % block_size, jj = j % block_size;
            // packed[block_i][block_j][ii][jj]
            packed[(block_i + block_j * griddim) * (block_size * block_size) + ii + jj * block_size] = A[i + j * lda];
        }
    }
    return packed;
}

static double* pack_and_transpose(int lda, double *A, int block_size){
    // padding
    int griddim = (lda + block_size - 1) / block_size;
    double *packed = (double *) _mm_malloc(griddim * griddim * block_size * block_size * sizeof(double), 64);

    memset(packed, 0, griddim * griddim * block_size * block_size * sizeof(double));

    for(int j = 0; j < lda; ++j){
        for(int i = 0; i < lda; ++i){
            // (i, j) -> (j, i) (transpose) and then pack
            int block_i = j / block_size;
            int block_j = i / block_size;
            int ii = j % block_size, jj = i % block_size;
            // packed[block_i][block_j][ii][jj]
            packed[(block_i + block_j * griddim) * (block_size * block_size) + ii + jj * block_size] = A[i + j * lda];
        }
    }
    return packed;
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
/* void square_dgemm(int lda, double* A, double* B, double* C) { */
/*     // For each block-row of A */
/*     for (int i = 0; i < lda; i += BLOCK_SIZE) { */
/*         // For each block-column of B */
/*         for (int j = 0; j < lda; j += BLOCK_SIZE) { */
/*             // Accumulate block dgemms into block of C */
/*             for (int k = 0; k < lda; k += BLOCK_SIZE) { */
/*                 // Correct block dimensions if block "goes off edge of" the matrix */
/*                 int M = min(BLOCK_SIZE, lda - i); */
/*                 int N = min(BLOCK_SIZE, lda - j); */
/*                 int K = min(BLOCK_SIZE, lda - k); */
/*                 // Perform individual block dgemm */
/*                 do_block_fixsize(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda); */
/*             } */
/*         } */
/*     } */
/* } */

void square_dgemm(int lda, double* A, double* B, double* C) {
    // Copy A, B, C in contiguous format
    double *A_packed_trans = pack_and_transpose(lda, A, BLOCK_SIZE);
    double *B_packed = pack(lda, B, BLOCK_SIZE);
    double *C_packed = pack(lda, C, BLOCK_SIZE);

    // Use CUDA naming convention
    int griddim = (lda + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Loop over each block of C
    for(int bj = 0; bj < griddim; ++bj){
        for(int bi = 0; bi < griddim; ++bi){
            // Loop over the bi column of A_packed_trans
            // Loop over the bj column of B_packed
            for(int bk = 0; bk < griddim; ++bk){
                // A -> (bk, bi) (because of already transposed)
                // originally it is the bi's block row
                double *A_block = A_packed_trans + (bk + bi * griddim) * BLOCK_SIZE * BLOCK_SIZE;
                // B -> (bk, bj)
                double *B_block = B_packed + (bk + bj * griddim) * BLOCK_SIZE * BLOCK_SIZE;
                // C -> (bi, bj)
                double *C_block = C_packed + (bi + bj * griddim) * BLOCK_SIZE * BLOCK_SIZE;
                // small block matrix multiplication
                for(int j = 0; j < BLOCK_SIZE; ++j){
                    for(int i = 0; i < BLOCK_SIZE; ++i){
                        double c = 0.;
                        for(int k = 0; k < BLOCK_SIZE; k += 8){
                            // (i, k) * (k, j)
                            // but because of transpose -> (k, i) * (k, j)
                            // SIMD
                            __m512d Ar;
                            __m512d Br;
                            __m512d Cr;
                            Ar = _mm512_load_pd(A_block + k + i * BLOCK_SIZE);
                            Br = _mm512_load_pd(B_block + k + j * BLOCK_SIZE);
                            Cr = _mm512_mul_pd(Ar, Br);
                            c += _mm512_reduce_add_pd(Cr);
                        }
                        C_block[i + j * BLOCK_SIZE] += c;
                    }
                }
            }
        }
    }
    // Copy C_packed back to C
    for(int j = 0; j < lda; ++j){
        for(int i = 0; i < lda; ++i){
            int block_i = i / BLOCK_SIZE;
            int block_j = j / BLOCK_SIZE;
            int ii = i % BLOCK_SIZE, jj = j % BLOCK_SIZE;
            C[i + j * lda] = C_packed[
                (block_i + block_j * griddim) * BLOCK_SIZE * BLOCK_SIZE + ii + jj * BLOCK_SIZE];
        }
    }
}

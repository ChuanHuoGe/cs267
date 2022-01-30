#include <immintrin.h>
#include <assert.h>

#define CACHELINE 8
#define BLOCK_SIZE 32

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#define EXPERIMENT 14
#define min(a, b) (((a) < (b)) ? (a) : (b))

const char* dgemm_desc = "Blocking experiment: " STRINGIFY(EXPERIMENT) ", block_size: " STRINGIFY(BLOCK_SIZE);

inline void cpy(int N_pad, int N, double *from, double *to){
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < N; ++i){
            to[i + j * N_pad] = from[i + j * N];
        }
        for(int i = N; i < N_pad; ++i){
            to[i + j * N_pad] = 0;
        }
    }
    for(int j = N; j < N_pad; j++){
        for(int i = 0; i < N_pad; i++){
            to[i + j * N_pad] = 0;
        }
    }
}

inline void transpose_cpy(int N_pad, int N, double *from, double *to){
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < N; ++i){
            to[j + i * N_pad] = from[i + j * N];
        }
        for(int i = N; i < N_pad; ++i){
            to[j + i * N_pad] = 0;
        }
    }
    for(int i = 0; i < N_pad; i++){
        for(int j = N; j < N_pad; j++){
            to[j + i * N_pad] = 0;
        }
    }
}
void square_dgemm_block_jki(int N, double* A, double* B, double* C) {

    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int i = 0; i < N_pad; i += BLOCK_SIZE){
        for(int j = 0; j < N_pad; j += BLOCK_SIZE){
            for(int k = 0; k < N_pad; k += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                __m512d a,b,c;
                for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * N_pad;

                    for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                        double *A_col = A_block + kk * N_pad;
                        double *B_element = B_col + kk;

                        for(int ii = 0; ii < BLOCK_SIZE; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

void square_dgemm_block_kji(int N, double* A, double* B, double* C) {

    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int i = 0; i < N_pad; i += BLOCK_SIZE){
        for(int j = 0; j < N_pad; j += BLOCK_SIZE){
            for(int k = 0; k < N_pad; k += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                __m512d a,b,c;
                for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                    double *A_col = A_block + kk * N_pad;
                    double *B_row = B_block + kk;

                    for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                        double *B_element = B_row + jj * N_pad;
                        double *C_col = C_block + jj * N_pad;

                        for(int ii = 0; ii < BLOCK_SIZE; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}
void square_dgemm_jki_block_jki(int N, double* A, double* B, double* C) {
    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int j = 0; j < N_pad; j += BLOCK_SIZE){
        for(int k = 0; k < N_pad; k += BLOCK_SIZE){
            for(int i = 0; i < N_pad; i += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                __m512d a,b,c;
                for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * N_pad;

                    for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                        double *A_col = A_block + kk * N_pad;
                        double *B_element = B_col + kk;

                        for(int ii = 0; ii < BLOCK_SIZE; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}
void square_dgemm_kji_block_jki(int N, double* A, double* B, double* C) {
    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int k = 0; k < N_pad; k += BLOCK_SIZE){
        for(int j = 0; j < N_pad; j += BLOCK_SIZE){
            for(int i = 0; i < N_pad; i += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                __m512d a,b,c;
                for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * N_pad;

                    for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                        double *A_col = A_block + kk * N_pad;
                        double *B_element = B_col + kk;

                        for(int ii = 0; ii < BLOCK_SIZE; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

void square_dgemm_jik_block_jki(int N, double* A, double* B, double* C) {
    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int j = 0; j < N_pad; j += BLOCK_SIZE){
        for(int i = 0; i < N_pad; i += BLOCK_SIZE){
            for(int k = 0; k < N_pad; k += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                __m512d a,b,c;
                for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * N_pad;

                    for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                        double *A_col = A_block + kk * N_pad;
                        double *B_element = B_col + kk;

                        for(int ii = 0; ii < BLOCK_SIZE; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

void square_dgemm_ikj_block_jki(int N, double* A, double* B, double* C) {
    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int i = 0; i < N_pad; i += BLOCK_SIZE){
        for(int k = 0; k < N_pad; k += BLOCK_SIZE){
            for(int j = 0; j < N_pad; j += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                __m512d a,b,c;
                for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * N_pad;

                    for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                        double *A_col = A_block + kk * N_pad;
                        double *B_element = B_col + kk;

                        for(int ii = 0; ii < BLOCK_SIZE; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

#define BI(i) ((i) / BLOCK_SIZE)
#define BJ(j) ((j) / BLOCK_SIZE)
#define BII(i) ((i) % BLOCK_SIZE)
#define BJJ(j) ((j) % BLOCK_SIZE)

void cpy_and_pack(int N_pad, int N, double *from, double *to){
    int griddim = N_pad / BLOCK_SIZE;
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < N; ++i){
            to[(BI(i) + BI(j) * griddim) * BLOCK_SIZE * BLOCK_SIZE + BII(i) + BJJ(j) * BLOCK_SIZE] = from[i + j * N];
        }
        for(int i = N; i < N_pad; ++i){
            to[(BI(i) + BI(j) * griddim) * BLOCK_SIZE * BLOCK_SIZE + BII(i) + BJJ(j) * BLOCK_SIZE] = 0;
        }
    }
    for(int j = N; j < N_pad; ++j){
        for(int i = 0; i < N_pad; ++i){
            to[(BI(i) + BI(j) * griddim) * BLOCK_SIZE * BLOCK_SIZE + BII(i) + BJJ(j) * BLOCK_SIZE] = 0;
        }
    }
}

void square_dgemm_jki_block_jki_packing(int N, double* A, double* B, double* C) {
    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    // only pack A because A will be visited O(griddim * n^2)
    cpy_and_pack(N_pad, N, A, A_align);
    // because the whole B array will only be visited exactly once
    // packing will probably not be benefitcial at all
    cpy(N_pad, N, B, B_align);
    cpy_and_pack(N_pad, N, C, C_align);

    int griddim = N_pad / BLOCK_SIZE;

    for(int j = 0; j < griddim; ++j){
        for(int k = 0; k < griddim; ++k){
            for(int i = 0; i < griddim; ++i){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + (i + k * griddim) * BLOCK_SIZE * BLOCK_SIZE;
                double *B_block = B_align + (k * BLOCK_SIZE) + (j * BLOCK_SIZE) * N_pad;
                double *C_block = C_align + (i + j * griddim) * BLOCK_SIZE * BLOCK_SIZE;

                __m512d a,b,c;
                for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * BLOCK_SIZE;

                    for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                        double *A_col = A_block + kk * BLOCK_SIZE;
                        double *B_element = B_col + kk;

                        for(int ii = 0; ii < BLOCK_SIZE; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[(BI(i) + BI(j) * griddim) * BLOCK_SIZE * BLOCK_SIZE + BII(i) + BJJ(j) * BLOCK_SIZE];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

#define UNROLLSTEP 4

void square_dgemm_jki_block_jki_unroll(int N, double* A, double* B, double* C) {
    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int j = 0; j < N_pad; j += BLOCK_SIZE){
        for(int k = 0; k < N_pad; k += BLOCK_SIZE){
            for(int i = 0; i < N_pad; i += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * N_pad;

                    for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                        double *A_col = A_block + kk * N_pad;
                        double *B_element = B_col + kk;

                        for(int ii = 0; ii < BLOCK_SIZE; ii += UNROLLSTEP * CACHELINE){
/*[[[cog
import cog

UNROLLSTEP = 4
CACHELINE = 8

# define
for i in range(UNROLLSTEP):
    cog.out(
    """
                            __m512d b{i};
                            b{i} = _mm512_set1_pd(B_element[0]);
    """.format(i=i))

for i in range(UNROLLSTEP):
    cog.out(
    """
                            __m512d a{i};
                            __m512d c{i};
    """.format(i=i))

for i in range(UNROLLSTEP):
    cog.out(
    """
                            a{i} = _mm512_load_pd(A_col + ii + {i} * CACHELINE);
                            c{i} = _mm512_load_pd(C_col + ii + {i} * CACHELINE);
                            c{i} = _mm512_fmadd_pd(a{i}, b{i}, c{i});
                            _mm512_store_pd(C_col + ii + {i} * CACHELINE, c{i});
    """.format(i=i))

]]]*/

__m512d b0;
b0 = _mm512_set1_pd(B_element[0]);
    
__m512d b1;
b1 = _mm512_set1_pd(B_element[0]);
    
__m512d b2;
b2 = _mm512_set1_pd(B_element[0]);
    
__m512d b3;
b3 = _mm512_set1_pd(B_element[0]);
    
__m512d a0;
__m512d c0;
    
__m512d a1;
__m512d c1;
    
__m512d a2;
__m512d c2;
    
__m512d a3;
__m512d c3;
    
a0 = _mm512_load_pd(A_col + ii + 0 * CACHELINE);
c0 = _mm512_load_pd(C_col + ii + 0 * CACHELINE);
c0 = _mm512_fmadd_pd(a0, b0, c0);
_mm512_store_pd(C_col + ii + 0 * CACHELINE, c0);
    
a1 = _mm512_load_pd(A_col + ii + 1 * CACHELINE);
c1 = _mm512_load_pd(C_col + ii + 1 * CACHELINE);
c1 = _mm512_fmadd_pd(a1, b1, c1);
_mm512_store_pd(C_col + ii + 1 * CACHELINE, c1);
    
a2 = _mm512_load_pd(A_col + ii + 2 * CACHELINE);
c2 = _mm512_load_pd(C_col + ii + 2 * CACHELINE);
c2 = _mm512_fmadd_pd(a2, b2, c2);
_mm512_store_pd(C_col + ii + 2 * CACHELINE, c2);
    
a3 = _mm512_load_pd(A_col + ii + 3 * CACHELINE);
c3 = _mm512_load_pd(C_col + ii + 3 * CACHELINE);
c3 = _mm512_fmadd_pd(a3, b3, c3);
_mm512_store_pd(C_col + ii + 3 * CACHELINE, c3);
    
//[[[end]]]
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

#undef BLOCK_SIZE
// aim for L2
#define BLOCK_SIZE 160
// aim for L1
#define BLOCK_SIZE2 32

void square_dgemm_jki_block_jki_two_level(int N, double* A, double* B, double* C) {
    assert(BLOCK_SIZE * BLOCK_SIZE2 == 0);

    int N_pad = (N + BLOCK_SIZE2 - 1) / BLOCK_SIZE2 * BLOCK_SIZE2;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int j = 0; j < N_pad; j += BLOCK_SIZE){
        for(int k = 0; k < N_pad; k += BLOCK_SIZE){
            for(int i = 0; i < N_pad; i += BLOCK_SIZE){
                int I = min(N_pad - i, BLOCK_SIZE);
                int J = min(N_pad - j, BLOCK_SIZE);
                int K = min(N_pad - k, BLOCK_SIZE);
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                for(int jj = 0; jj < J; jj += BLOCK_SIZE2){
                    for(int kk = 0; kk < K; kk += BLOCK_SIZE2){
                        for(int ii = 0; ii < I; ii += BLOCK_SIZE2){
                            double *A_block2 = A_block + ii + kk * N_pad;
                            double *B_block2 = B_block + kk + jj * N_pad;
                            double *C_block2 = C_block + ii + jj * N_pad;


                            for(int jjj = 0; jjj < BLOCK_SIZE2; ++jjj){
                                double *B_col = B_block2 + jjj * N_pad;
                                double *C_col = C_block2 + jjj * N_pad;

                                for(int kkk = 0; kkk < BLOCK_SIZE2; ++kkk){
                                    double *A_col = A_block2 + kkk * N_pad;
                                    double *B_element = B_col + kkk;

                                    for(int iii = 0; iii < BLOCK_SIZE2; iii += CACHELINE){
                                        __m512d a,b,c;
                                        a = _mm512_load_pd(A_col + iii);
                                        b = _mm512_set1_pd(B_element[0]);
                                        c = _mm512_load_pd(C_col + iii);
                                        c = _mm512_fmadd_pd(a, b, c);
                                        _mm512_store_pd(C_col + iii, c);
                                    }
                                }
                            }

                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}
#undef BLOCK_SIZE
#define BLOCK_SIZE 32

#define PREFETCH_DIST 8

void square_dgemm_jki_block_jki_prefetch(int N, double* A, double* B, double* C) {
    int N_pad = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int j = 0; j < N_pad; j += BLOCK_SIZE){
        for(int k = 0; k < N_pad; k += BLOCK_SIZE){
            for(int i = 0; i < N_pad; i += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                __m512d a,b,c;
                for(int jj = 0; jj < BLOCK_SIZE; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * N_pad;

                    for(int kk = 0; kk < BLOCK_SIZE; ++kk){
                        double *A_col = A_block + kk * N_pad;
                        double *B_element = B_col + kk;

                        _mm_prefetch(A_col + PREFETCH_DIST * N_pad, _MM_HINT_T0);
                        _mm_prefetch(C_col + PREFETCH_DIST * N_pad, _MM_HINT_T0);

                        for(int ii = 0; ii < BLOCK_SIZE; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}
//----------------------------------------
void square_dgemm_jki_block_jki_nopad(int N, double* A, double* B, double* C) {
    int N_pad = (N + BLOCK_SIZE - 1) / CACHELINE * CACHELINE;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int j = 0; j < N_pad; j += BLOCK_SIZE){
        int J = min(BLOCK_SIZE, N_pad - j);

        for(int k = 0; k < N_pad; k += BLOCK_SIZE){
            int K = min(BLOCK_SIZE, N_pad - k);
            for(int i = 0; i < N_pad; i += BLOCK_SIZE){
                // read: A_align[i:i+BLOCK_SIZE][k:k+BLOCK_SIZE]
                // read: B_align[k:k+BLOCK_SIZE][j:j+BLOCK_SIZE]
                // read&write: C_align[i:i+BLOCK_SIZE][j:j+BLOCK_SIZE]
                //
                // (I x K) x (K x J)
                int I = min(BLOCK_SIZE, N_pad - i);

                double *A_block = A_align + i + k * N_pad;
                double *B_block = B_align + k + j * N_pad;
                double *C_block = C_align + i + j * N_pad;
                __m512d a,b,c;
                for(int jj = 0; jj < J; ++jj){
                    double *B_col = B_block + jj * N_pad;
                    double *C_col = C_block + jj * N_pad;

                    for(int kk = 0; kk < K; ++kk){
                        double *A_col = A_block + kk * N_pad;
                        double *B_element = B_col + kk;

                        for(int ii = 0; ii < I; ii += CACHELINE){
                            a = _mm512_load_pd(A_col + ii);
                            b = _mm512_set1_pd(B_element[0]);
                            c = _mm512_load_pd(C_col + ii);
                            c = _mm512_fmadd_pd(a, b, c);
                            _mm512_store_pd(C_col + ii, c);
                        }
                    }
                }
            }
        }
    }
    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

#define KC 256
#define MC 256
#define NR 32
#define MR 32
void square_dgemm_gotoblas_block_jki(int N, double* A, double* B, double* C) {
    assert(NR == MR);
    assert(NR % CACHELINE == 0);
    int N_pad = (N + MR - 1) / MR * MR;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int pc = 0; pc < N_pad; pc += KC){
        int K = min(N_pad - pc, KC);
        /* assert(K % NR == 0); */
        for(int ic = 0; ic < N_pad; ic += MC){
            int IR = min(N_pad - ic, MC);
            /* assert(IR % NR == 0); */
            for(int jr = 0; jr < N_pad; jr += NR){
                for(int ir = 0; ir < IR; ir += MR){
                    // A_align[ic+ir:ic+ir+MR][pc:pc+K]
                    // B_align[pc:pc+K][jr:jr+NR]
                    // C_align[ic+ir:ic+ir+MR][jr:jr+NR]
                    double *const A_sliver = A_align + (ic + ir) + (pc) * N_pad;
                    double *const B_sliver = B_align + (pc) + (jr) * N_pad;
                    double *const C_block = C_align + (ic + ir) + (jr) * N_pad;
                    // (MR x K) x (K x NR)
                    __m512d a,b,c;
                    for(int j = 0; j < NR; ++j){
                        double *const B_col = B_sliver + j * N_pad;
                        double *const C_col = C_block + j * N_pad;
                        for(int k = 0; k < K; ++k){
                            double *const B_element = B_col + k;
                            double *const A_col = A_sliver + k * N_pad;
                            for(int i = 0; i < MR; i += CACHELINE){
                                a = _mm512_load_pd(A_col + i);
                                b = _mm512_set1_pd(B_element[0]);
                                c = _mm512_load_pd(C_col + i);
                                c = _mm512_fmadd_pd(a, b, c);
                                _mm512_store_pd(C_col + i, c);
                            }
                        }
                    }
                }
            }
        }
    }

    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}
void square_dgemm_gotoblas_block_kji(int N, double* A, double* B, double* C) {
    assert(NR == MR);
    assert(NR % CACHELINE == 0);
    int N_pad = (N + MR - 1) / MR * MR;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int pc = 0; pc < N_pad; pc += KC){
        int K = min(N_pad - pc, KC);
        /* assert(K % NR == 0); */
        for(int ic = 0; ic < N_pad; ic += MC){
            int IR = min(N_pad - ic, MC);
            /* assert(IR % NR == 0); */
            for(int jr = 0; jr < N_pad; jr += NR){
                for(int ir = 0; ir < IR; ir += MR){
                    // A_align[ic+ir:ic+ir+MR][pc:pc+K]
                    // B_align[pc:pc+K][jr:jr+NR]
                    // C_align[ic+ir:ic+ir+MR][jr:jr+NR]
                    double *const A_sliver = A_align + (ic + ir) + (pc) * N_pad;
                    double *const B_sliver = B_align + (pc) + (jr) * N_pad;
                    double *const C_block = C_align + (ic + ir) + (jr) * N_pad;
                    // (MR x K) x (K x NR)
                    __m512d a,b,c;
                    for(int k = 0; k < K; ++k){
                        double *const B_row = B_sliver + k;
                        double *const A_col = A_sliver + k * N_pad;

                        for(int j = 0; j < NR; ++j){
                            double *const C_col = C_block + j * N_pad;
                            double *const B_element = B_row + j * N_pad;

                            for(int i = 0; i < MR; i += CACHELINE){
                                a = _mm512_load_pd(A_col + i);
                                b = _mm512_set1_pd(B_element[0]);
                                c = _mm512_load_pd(C_col + i);
                                c = _mm512_fmadd_pd(a, b, c);
                                _mm512_store_pd(C_col + i, c);
                            }
                        }
                    }
                }
            }
        }
    }

    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

// Check https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
void square_dgemm_gotoblas_block_kji_packing(int N, double* A, double* B, double* C) {
    assert(NR == MR);
    assert(NR % CACHELINE == 0);
    int N_pad = (N + MR - 1) / MR * MR;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    double *A_pack = (double *)_mm_malloc(MC * KC * sizeof(double), CACHELINE * sizeof(double));
    double *B_pack = (double *)_mm_malloc(KC * N_pad * sizeof(double), CACHELINE * sizeof(double));

    for(int pc = 0; pc < N_pad; pc += KC){
        int K = min(N_pad - pc, KC);
        for(int ic = 0; ic < N_pad; ic += MC){
            int IR = min(N_pad - ic, MC);

            // pack A
            for(int j = 0; j < K; ++j){
                for(int i = 0; i < IR; ++i){
                    A_pack[(i / MR) * MR * K + (i % MR) + j * MR] = A_align[(ic + i) + (pc + j) * N_pad];
                }
            }
            // pack B
            for(int j = 0; j < N_pad; ++j){
                for(int i = 0; i < K; ++i){
                    B_pack[(j / NR) * K * NR + (i * NR + j % NR)] = B_align[(pc + i) + (j) * N_pad];
                }
            }

            for(int jr = 0; jr < N_pad; jr += NR){
                for(int ir = 0; ir < IR; ir += MR){
                    // A_align[ic+ir:ic+ir+MR][pc:pc+K]
                    // B_align[pc:pc+K][jr:jr+NR]
                    // C_align[ic+ir:ic+ir+MR][jr:jr+NR]

                    double *const A_sliver = A_pack + (ir / MR) * MR * K;
                    double *const B_sliver = B_pack + (jr / NR) * K * NR;
                    double *const C_block = C_align + (ic + ir) + (jr) * N_pad;
                    // (MR x K) x (K x NR)
                    __m512d a,b,c;
                    for(int k = 0; k < K; ++k){
                        double *const B_row = B_sliver + k * NR;
                        double *const A_col = A_sliver + k * MR;

                        for(int j = 0; j < NR; ++j){
                            double *const C_col = C_block + j * N_pad;
                            double *const B_element = B_row + j;

                            for(int i = 0; i < MR; i += CACHELINE){
                                a = _mm512_load_pd(A_col + i);
                                b = _mm512_set1_pd(B_element[0]);
                                c = _mm512_load_pd(C_col + i);
                                c = _mm512_fmadd_pd(a, b, c);
                                _mm512_store_pd(C_col + i, c);
                            }
                        }
                    }
                }
            }
        }
    }

    // put back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
    _mm_free(A_pack);
    _mm_free(B_pack);
}
//
// entry point
void square_dgemm(int N, double* A, double* B, double* C) {
#if EXPERIMENT == 1
    // about 19%
    square_dgemm_block_jki(N, A, B, C);
#elif EXPERIMENT == 2
    // about 12%
    square_dgemm_block_kji(N, A, B, C);
#elif EXPERIMENT == 3
    // about 22%
    square_dgemm_jki_block_jki(N, A, B, C);
#elif EXPERIMENT == 4
    // about 20.5%
    square_dgemm_kji_block_jki(N, A, B, C);
#elif EXPERIMENT == 5
    // about 20.49%
    square_dgemm_jik_block_jki(N, A, B, C);
#elif EXPERIMENT == 6
    // about 17.86%
    square_dgemm_ikj_block_jki(N, A, B, C);
#elif EXPERIMENT == 7
    // about 20.8%
    square_dgemm_jki_block_jki_packing(N, A, B, C);
#elif EXPERIMENT == 8
    // about 22%
    square_dgemm_jki_block_jki_unroll(N, A, B, C);
#elif EXPERIMENT == 9
    // about 20% (weird...)
    square_dgemm_jki_block_jki_two_level(N, A, B, C);
#elif EXPERIMENT == 10
    // about 19%
    square_dgemm_jki_block_jki_prefetch(N, A, B, C);
#elif EXPERIMENT == 11
    // about 9.51%
    square_dgemm_jki_block_jki_nopad(N, A, B, C);
#elif EXPERIMENT == 12
    // about 18%
    square_dgemm_gotoblas_block_jki(N, A, B, C);
#elif EXPERIMENT == 13
    // about 13%
    square_dgemm_gotoblas_block_kji(N, A, B, C);
#elif EXPERIMENT == 14
    // about 13%
    square_dgemm_gotoblas_block_kji_packing(N, A, B, C);
#else
    assert(0);
#endif
}

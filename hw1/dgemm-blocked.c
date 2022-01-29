#include <immintrin.h>

#define CACHELINE 8
#define BLOCK_SIZE 32

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#define EXPERIMENT 2

const char* dgemm_desc = "Blocking experiment: " STRINGIFY(EXPERIMENT) "";

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
void square_dgemm_block_jik(int N, double* A, double* B, double* C) {

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
// entry point
void square_dgemm(int N, double* A, double* B, double* C) {
#if EXPERIMENT == 1
    square_dgemm_block_jik(N, A, B, C);
#elif EXPERIMENT == 2
    square_dgemm_block_kji(N, A, B, C);
#else
    assert(0);
#endif
}

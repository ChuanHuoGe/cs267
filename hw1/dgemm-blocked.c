#include <immintrin.h>
#include <string.h>
#include <assert.h>

const char* dgemm_desc = "Pure SIMD";
/*
It likely only has 32 AVX512 registers. But the AVX2 instruction set only supports 16 ymm registers in the instruction set. Any AVX2 operations executing on KNL will only one half of each AVX512 register (there aren't a separate set of registers for AVX2, it's just going to repurpose the registers for AVX512. From there, there are two possibilities:
- You would only be able to use 16 registers, since that's the ymm count in the AVX2 instruction set

- The compiler will be smart enough to recognize that it really has 32 zmm registers available, and it will generate instructions to use the bottom half of each one.

Either way, you won't see 64 256-bit registers. I'm not sure if the compiler is even smart enough to do the second thing. At this point, you should generate the assembly code, figure out what the behavior is, and include it in your report.

A side note: we should distinguish between the Cori machine, which has thousands of both KNL and Haswell nodes, and a single KNL chip, which is what we are talking about.
 */

// Assume there are only 16 AVX512 registers

// cacheline is 64 bytes -> 8 doubles
#define CACHELINE 8

#define MICRO_H 16
// W is allowed to be not a multiple of CACHELINE
#define MICRO_W 6

void cpy(int N_pad, int N, double *from, double *to){
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

void transpose_cpy(int N_pad, int N, double *from, double *to){
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

// A: 16 x 1
// B: 1 x 6 (but transposed)
// C: 16 x 6
static void micro_kernel_16by6(
        int N_pad,
        double *A_panel, double *B_panel, double *C_block
        ){

for(int k = 0; k < N_pad; ++k){
    /* N_pad, A_panel + k * N_pad, B_panel + k * N_pad, C_block */
    double *A = A_panel + k * N_pad;
    double *B = B_panel + k * N_pad;
    double *C = C_block;
/*[[[cog
import cog

AVX_DOUBLE_NUM = 8

# block C
for i in range(16 // AVX_DOUBLE_NUM):
    for j in range(6):
        cog.outl("__m512d c{}{};".format(i, j))

# column of A
for i in range(16 // AVX_DOUBLE_NUM):
    cog.outl("__m512d a{};".format(i))

# col of B (transposed)
# we only have 2 registers remaining (we only have 16 registers)
step = 2
for i in range(step):
    cog.outl("__m512d b{};".format(i))

# load block C
for j in range(6):
    for i in range(16 // AVX_DOUBLE_NUM):
        cog.outl("c{i}{j} = _mm512_load_pd(C + {i} * {AVX_DOUBLE_NUM} + N_pad * {j});".format(
                i=i, j=j, AVX_DOUBLE_NUM=AVX_DOUBLE_NUM))

# load A
for i in range(16 // AVX_DOUBLE_NUM):
    cog.outl("a{i} = _mm512_load_pd(A + {i} * {AVX_DOUBLE_NUM});".format(i=i, AVX_DOUBLE_NUM=AVX_DOUBLE_NUM))

# computation!!
for i in range(16 // AVX_DOUBLE_NUM):
    for j in range(6 // step):
        # tricky part: because we only have 2 registers for B
        for k in range(step):
            bidx = k + j * step
            cog.outl("b{k} = _mm512_set1_pd(B[{bidx}]);".format(k=k, bidx=bidx))

        for k in range(step):
            bidx = k + j * step
            cog.outl("c{i}{bidx} = _mm512_fmadd_pd(a{i}, b{k}, c{i}{bidx});".format(i=i,k=k,bidx=bidx))

# write
for j in range(6):
    for i in range(16 // AVX_DOUBLE_NUM):
        cog.outl("_mm512_store_pd(C + {i} * {AVX_DOUBLE_NUM} + N_pad * {j}, c{i}{j});".format(i=i, j=j,
                AVX_DOUBLE_NUM=AVX_DOUBLE_NUM))


]]]*/
__m512d c00;
__m512d c01;
__m512d c02;
__m512d c03;
__m512d c04;
__m512d c05;
__m512d c10;
__m512d c11;
__m512d c12;
__m512d c13;
__m512d c14;
__m512d c15;
__m512d a0;
__m512d a1;
__m512d b0;
__m512d b1;
c00 = _mm512_load_pd(C + 0 * 8 + N_pad * 0);
c10 = _mm512_load_pd(C + 1 * 8 + N_pad * 0);
c01 = _mm512_load_pd(C + 0 * 8 + N_pad * 1);
c11 = _mm512_load_pd(C + 1 * 8 + N_pad * 1);
c02 = _mm512_load_pd(C + 0 * 8 + N_pad * 2);
c12 = _mm512_load_pd(C + 1 * 8 + N_pad * 2);
c03 = _mm512_load_pd(C + 0 * 8 + N_pad * 3);
c13 = _mm512_load_pd(C + 1 * 8 + N_pad * 3);
c04 = _mm512_load_pd(C + 0 * 8 + N_pad * 4);
c14 = _mm512_load_pd(C + 1 * 8 + N_pad * 4);
c05 = _mm512_load_pd(C + 0 * 8 + N_pad * 5);
c15 = _mm512_load_pd(C + 1 * 8 + N_pad * 5);
a0 = _mm512_load_pd(A + 0 * 8);
a1 = _mm512_load_pd(A + 1 * 8);
b0 = _mm512_set1_pd(B[0]);
b1 = _mm512_set1_pd(B[1]);
c00 = _mm512_fmadd_pd(a0, b0, c00);
c01 = _mm512_fmadd_pd(a0, b1, c01);
b0 = _mm512_set1_pd(B[2]);
b1 = _mm512_set1_pd(B[3]);
c02 = _mm512_fmadd_pd(a0, b0, c02);
c03 = _mm512_fmadd_pd(a0, b1, c03);
b0 = _mm512_set1_pd(B[4]);
b1 = _mm512_set1_pd(B[5]);
c04 = _mm512_fmadd_pd(a0, b0, c04);
c05 = _mm512_fmadd_pd(a0, b1, c05);
b0 = _mm512_set1_pd(B[0]);
b1 = _mm512_set1_pd(B[1]);
c10 = _mm512_fmadd_pd(a1, b0, c10);
c11 = _mm512_fmadd_pd(a1, b1, c11);
b0 = _mm512_set1_pd(B[2]);
b1 = _mm512_set1_pd(B[3]);
c12 = _mm512_fmadd_pd(a1, b0, c12);
c13 = _mm512_fmadd_pd(a1, b1, c13);
b0 = _mm512_set1_pd(B[4]);
b1 = _mm512_set1_pd(B[5]);
c14 = _mm512_fmadd_pd(a1, b0, c14);
c15 = _mm512_fmadd_pd(a1, b1, c15);
_mm512_store_pd(C + 0 * 8 + N_pad * 0, c00);
_mm512_store_pd(C + 1 * 8 + N_pad * 0, c10);
_mm512_store_pd(C + 0 * 8 + N_pad * 1, c01);
_mm512_store_pd(C + 1 * 8 + N_pad * 1, c11);
_mm512_store_pd(C + 0 * 8 + N_pad * 2, c02);
_mm512_store_pd(C + 1 * 8 + N_pad * 2, c12);
_mm512_store_pd(C + 0 * 8 + N_pad * 3, c03);
_mm512_store_pd(C + 1 * 8 + N_pad * 3, c13);
_mm512_store_pd(C + 0 * 8 + N_pad * 4, c04);
_mm512_store_pd(C + 1 * 8 + N_pad * 4, c14);
_mm512_store_pd(C + 0 * 8 + N_pad * 5, c05);
_mm512_store_pd(C + 1 * 8 + N_pad * 5, c15);
//[[[end]]]
    }
}

int gcd(int a, int b){
    int t;
    while(b != 0){
        t = a % b;
        a = b;
        b = t;
    }
    return a;
}

int lcm(int a, int b){
    return a / gcd(a, b) * b;
}

void square_dgemm(int N, double* A, double* B, double* C) {
    // make N_pad a multiple of 8
    /* int N_pad = ((N + CACHELINE - 1) / CACHELINE) * CACHELINE; */
    int lcm_hw = lcm(MICRO_H, MICRO_W);
    assert(lcm_hw % CACHELINE == 0);

    int N_pad = ((N + lcm_hw - 1) / lcm_hw) * lcm_hw;
    assert(N_pad % MICRO_W == 0);
    assert(N_pad % MICRO_H == 0);

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    // copy A
    cpy(N_pad, N, A, A_align);
    // transpose copy B
    transpose_cpy(N_pad, N, B, B_align);
    // copy C
    cpy(N_pad, N, C, C_align);

    for(int j = 0; j < N_pad; j += MICRO_W){
        for(int i = 0; i < N_pad; i += MICRO_H){
            // row panel
            double *A_panel = A_align + i;
            double *B_panel = B_align + j;

            double *C_block = C_align + i + j * N_pad;

            // Compute the block for C using
            // the sum of outer product of A's column and B's column(transposed)
            micro_kernel_16by6(N_pad, A_panel, B_panel, C_block);
        }
    }

    // copy C back
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
}

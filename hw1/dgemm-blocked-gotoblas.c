#include <immintrin.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

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

#define GOTOBLAS
static void gotoblas_gemm(int, double *, double *, double *);

void square_dgemm(int N, double* A, double* B, double* C) {
#ifdef GOTOBLAS
    gotoblas_gemm(N, A, B, C);
#else
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
#endif
}

#define NC 32
#define KC 32
#define MC 32
#define NR 8
#define MR 16

static void gotoblas_microkernel(
        int N_pad, double *A_panel, double *B_panel, double *C_block){
    double *C = C_block;
    for(int k = 0; k < KC; k++){
        /* double *A = A_panel + k * N_pad; */
        double *A = A_panel + k * MR;

        /* double *B = B_panel + k; */
        double *B = B_panel + k * NR;

/*[[[cog
import cog

AVX_DOUBLE_NUM = 8
NR = 8

# block C
for i in range(16 // AVX_DOUBLE_NUM):
    for j in range(8):
        cog.outl("__m512d c{}{};".format(i, j))

# column of A
for i in range(16 // AVX_DOUBLE_NUM):
    cog.outl("__m512d a{};".format(i))

# col of B
# we only have 2 registers remaining (we only have 16 registers)
step = 2
for i in range(step):
    cog.outl("__m512d b{};".format(i))

# load block C
for j in range(8):
    for i in range(16 // AVX_DOUBLE_NUM):
        cog.outl("c{i}{j} = _mm512_load_pd(C + {i} * {AVX_DOUBLE_NUM} + N_pad * {j});".format(
                i=i, j=j, AVX_DOUBLE_NUM=AVX_DOUBLE_NUM))

# load A
for i in range(16 // AVX_DOUBLE_NUM):
    cog.outl("a{i} = _mm512_load_pd(A + {i} * {AVX_DOUBLE_NUM});".format(i=i, AVX_DOUBLE_NUM=AVX_DOUBLE_NUM))

# computation!!
for i in range(16 // AVX_DOUBLE_NUM):
    for j in range(8 // step):
        # tricky part: because we only have 2 registers for B
        for k in range(step):
            bidx = k + j * step
            cog.outl("b{k} = _mm512_set1_pd(B[{bidx}]);".format(k=k, bidx=bidx))

        for k in range(step):
            bidx = k + j * step
            cog.outl("c{i}{bidx} = _mm512_fmadd_pd(a{i}, b{k}, c{i}{bidx});".format(i=i,k=k,bidx=bidx))

# write
for j in range(8):
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
__m512d c06;
__m512d c07;
__m512d c10;
__m512d c11;
__m512d c12;
__m512d c13;
__m512d c14;
__m512d c15;
__m512d c16;
__m512d c17;
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
c06 = _mm512_load_pd(C + 0 * 8 + N_pad * 6);
c16 = _mm512_load_pd(C + 1 * 8 + N_pad * 6);
c07 = _mm512_load_pd(C + 0 * 8 + N_pad * 7);
c17 = _mm512_load_pd(C + 1 * 8 + N_pad * 7);
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
b0 = _mm512_set1_pd(B[6]);
b1 = _mm512_set1_pd(B[7]);
c06 = _mm512_fmadd_pd(a0, b0, c06);
c07 = _mm512_fmadd_pd(a0, b1, c07);
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
b0 = _mm512_set1_pd(B[6]);
b1 = _mm512_set1_pd(B[7]);
c16 = _mm512_fmadd_pd(a1, b0, c16);
c17 = _mm512_fmadd_pd(a1, b1, c17);
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
_mm512_store_pd(C + 0 * 8 + N_pad * 6, c06);
_mm512_store_pd(C + 1 * 8 + N_pad * 6, c16);
_mm512_store_pd(C + 0 * 8 + N_pad * 7, c07);
_mm512_store_pd(C + 1 * 8 + N_pad * 7, c17);
//[[[end]]]
    }
}

static void gotoblas_gemm(
        int N, double *A, double *B, double *C){

    int multiple = lcm(lcm(lcm(lcm(lcm(NC, KC), MC), NR), MR), CACHELINE);
    assert(multiple % NC == 0);
    assert(multiple % KC == 0);
    assert(multiple % MC == 0);
    assert(multiple % NR == 0);
    assert(multiple % MR == 0);

    int N_pad = (N + multiple - 1) / multiple * multiple;

    double *A_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *B_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));
    double *C_align = (double *)_mm_malloc(N_pad * N_pad * sizeof(double), CACHELINE * sizeof(double));

    double *A_packed = _mm_malloc(MC * KC * sizeof(double), CACHELINE * sizeof(double));
    double *B_packed = _mm_malloc(MC * NC * sizeof(double), CACHELINE * sizeof(double));

    cpy(N_pad, N, A, A_align);
    cpy(N_pad, N, B, B_align);
    cpy(N_pad, N, C, C_align);

    for(int jc = 0; jc < N_pad; jc += NC){
        for(int pc = 0; pc < N_pad; pc += KC){

            // Pack B_align[pc:pc+KC, jc:jc+NC] column contiguous
            // but each column inside will be row contiguous

            for(int j = 0; j < NC; ++j){
                for(int i = 0; i < KC; ++i){
                    int bidx = j / NR;
                    B_packed[bidx * KC * NR + i * NR + (j % NR)] = B_align[pc + i + (jc + j) * N_pad];
                }
            }

            for(int ic = 0; ic < N_pad; ic += MC){

                // Pack A_align[ic:ic+MC, pc:pc+KC]
                // into column contiguous
                for(int j = 0; j < KC; ++j){
                    for(int i = 0; i < MC; ++i){
                        int bidx = i / MR;
                        // A_packed[bidx][i % MR][j] = A_align[ic + i + (pc + j) * N_pad];
                        A_packed[bidx * MR * KC + (i % MR) + j * MR] = A_align[ic + i + (pc + j) * N_pad];
                    }
                }

                for(int jr = 0; jr < NC; jr += NR){
                    for(int ir = 0; ir < MC; ir += MR){
                        // A_align[ic+ir:ic+ir+MR, pc:pc+KC] * 
                        // B_align[pc:pc+KC, jc+jr:jc+jr+NR]
                        // writing to
                        //
                        // C_align[ic+ir:ic+ir+MR, jc+jr:jc+jr+NR]
                        // (MR by NR)
                        //
                        /* double *microA = A_align + ic + ir + pc * N_pad; */
                        double *microA = A_packed + (ir / MR) * MR * KC; // which row panel
                        /* double *microB = B_align + pc + (jc + jr) * N_pad; */
                        double *microB = B_packed + (jr / NR) * KC * NR;
                        double *microC = C_align + ic + ir + (jc + jr) * N_pad;
                        gotoblas_microkernel(N_pad, microA, microB, microC);
                    }
                }
            }
        }
    }

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            C[i + j * N] = C_align[i + j * N_pad];
        }
    }

    _mm_free(A_align);
    _mm_free(B_align);
    _mm_free(C_align);
    _mm_free(A_packed);
}

#include <stdio.h>
#include "nn.h"
#ifdef __riscv
#include <riscv_vector.h>
#endif
#include "math.h"

// Helper function for complex multiplication
static void complex_mul(float a_real, float a_imag, float b_real, float b_imag, float* out_real, float* out_imag) {
    *out_real = a_real * b_real - a_imag * b_imag;
    *out_imag = a_real * b_imag + a_imag * b_real;
}

// Helper function for complex exponential
static void complex_exp(float a_real, float a_imag, float* out_real, float* out_imag) {
    float exp_a = my_exp(a_real);
    *out_real = exp_a * my_cos(a_imag);
    *out_imag = exp_a * my_sin(a_imag);
}

void hilbert_scan(float input[IN_CHANNELS][IMG_SIZE][IMG_SIZE], float output[SEQ_LEN][IN_CHANNELS], const int* hilbert_indices) {

//Flatten input manually and reorder
    for (int d = 0 ; d < SEQ_LEN ; d++) {
        int flat_idx = hilbert_indices[d];
        
        // SAFETY BOUNDS CHECK to prevent Segfaults if binary file formatting is weird
        if (flat_idx < 0 || flat_idx >= IMG_SIZE * IMG_SIZE) {
            flat_idx = 0; 
        }
        
        int y = flat_idx / IMG_SIZE;
        int x = flat_idx % IMG_SIZE;
        for (int c = 0 ; c < IN_CHANNELS ; c++) {
            output[d][c] = input[c][y][x];
        }
    }
}

//Generic linear layer implementation (linear uproject & fc kept for compatibility with main.c and test.c)
void linear(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int batch_size,
    int in_features,
    int out_features
) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_features; j++) {
            float acc = bias[j];
            for (int k = 0; k < in_features; k++) {
                acc += input[i * in_features + k] * weight[j * in_features + k];
            }
            output[i * out_features + j] = acc;
        }
    }
}


void linear_uproject(
    float input[SEQ_LEN][IN_CHANNELS],
    float output[SEQ_LEN][D_MODEL],
    const float weight[D_MODEL][IN_CHANNELS],
    const float bias[D_MODEL]
) {
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            output[i][j] = bias[j];
            for (int k = 0; k < IN_CHANNELS; k++) {
                output[i][j] += input[i][k] * weight[j][k];
            }
        }
    }
}


void linear_fc(
    const float input[D_MODEL],
    float output[N_CLASSES],
    const float weight[N_CLASSES][D_MODEL],
    const float bias[N_CLASSES]
) {
    for (int i = 0; i < N_CLASSES; i++) {
        output[i] = bias[i];
        for (int j = 0; j < D_MODEL; j++) {
            output[i] += input[j] * weight[i][j];
        }
    }
}


void s4d_layer(
    float input[SEQ_LEN][D_MODEL],
    float output[SEQ_LEN][D_MODEL],
    const float* log_dt,
    const float* log_A_real,
    const float* A_imag,
    const float* C_real,
    const float* C_imag,
    const float* D
) {
    int half_state = D_STATE / 2;  // 32

    for (int h = 0; h < D_MODEL; h++) {
        float dt = my_exp(log_dt[h]);

        float A_bar_real_arr[32], A_bar_imag_arr[32];
        float B_bar_real_arr[32], B_bar_imag_arr[32];
        float C_real_arr[32], C_imag_arr[32];  // hoisted C loads (loop-invariant across t)

        // Discretize once per channel, these stay constant for all 4096 timesteps.
        for (int n = 0; n < half_state; n++) {
            float lambda_real = -my_exp(log_A_real[h * half_state + n]);
            float lambda_imag = A_imag[h * half_state + n];

            // A_bar = exp(lambda * dt)
            complex_exp(lambda_real * dt, lambda_imag * dt, &A_bar_real_arr[n], &A_bar_imag_arr[n]);

            // B_bar = (A_bar - 1) / lambda. B is fixed at 1 in this parameterization,
            // so there's nothing extra to multiply in, B_bar is the whole input gain.
            float N_real = A_bar_real_arr[n] - 1.0f;
            float N_imag = A_bar_imag_arr[n];
            float denom = lambda_real * lambda_real + lambda_imag * lambda_imag;

            B_bar_real_arr[n] = (N_real * lambda_real + N_imag * lambda_imag) / denom;
            B_bar_imag_arr[n] = (N_imag * lambda_real - N_real * lambda_imag) / denom;

            // Hoist the strided C loads here (they only depend on h,n -- not t).
            C_real_arr[n] = C_real[(h * half_state + n) * 2];
            C_imag_arr[n] = C_imag[(h * half_state + n) * 2];
        }

        // State vector for this channel, zeroed at t=0.
        float x_real[32] = {0};
        float x_imag[32] = {0};

#ifdef __riscv
        {
            size_t vl = __riscv_vsetvl_e32m4(half_state);           // 32 states, one vector
            vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
            // Load per-channel CONSTANTS once; kept resident across all timesteps.
            vfloat32m4_t var = __riscv_vle32_v_f32m4(A_bar_real_arr, vl);
            vfloat32m4_t vai = __riscv_vle32_v_f32m4(A_bar_imag_arr, vl);
            vfloat32m4_t vbr = __riscv_vle32_v_f32m4(B_bar_real_arr, vl);
            vfloat32m4_t vbi = __riscv_vle32_v_f32m4(B_bar_imag_arr, vl);
            vfloat32m4_t vcr = __riscv_vle32_v_f32m4(C_real_arr, vl);
            vfloat32m4_t vci = __riscv_vle32_v_f32m4(C_imag_arr, vl);
            // Recurrent STATE stays in vector registers across the whole t-loop.
            vfloat32m4_t vxr = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vxi = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            for (int t = 0; t < SEQ_LEN; t++) {
                float u_t = input[t][h];
                vfloat32m4_t vdr = __riscv_vfmul_vv_f32m4(var, vxr, vl);
                vdr = __riscv_vfnmsac_vv_f32m4(vdr, vai, vxi, vl);   // ar*xr - ai*xi
                vfloat32m4_t vdi = __riscv_vfmul_vv_f32m4(var, vxi, vl);
                vdi = __riscv_vfmacc_vv_f32m4(vdi, vai, vxr, vl);    // ar*xi + ai*xr
                vxr = __riscv_vfmacc_vf_f32m4(vdr, u_t, vbr, vl);    // + B_r*u
                vxi = __riscv_vfmacc_vf_f32m4(vdi, u_t, vbi, vl);    // + B_i*u
                vfloat32m4_t vt = __riscv_vfmul_vv_f32m4(vcr, vxr, vl);
                vt = __riscv_vfnmsac_vv_f32m4(vt, vci, vxi, vl);     // Re(C*x)
                vfloat32m1_t vred = __riscv_vfredosum_vs_f32m4_f32m1(vt, vzero, vl);
                output[t][h] = D[h]*u_t + 2.0f*__riscv_vfmv_f_s_f32m1_f32(vred);
            }
        }
#else
        for (int t = 0; t < SEQ_LEN; t++) {
            float u_t = input[t][h];
            float y = D[h] * u_t;
            for (int n = 0; n < half_state; n++) {
                float decayed_real, decayed_imag;
                complex_mul(A_bar_real_arr[n], A_bar_imag_arr[n], x_real[n], x_imag[n], &decayed_real, &decayed_imag);
                x_real[n] = decayed_real + B_bar_real_arr[n] * u_t;
                x_imag[n] = decayed_imag + B_bar_imag_arr[n] * u_t;
                y += 2.0f * (C_real_arr[n] * x_real[n] - C_imag_arr[n] * x_imag[n]);
            }
            output[t][h] = y;
        }
#endif
    }
}

#ifdef __riscv
static inline vfloat32m4_t v_exp_m4(vfloat32m4_t x, size_t vl){
    x = __riscv_vfmax_vf_f32m4(x, -88.0f, vl);
    x = __riscv_vfmin_vf_f32m4(x,  88.0f, vl);
    vfloat32m4_t t = __riscv_vfmul_vf_f32m4(x, 1.44269502f, vl);        // x/ln2
    vint32m4_t   n = __riscv_vfcvt_x_f_v_i32m4(t, vl);                  // round to nearest
    vfloat32m4_t nf = __riscv_vfcvt_f_x_v_f32m4(n, vl);
    vfloat32m4_t r = __riscv_vfnmsac_vf_f32m4(x, 0.693145752f, nf, vl); // x - nf*ln2_hi
    r = __riscv_vfnmsac_vf_f32m4(r, 1.42860677e-06f, nf, vl);          // - nf*ln2_lo
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(0.00139560562f, vl);       // c6
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 0.00837512873f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 0.041666083f,  vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 0.166664153f,  vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 0.5f,          vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 1.0f,          vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 1.0f,          vl);
    vint32m4_t e = __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(n, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m4(p, __riscv_vreinterpret_v_i32m4_f32m4(e), vl);
}
static inline vfloat32m4_t v_tanh_m4(vfloat32m4_t x, size_t vl){
    vfloat32m4_t e2x = v_exp_m4(__riscv_vfadd_vv_f32m4(x, x, vl), vl);
    vfloat32m4_t d = __riscv_vfadd_vf_f32m4(e2x, 1.0f, vl);
    return __riscv_vfrsub_vf_f32m4(__riscv_vfrdiv_vf_f32m4(d, 2.0f, vl), 1.0f, vl); // 1 - 2/(e2x+1)
}
#endif

void gelu(float* x, int size) {
    const float pi = 3.141592653589793f;
    const float k = my_sqrt(2.0f / pi);
    const float coeff = 0.044715f;

#ifdef __riscv
    size_t vl;
    for (int i = 0; i < size; i += (int)vl) {
        vl = __riscv_vsetvl_e32m4(size - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(&x[i], vl);
        vfloat32m4_t vx3 = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(vx, vx, vl), vx, vl);
        vfloat32m4_t inner = __riscv_vfmul_vf_f32m4(vx3, coeff, vl);       // coeff*x^3
        inner = __riscv_vfadd_vv_f32m4(inner, vx, vl);                    // x + coeff*x^3
        inner = __riscv_vfmul_vf_f32m4(inner, k, vl);                     // * k
        vfloat32m4_t vt = v_tanh_m4(inner, vl);
        vfloat32m4_t r = __riscv_vfmul_vv_f32m4(__riscv_vfadd_vf_f32m4(vt, 1.0f, vl), vx, vl);
        r = __riscv_vfmul_vf_f32m4(r, 0.5f, vl);                          // 0.5*x*(1+tanh)
        __riscv_vse32_v_f32m4(&x[i], r, vl);
    }
#else
    for (int i = 0; i < size; i++) {
        float x_cubed = x[i] * x[i] * x[i];
        float inner = k * (x[i] + coeff * x_cubed);
        float tanh_val = my_tanh(inner);
        x[i] = 0.5f * x[i] * (1.0f + tanh_val);
    }
#endif
}


void softmax(float* logits, int size) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        logits[i] = my_exp(logits[i] - max_val);
        sum += logits[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        logits[i] /= sum;
    }
}

void take_last_timestamp(float input[SEQ_LEN][D_MODEL], float output[D_MODEL]) {
    for (int j = 0; j < D_MODEL; j++) {
        output[j] = input[SEQ_LEN - 1][j];
    }
}

#include <stdint.h>
static inline uint64_t rdinstret(void){
#ifdef __riscv
    uint32_t lo, hi, hi2;
    do { __asm__ volatile("csrr %0, instreth":"=r"(hi));
         __asm__ volatile("csrr %0, instret" :"=r"(lo));
         __asm__ volatile("csrr %0, instreth":"=r"(hi2)); } while (hi!=hi2);
    return ((uint64_t)hi<<32)|lo;
#else
    return 0;
#endif
}
static uint64_t _ir_prev;
#define LMARK(n) do{ uint64_t _x=rdinstret(); printf("LAYERCOUNT %s %llu\n", n, (unsigned long long)(_x-_ir_prev)); _ir_prev=rdinstret(); }while(0)

void model_forward(
    float image[IN_CHANNELS][IMG_SIZE][IMG_SIZE],
    float probabilities[N_CLASSES],
    const float* model_weights,     // This points to the loaded .bin file
    const int* hilbert_indices       // First 4096 integers from weights
) {
    // Calculate offsets (in bytes, but we'll use pointer arithmetic)
    int offset = 0;
    // 1. Hilbert indices (already passed separately)
    // const int* hilbert_indices = (const int*)model_weights;
    offset += 4096 * sizeof(int);

    // 2. uproject.weight (64 × 1 floats)
    const float* uproject_weight = (const float*)((const char*)model_weights + offset);
    offset += 64 * 1 * sizeof(float);

    // 3. uproject.bias (64 floats)
    const float* uproject_bias = (const float*)((const char*)model_weights + offset);
    offset += 64 * sizeof(float);

    // 4. s4_1.log_dt (64 floats)
    const float* s4d1_log_dt = (const float*)((const char*)model_weights + offset);
    offset += 64 * sizeof(float);

    // 5. s4_1.log_A_real (64 × 32 = 2048 floats)
    const float* s4d1_log_A_real = (const float*)((const char*)model_weights + offset);
    offset += 64 * 32 * sizeof(float);

    // 6. s4_1.A_imag (64 × 32 = 2048 floats)
    const float* s4d1_A_imag = (const float*)((const char*)model_weights + offset);
    offset += 64 * 32 * sizeof(float);

    // 7. s4_1.C (64 × 32 × 2 = 4096 floats) - real and imag interleaved
    const float* s4d1_C_real = (const float*)((const char*)model_weights + offset);
    const float* s4d1_C_imag = s4d1_C_real + 1;  // But careful! Need proper indexing
    offset += 64 * 32 * 2 * sizeof(float);

    // 8. s4_1.D (64 floats)
    const float* s4d1_D = (const float*)((const char*)model_weights + offset);
    offset += 64 * sizeof(float);

    // 9. s4_2.log_dt (64 floats)
    const float* s4d2_log_dt = (const float*)((const char*)model_weights + offset);
    offset += 64 * sizeof(float);

    // 10. s4_2.log_A_real (64 × 32 = 2048 floats)
    const float* s4d2_log_A_real = (const float*)((const char*)model_weights + offset);
    offset += 64 * 32 * sizeof(float);

    // 11. s4_2.A_imag (64 × 32 = 2048 floats)
    const float* s4d2_A_imag = (const float*)((const char*)model_weights + offset);
    offset += 64 * 32 * sizeof(float);

    // 12. s4_2.C (64 × 32 × 2 = 4096 floats)
    const float* s4d2_C_real = (const float*)((const char*)model_weights + offset);
    const float* s4d2_C_imag = s4d2_C_real + 1;
    offset += 64 * 32 * 2 * sizeof(float);

    // 13. s4_2.D (64 floats)
    const float* s4d2_D = (const float*)((const char*)model_weights + offset);
    offset += 64 * sizeof(float);

    // 14. fc.weight (4 × 64 = 256 floats)
    const float* fc_weight = (const float*)((const char*)model_weights + offset);
    offset += 4 * 64 * sizeof(float);

    // 15. fc.bias (4 floats)
    const float* fc_bias = (const float*)((const char*)model_weights + offset);

    // now we use all the weights in forward pass

    // Buffers
    static float hilbert_out[SEQ_LEN][IN_CHANNELS];
    static float proj_out[SEQ_LEN][D_MODEL];
    static float s4d1_out[SEQ_LEN][D_MODEL];
    //static float after_gelu1[SEQ_LEN][D_MODEL];
    static  float s4d2_out[SEQ_LEN][D_MODEL];
    //static float after_gelu2[SEQ_LEN][D_MODEL];
    static float pooled[D_MODEL];
    static float logits[N_CLASSES];

    // 1. Hilbert Scan
    _ir_prev=rdinstret();
    hilbert_scan(image, hilbert_out, hilbert_indices);
    LMARK("hilbert");

    // 2. Input Projection
    linear_uproject(hilbert_out, proj_out, 
                    (const float(*)[IN_CHANNELS])uproject_weight,
                    uproject_bias);
    LMARK("input_proj");

    // 3. S4D Layer 1
    s4d_layer(proj_out, s4d1_out,
              s4d1_log_dt, s4d1_log_A_real, s4d1_A_imag,
              s4d1_C_real, s4d1_C_imag, s4d1_D);
    LMARK("s4_1");

    // 4. GELU
    gelu(&s4d1_out[0][0], SEQ_LEN * D_MODEL);
    LMARK("gelu_1");

    // 5. S4D Layer 2
    s4d_layer(s4d1_out, s4d2_out,
              s4d2_log_dt, s4d2_log_A_real, s4d2_A_imag,
              s4d2_C_real, s4d2_C_imag, s4d2_D);
    LMARK("s4_2");

    // 6. GELU
    gelu(&s4d2_out[0][0], SEQ_LEN * D_MODEL);
    LMARK("gelu_2");

    // 7. Take Last Timestamp
    take_last_timestamp(s4d2_out, pooled);
    LMARK("ttls");

    // 8. Final Classification
    linear_fc(pooled, logits,
              (const float(*)[D_MODEL])fc_weight,
              fc_bias);
    LMARK("output_proj");

    // 9. Softmax
    for (int i = 0; i < N_CLASSES; i++) {
        probabilities[i] = logits[i];
    }
    softmax(probabilities, N_CLASSES);
    LMARK("softmax");
}

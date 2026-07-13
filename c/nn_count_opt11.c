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

    // Constants laid out TRANSPOSED as [state][channel] so a fixed state n is a
    // contiguous vector across all 64 channels. Lets us vectorize the batch (channel)
    // dimension and turn the per-timestep reduction into a vector FMA accumulation.
    static float AR[32][D_MODEL], AI[32][D_MODEL];
    static float BR[32][D_MODEL], BI[32][D_MODEL];
    static float CR[32][D_MODEL], CI[32][D_MODEL];

    for (int h = 0; h < D_MODEL; h++) {
        float dt = my_exp(log_dt[h]);
        for (int n = 0; n < half_state; n++) {
            float lr = -my_exp(log_A_real[h * half_state + n]);
            float li = A_imag[h * half_state + n];
            float ar, ai;
            complex_exp(lr * dt, li * dt, &ar, &ai);
            AR[n][h] = ar; AI[n][h] = ai;
            float Nr = ar - 1.0f, Ni = ai;
            float denom = lr * lr + li * li;
            BR[n][h] = (Nr * lr + Ni * li) / denom;
            BI[n][h] = (Ni * lr - Nr * li) / denom;
            CR[n][h] = C_real[(h * half_state + n) * 2];
            CI[n][h] = C_imag[(h * half_state + n) * 2];
        }
    }

    // Per-channel complex state, zeroed. Column h is channel h's 32-state vector.
    static float XR[32][D_MODEL], XI[32][D_MODEL];
    for (int n = 0; n < half_state; n++)
        for (int h = 0; h < D_MODEL; h++) { XR[n][h] = 0.0f; XI[n][h] = 0.0f; }

#ifdef __riscv
    // Vectorize over CHANNELS. For each timestep, update all 64 channels' states in
    // parallel and accumulate y with vfmacc -- no vfredosum anywhere in the scan.
    for (int t = 0; t < SEQ_LEN; t++) {
        for (int h0 = 0; h0 < D_MODEL; ) {
            size_t vl = __riscv_vsetvl_e32m4(D_MODEL - h0);
            vfloat32m4_t vu = __riscv_vle32_v_f32m4(&input[t][h0], vl);
            vfloat32m4_t vy = __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(&D[h0], vl), vu, vl);
            for (int n = 0; n < half_state; n++) {
                vfloat32m4_t vxr = __riscv_vle32_v_f32m4(&XR[n][h0], vl);
                vfloat32m4_t vxi = __riscv_vle32_v_f32m4(&XI[n][h0], vl);
                vfloat32m4_t var = __riscv_vle32_v_f32m4(&AR[n][h0], vl);
                vfloat32m4_t vai = __riscv_vle32_v_f32m4(&AI[n][h0], vl);
                vfloat32m4_t vbr = __riscv_vle32_v_f32m4(&BR[n][h0], vl);
                vfloat32m4_t vbi = __riscv_vle32_v_f32m4(&BI[n][h0], vl);
                vfloat32m4_t vcr = __riscv_vle32_v_f32m4(&CR[n][h0], vl);
                vfloat32m4_t vci = __riscv_vle32_v_f32m4(&CI[n][h0], vl);
                // x' = A*x + B*u   (complex, u real)
                vfloat32m4_t nr = __riscv_vfmul_vv_f32m4(var, vxr, vl);
                nr = __riscv_vfnmsac_vv_f32m4(nr, vai, vxi, vl);
                nr = __riscv_vfmacc_vv_f32m4(nr, vbr, vu, vl);
                vfloat32m4_t ni = __riscv_vfmul_vv_f32m4(var, vxi, vl);
                ni = __riscv_vfmacc_vv_f32m4(ni, vai, vxr, vl);
                ni = __riscv_vfmacc_vv_f32m4(ni, vbi, vu, vl);
                __riscv_vse32_v_f32m4(&XR[n][h0], nr, vl);
                __riscv_vse32_v_f32m4(&XI[n][h0], ni, vl);
                // y += 2*Re(C*x')  -- accumulated across states, NO reduction
                vfloat32m4_t vt = __riscv_vfmul_vv_f32m4(vcr, nr, vl);
                vt = __riscv_vfnmsac_vv_f32m4(vt, vci, ni, vl);
                vy = __riscv_vfmacc_vf_f32m4(vy, 2.0f, vt, vl);
            }
            __riscv_vse32_v_f32m4(&output[t][h0], vy, vl);
            h0 += (int)vl;
        }
    }
#else
    for (int h = 0; h < D_MODEL; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            float u_t = input[t][h];
            float y = D[h] * u_t;
            for (int n = 0; n < half_state; n++) {
                float dr, di;
                complex_mul(AR[n][h], AI[n][h], XR[n][h], XI[n][h], &dr, &di);
                XR[n][h] = dr + BR[n][h] * u_t;
                XI[n][h] = di + BI[n][h] * u_t;
                y += 2.0f * (CR[n][h] * XR[n][h] - CI[n][h] * XI[n][h]);
            }
            output[t][h] = y;
        }
    }
#endif
}

void gelu(float* x, int size) {
    const float pi = 3.141592653589793f;
    const float k = my_sqrt(2.0f / pi);
    const float coeff = 0.044715f;

    for (int i = 0; i < size; i++) {
        float x_cubed = x[i] * x[i] * x[i];
        float inner = k * (x[i] + coeff * x_cubed);
        float tanh_val = my_tanh(inner);
        x[i] = 0.5f * x[i] * (1.0f + tanh_val);
    }
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

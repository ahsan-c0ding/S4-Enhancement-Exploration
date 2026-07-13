#include <stdio.h>
#include "nn.h"
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
        /* progress print removed for clean counting */
        /* fflush removed */ 
        
        float dt = my_exp(log_dt[h]);
        float kernel[SEQ_LEN];
        
        // Arrays to hold our properly discretized C_bar
        float C_bar_real_arr[32];
        float C_bar_imag_arr[32];
        float lambda_real_arr[32];
        float lambda_imag_arr[32];

        // 1. Discretize C -> C_bar using Complex Division
        for (int n = 0; n < half_state; n++) {
            float lambda_real = -my_exp(log_A_real[h * half_state + n]);
            float lambda_imag = A_imag[h * half_state + n];
            lambda_real_arr[n] = lambda_real;
            lambda_imag_arr[n] = lambda_imag;

            // Compute A_bar
            float A_bar_real, A_bar_imag;
            complex_exp(lambda_real * dt, lambda_imag * dt, &A_bar_real, &A_bar_imag);

            // Compute Complex Division: Z = (A_bar - 1) / lambda
            float N_real = A_bar_real - 1.0f;
            float N_imag = A_bar_imag;
            float denom = lambda_real * lambda_real + lambda_imag * lambda_imag;
            
            float Z_real = (N_real * lambda_real + N_imag * lambda_imag) / denom;
            float Z_imag = (N_imag * lambda_real - N_real * lambda_imag) / denom;

            // Compute C_bar = C * Z
            float C_real_val = C_real[(h * half_state + n) * 2];
            float C_imag_val = C_imag[(h * half_state + n) * 2];

            complex_mul(C_real_val, C_imag_val, Z_real, Z_imag, &C_bar_real_arr[n], &C_bar_imag_arr[n]);
        }

        // 2. Generate kernel using Direct Exponential Scaling
        for (int t = 0; t < SEQ_LEN; t++) {
            kernel[t] = 0.0f;
            
            for (int n = 0; n < half_state; n++) {
                float t_dt = t * dt; 
                
                // Compute A_bar^t directly: exp(t * dt * lambda)
                float A_bar_t_real, A_bar_t_imag;
                complex_exp(t_dt * lambda_real_arr[n], t_dt * lambda_imag_arr[n], &A_bar_t_real, &A_bar_t_imag);

                // Multiply C_bar * A_bar^t
                float term_real, term_imag;
                complex_mul(C_bar_real_arr[n], C_bar_imag_arr[n], A_bar_t_real, A_bar_t_imag, &term_real, &term_imag);

                // Add to kernel
                kernel[t] += 2.0f * term_real;
            }
        }

        // 3. Causal convolution
        for (int k = 0; k < SEQ_LEN; k++) {
            output[k][h] = D[h] * input[k][h];
            for (int j = 0; j <= k; j++) {
                output[k][h] += kernel[j] * input[k - j][h];
            }
        }
    }
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

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
        
        // safety check to prevent Segfaults if binary file formatting is weird
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


// Diagonal S4D layer Recurrent Implementation
//
// The previous implementation built an explicit kernel and did an O(L^2) causal
// convolution. This one discretizes A and B once per channel (O(N)), then
// walks the sequence with a simple state-update loop (O(L*N))
//
// The math is the same ZOH discretization from our report,just applied 
// element-wise since A is diagonal, so no matrix-exp is needed.
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
        }

        // State vector for this channel, zeroed at t=0.
        float x_real[32] = {0};
        float x_imag[32] = {0};

        for (int t = 0; t < SEQ_LEN; t++) {
            float u_t = input[t][h];
            float y = D[h] * u_t;  // feedthrough term, same as before

            for (int n = 0; n < half_state; n++) {
                // x_t = A_bar * x_{t-1} + B_bar * u_t
                float decayed_real, decayed_imag;
                complex_mul(A_bar_real_arr[n], A_bar_imag_arr[n], x_real[n], x_imag[n], &decayed_real, &decayed_imag);

                x_real[n] = decayed_real + B_bar_real_arr[n] * u_t;
                x_imag[n] = decayed_imag + B_bar_imag_arr[n] * u_t;

                // y_t += 2*Re(C * x_t). The factor of 2 accounts for the conjugate
                // pair we never explicitly store, we only keep n//2 modes.
                float term_real, term_imag;
                float C_real_val = C_real[(h * half_state + n) * 2];
                float C_imag_val = C_imag[(h * half_state + n) * 2];
                complex_mul(C_real_val, C_imag_val, x_real[n], x_imag[n], &term_real, &term_imag);
                y += 2.0f * term_real;
            }

            output[t][h] = y;
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
    hilbert_scan(image, hilbert_out, hilbert_indices);

    // 2. Input Projection
    linear((float*)hilbert_out, (float*)proj_out,
           uproject_weight, uproject_bias,
           SEQ_LEN, IN_CHANNELS, D_MODEL);

    // 3. S4D Layer 1
    s4d_layer(proj_out, s4d1_out,
              s4d1_log_dt, s4d1_log_A_real, s4d1_A_imag,
              s4d1_C_real, s4d1_C_imag, s4d1_D);

    // 4. GELU
    gelu(&s4d1_out[0][0], SEQ_LEN * D_MODEL);

    // 5. S4D Layer 2
    s4d_layer(s4d1_out, s4d2_out,
              s4d2_log_dt, s4d2_log_A_real, s4d2_A_imag,
              s4d2_C_real, s4d2_C_imag, s4d2_D);

    // 6. GELU
    gelu(&s4d2_out[0][0], SEQ_LEN * D_MODEL);

    // 7. Take Last Timestamp
    take_last_timestamp(s4d2_out, pooled);

    // 8. Final Classification
    linear(pooled, logits,
           fc_weight, fc_bias,
           1, D_MODEL, N_CLASSES);

    // 9. Softmax
    for (int i = 0; i < N_CLASSES; i++) {
        probabilities[i] = logits[i];
    }
    softmax(probabilities, N_CLASSES);
}

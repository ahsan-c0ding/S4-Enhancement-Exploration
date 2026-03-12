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

// Input Projection Layer: (4096, C) -> (4096, 64)
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

// Final Classification Layer: (64) -> (4)
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
        printf("\rProcessing S4D channel %d out of 64...", h + 1);
        fflush(stdout); 
        
        float dt = my_exp(log_dt[h]);
        float kernel[SEQ_LEN];
        
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

            float A_bar_real, A_bar_imag;
            complex_exp(lambda_real * dt, lambda_imag * dt, &A_bar_real, &A_bar_imag);

            float N_real = A_bar_real - 1.0f;
            float N_imag = A_bar_imag;
            float denom = lambda_real * lambda_real + lambda_imag * lambda_imag;
            
            float Z_real = (N_real * lambda_real + N_imag * lambda_imag) / denom;
            float Z_imag = (N_imag * lambda_real - N_real * lambda_imag) / denom;

            float C_real_val = C_real[(h * half_state + n) * 2];
            float C_imag_val = C_imag[(h * half_state + n) * 2];

            complex_mul(C_real_val, C_imag_val, Z_real, Z_imag, &C_bar_real_arr[n], &C_bar_imag_arr[n]);
        }

        // 2. Generate kernel using Direct Exponential Scaling
        for (int t = 0; t < SEQ_LEN; t++) {
            kernel[t] = 0.0f;
            
            for (int n = 0; n < half_state; n++) {
                float t_dt = t * dt; 
                
                float A_bar_t_real, A_bar_t_imag;
                complex_exp(t_dt * lambda_real_arr[n], t_dt * lambda_imag_arr[n], &A_bar_t_real, &A_bar_t_imag);

                float term_real, term_imag;
                complex_mul(C_bar_real_arr[n], C_bar_imag_arr[n], A_bar_t_real, A_bar_t_imag, &term_real, &term_imag);

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
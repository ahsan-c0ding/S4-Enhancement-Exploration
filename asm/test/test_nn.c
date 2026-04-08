#include <stdio.h>

extern void complex_mul(float ar, float ai, float br, float bi, float* out_r, float* out_i);
extern void hilbert_scan(float input[1][64][64], float output[4096][1], const int* indices);
extern void take_last_timestamp(float input[4096][64], float output[64]);
extern void linear(const float* input, float* output, const float* weight, const float* bias,
                   int batch_size, int in_features, int out_features);
extern void gelu(float* x, int size);
extern void softmax(float* logits, int size);

int main() {

    // --- linear only, nothing else ---
    float lin_input[]  = {1.0f, 2.0f};
    float lin_weight[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float lin_bias[]   = {0.0f, 1.0f};
    float lin_output[2];

    // Step 1: batch_size=0, should exit immediately
    //linear(lin_input, lin_output, lin_weight, lin_bias, 0, 2, 2);
    return 103;
}
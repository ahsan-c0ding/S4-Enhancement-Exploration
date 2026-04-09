#include <stdio.h>

extern void complex_mul(float ar, float ai, float br, float bi, float* out_r, float* out_i);
extern void hilbert_scan(float input[1][64][64], float output[4096][1], const int* indices);
extern void take_last_timestamp(float input[4096][64], float output[64]);
extern void linear(const float* input, float* output, const float* weight, const float* bias,
                   int batch_size, int in_features, int out_features);
extern void gelu(float* x, int size);
extern void softmax(float* logits, int size);
extern void complex_exp(float a_real, float a_imag, float* out_real, float* out_imag);

int main() {

    // --- linear ---
    float lin_input[]  = {1.0f, 2.0f};
    float lin_weight[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float lin_bias[]   = {0.0f, 1.0f};
    float lin_output[2];

    printf("Testing linear batch_size=0...\n");
    linear(lin_input, lin_output, lin_weight, lin_bias, 0, 2, 2);
    printf("  returned OK\n");

    printf("Testing linear batch_size=1...\n");
    linear(lin_input, lin_output, lin_weight, lin_bias, 1, 2, 2);
    printf("  returned OK\n");
    printf("  output[0] = %f (expected 1.0)\n", lin_output[0]);
    printf("  output[1] = %f (expected 3.0)\n", lin_output[1]);

    float lin_input2[]  = {1.0f, 2.0f, 3.0f};
    float lin_weight2[] = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f};
    float lin_bias2[]   = {0.0f, 0.0f};
    float lin_output2[2];

    printf("Testing linear 1x3->1x2...\n");
    linear(lin_input2, lin_output2, lin_weight2, lin_bias2, 1, 3, 2);
    printf("  returned OK\n");
    printf("  output[0] = %f (expected 6.0)\n", lin_output2[0]);
    printf("  output[1] = %f (expected 14.0)\n", lin_output2[1]);

    printf("All linear tests done.\n");

       // --- gelu ---
    printf("Testing gelu...\n");

    float gelu_x[3] = {0.0f, 1.0f, -1.0f};
    gelu(gelu_x, 3);
    printf("  gelu(0)  = %f (expected 0.0)\n",    gelu_x[0]);
    printf("  gelu(1)  = %f (expected 0.8413)\n", gelu_x[1]);
    printf("  gelu(-1) = %f (expected -0.1587)\n", gelu_x[2]);

    float gelu_y[1] = {5.0f};
    gelu(gelu_y, 1);
    printf("  gelu(5)  = %f (expected ~5.0)\n", gelu_y[0]);

    float gelu_z[1] = {-5.0f};
    gelu(gelu_z, 1);
    printf("  gelu(-5) = %f (expected ~0.0)\n", gelu_z[0]);

    printf("gelu done.\n\n");

    // --- softmax ---
    printf("Testing softmax...\n");

    // All equal -> each should be 0.25
    float sm1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    softmax(sm1, 4);
    printf("  softmax([0,0,0,0]):\n");
    printf("    [0] = %f (expected 0.25)\n", sm1[0]);
    printf("    [1] = %f (expected 0.25)\n", sm1[1]);
    printf("    [2] = %f (expected 0.25)\n", sm1[2]);
    printf("    [3] = %f (expected 0.25)\n", sm1[3]);

    // One dominant -> first near 1.0, rest near 0.0
    float sm2[4] = {10.0f, 0.0f, 0.0f, 0.0f};
    softmax(sm2, 4);
    printf("  softmax([10,0,0,0]):\n");
    printf("    [0] = %f (expected ~1.0)\n",  sm2[0]);
    printf("    [1] = %f (expected ~0.0)\n",  sm2[1]);
    printf("    [2] = %f (expected ~0.0)\n",  sm2[2]);
    printf("    [3] = %f (expected ~0.0)\n",  sm2[3]);

    // Ascending -> probabilities should be strictly increasing
    float sm3[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    softmax(sm3, 4);
    printf("  softmax([1,2,3,4]):\n");
    printf("    [0] = %f (expected 0.0321)\n", sm3[0]);
    printf("    [1] = %f (expected 0.0871)\n", sm3[1]);
    printf("    [2] = %f (expected 0.2369)\n", sm3[2]);
    printf("    [3] = %f (expected 0.6439)\n", sm3[3]);
    float sum = sm3[0] + sm3[1] + sm3[2] + sm3[3];
    printf("    sum  = %f (expected 1.0)\n",   sum);

    printf("softmax done.\n");
    printf("Testing complex_exp...\n");

float ce_r, ce_i;

// exp(0 + 0i) = 1 + 0i
complex_exp(0.0f, 0.0f, &ce_r, &ce_i);
printf("  exp(0+0i) = %f + %fi (expected 1.0 + 0.0i)\n", ce_r, ce_i);

// exp(0 + pi*i) = -1 + 0i  (Euler's formula)
complex_exp(0.0f, 3.14159265f, &ce_r, &ce_i);
printf("  exp(0+pi*i) = %f + %fi (expected -1.0 + 0.0i)\n", ce_r, ce_i);

// exp(1 + 0i) = e + 0i
complex_exp(1.0f, 0.0f, &ce_r, &ce_i);
printf("  exp(1+0i) = %f + %fi (expected 2.7182 + 0.0i)\n", ce_r, ce_i);

// exp(0 + pi/2 * i) = 0 + 1i
complex_exp(0.0f, 1.5707963f, &ce_r, &ce_i);
printf("  exp(0+pi/2*i) = %f + %fi (expected 0.0 + 1.0i)\n", ce_r, ce_i);

printf("complex_exp done.\n\n");
    return 0;
}

#include <stdio.h>

extern float my_exp(float x);
extern float my_sin(float x);
extern float my_cos(float x);
extern float my_tanh(float x);
extern float my_sqrt(float x);
extern float my_pow(float x, float y);
extern float my_log(float x);

int main() {
    float x = 1.0f;

    if ((int)my_exp(x) != 2) return 1;                         // e^1 ≈ 2.718
    if ((int)my_exp(2.0f) != 7) return 10;                     // e^2 ≈ 7.389
    if ((int)(my_exp(-1.0f) * 10000) != 3678) return 11;       // e^-1 ≈ 0.3678
    if ((int)(my_exp(0.5f) * 1000) != 1648) return 12;         // e^0.5 ≈ 1.6487
    if ((int)my_exp(0.0f) != 1) return 13;                     // e^0 = 1.0

    if ((int)(my_sin(x) * 100) != 84) return 2;                // sin(1) ≈ 0.8414
    if ((int)(my_sin(0.0f) * 100) != 0) return 20;             // sin(0) = 0
    if ((int)(my_sin(0.5f) * 10000) != 4794) return 21;        // sin(0.5) ≈ 0.4794
    if ((int)(my_sin(2.0f) * 1000) != 909) return 22;          // sin(2) ≈ 0.9093
    if ((int)(my_sin(-1.0f) * 100) != -84) return 23;          // sin(-x) = -sin(x)

    if ((int)(my_cos(x) * 100) != 54) return 3;                // cos(1) ≈ 0.5403
    if ((int)(my_cos(0.5f) * 1000) != 877) return 30;          // cos(0.5) ≈ 0.8775
    if ((int)(my_cos(2.0f) * 1000) != -416) return 31;         // cos(2) ≈ -0.4161
    if ((int)(my_cos(-1.0f) * 100) != 54) return 32;           // cos(-x) = cos(x)

    if ((int)(my_tanh(x) * 100) != 76) return 4;               // tanh(1) ≈ 0.7615
    if ((int)(my_tanh(0.0f) * 100) != 0) return 40;            // tanh(0) = 0
    if ((int)(my_tanh(0.5f) * 1000) != 462) return 41;         // tanh(0.5) ≈ 0.4621
    if ((int)(my_tanh(2.0f) * 100) != 96) return 42;           // tanh(2) ≈ 0.9640
    if ((int)(my_tanh(-1.0f) * 100) != -76) return 43;         // tanh(-x) = -tanh(x)
    if (my_tanh(15.0f) <= 0.99f) return 44;                    // large x clamps to ~1

    if ((int)my_sqrt(4.0f) != 2) return 5;                     // sqrt(4) = 2
    if ((int)my_sqrt(9.0f) != 3) return 50;                    // sqrt(9) = 3
    if ((int)my_sqrt(100.0f) != 10) return 51;                 // sqrt(100) = 10
    if ((int)(my_sqrt(2.0f) * 10000) != 14142) return 52;      // sqrt(2) ≈ 1.41421
    if ((int)my_sqrt(0.0f) != 0) return 53;                    // sqrt(0) = 0

    if ((int)(my_log(1.0f) * 100) != 0) return 6;              // ln(1) = 0
    if ((int)(my_log(2.0f) * 10000) != 6931) return 7;         // ln(2) ≈ 0.6931
    if ((int)(my_log(10.0f) * 1000) != 2302) return 8;         // ln(10) ≈ 2.302
    if (my_log(-1.0f) > 0.0f) return 9;                        // ln(x<=0) = -inf

    if ((int)(my_pow(2.0f, 0.5f) * 10000) != 14142) return 60; // 2^0.5 = sqrt(2)
    if ((int)(my_pow(10.0f, 0.5f) * 1000) != 3162) return 61;  // 10^0.5 ≈ 3.1622
    if ((int)(my_pow(10.0f, 0.3f) * 100) != 199) return 62;    // 10^0.3 ≈ 1.9952
    if (my_pow(-1.0f, 2.0f) != 0.0f) return 63;                // x<=0 returns 0

    return 0;
}
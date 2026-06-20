#include "math.h"

// e^x calculated using taylor series
float my_exp(float x){
    // Prevent catastrophic underflow/overflow bounds
    if(x < -80.0f) return 0.0f;
    if(x > 80.0f) return 5.5406e+34f; // Max float limit approx

    if(x < 0){
        return 1.0f / my_exp(-x);
    }
    
    float result = 1.0f;
    float term = 1.0f;

    // INCREASED TO 50 FOR LARGE NUMBER CONVERGENCE
    for(int i = 1; i <= 50; i++){
        term = term * x / i;
        result = result + term;

        if(term < 0.0000001f && term > -0.0000001f){
            break;
        }
    }
    return result;
}

// log(x) calculated using newtons method
float my_log(float x){
    if(x <= 0){
        return -1e10f;
    }
    float y = x - 1.0f;

    for(int i = 0; i < 20; i++){
        float exp_y = my_exp(y);
        y = y - (exp_y - x) / exp_y;

        float difference = my_exp(y) - x;
        if(difference < 0.0000001f && difference > -0.0000001f){
            break;
        }
    }
    return y;
}

// sin(x) calculated using taylor series
float my_sin(float x) {
    float pi = 3.141592653589793f;
    
    int quotients = (int)(x / (2 * pi));
    x = x - (quotients * 2 * pi);
    if (x > pi) x -= 2 * pi;
    if (x < -pi) x += 2 * pi;

    float result = 0;
    float term = x;
    float x_sq = x * x;

    // INCREASED TO 50
    for (int i = 1; i <= 50; i += 2) {
        result = result + term;
        term = -term * x_sq / ((i+1) * (i+2));

        if (term < 0.0000001f && term > -0.0000001f) {
            break;
        }
    }
    return result;
}

// cos(x) calculated using taylor series
float my_cos(float x) {
    float pi = 3.141592653589793f;
    
    int quotients = (int)(x / (2 * pi));
    x = x - (quotients * 2 * pi);
    if (x > pi) x -= 2 * pi;
    if (x < -pi) x += 2 * pi;

    float result = 0;
    float term = 1.0f;
    float x_sq = x * x;

    // INCREASED TO 50
    for (int i = 0; i <= 50; i += 2) {
        result = result + term;
        term = -term * x_sq / ((i+1) * (i+2));

        if (term < 0.0000001f && term > -0.0000001f) {
            break;
        }
    }
    return result;
}

// tanh(x)
float my_tanh(float x){
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;

    float expPos= my_exp(x);
    float expNeg = my_exp(-x);

    return (expPos - expNeg) / (expPos + expNeg);
}

// pow(x) for only integers
float my_pow_int(float x, int y){
    float result = 1.0f;
    if (y < 0) {
        x = 1.0f / x;
        y = -y;
    }
    for (int i = 0; i < y; i++) {
        result *= x;
    }
    return result;
}

// pow for general exponentiation
float my_pow(float x, float y){
    if(x <= 0) return 0;
    return my_exp(y * my_log(x));
}

// HIGH-PRECISION SQRT (Babylonian Method)
float my_sqrt(float x){
    if(x <= 0.0f) return 0.0f;
    float res = x;
    // 10 iterations of Newton-Raphson is enough to max out 32-bit float precision
    for(int i = 0; i < 10; i++) {
        res = 0.5f * (res + x / res);
    }
    return res;
}
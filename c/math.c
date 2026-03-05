#include "math.h"

//e^x calculated using taylor series  e^x = 1 + x + x^2/2! + x^3/3! + x^4/4! + ...
float my_exp(float x){
   if(x < 0){
	return 1.0f / my_exp(-x);
   }
   float result = 1.0f;
   float term = 1.0f;

   for(int i = 1; i <= 15; i++){
	term = term * x / i;
	result = result + term;

	if(term < 0.000001f && term > -0.000001f){
		break;
	}
   }
   return result;
}

//log(x) calculated using newtons method
float my_log(float x){
   if(x <= 0){
	return -1e10f;
   }
   float y = x - 1.0f;

   for(int i = 0; i < 20; i++){
	float exp_y = my_exp(y);
	y = y - (exp_y - x) / exp_y;

	float difference = my_exp(y) - x;
	if(difference < 0.000001f && diff > -0.000001f){
		break;
	}
   }
   return y;

}

//sin(x) is calculated using taylor series sin x = x - x^3/3! + x^5/5! - x^7/7! + ...
float my_sin(float x) {
    float pi = 3.141592653589793f;
    while (x > pi) x -= 2 * pi;
    while (x < -pi) x += 2 * pi;

    float result = 0;
    float term = x;
    float x_sq = x * x;

    for (int i = 1; i <= 15; i += 2) {
        result = result + term;
        term = -term * x_sq / ((i+1) * (i+2));

        if (term < 0.000001f && term > -0.000001f) {
            break;
        }
    }
    return result;
}

//cos(x) is calculated using taylor series cos x = 1 - x^2/2! + x^4/4! - x^6/6! + ...
float my_cos(float x) {

    float pi = 3.141592653589793f;
    while (x > pi) x -= 2 * pi;
    while (x < -pi) x += 2 * pi;

    float result = 0;
    float term = 1.0f;
    float x_sq = x * x;

    for (int i = 0; i <= 15; i += 2) {
        result = result + term;
        term = -term * x_sq / ((i+1) * (i+2));

        if (term < 0.000001f && term > -0.000001f) {
            break;
        }
    }
    return result;
}

//tanh(x) is calculated using exp(x)
float my_tanh(float x){
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;

    float expPos= my_exp(x);
    float expNeg = my_exp(-x);

    return (expPos - expNeg) / (expPos + expNeg);
}

//pow(x) 


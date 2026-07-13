#include <stdio.h>
#include <string.h>
#include "nn.h"
#include "weights_data_old.h"
#include "image_data.h"
#define WEIGHTS_SIZE_FLOATS 21124
int main(void){
    static float raw_weights_aligned[WEIGHTS_SIZE_FLOATS];
    static float input_image[IN_CHANNELS][IMG_SIZE][IMG_SIZE];
    static float output_probs[N_CLASSES];
    memcpy(raw_weights_aligned, WEIGHTS_BLOB, sizeof(WEIGHTS_BLOB));
    memcpy(input_image, IMAGE_BLOB, sizeof(IMAGE_BLOB));
    const int* hilbert_indices=(const int*)raw_weights_aligned;
    const float* model_weights=(const float*)raw_weights_aligned;
    model_forward(input_image, output_probs, model_weights, hilbert_indices);
    const char* cn[4]={"Round Elliptical","In-between Elliptical","Cigar-shaped Elliptical","Edge-on Disk"};
    int best=0; for(int i=0;i<N_CLASSES;i++){ printf("Class %d %s %6.2f%%\n",i,cn[i],output_probs[i]*100.0f); if(output_probs[i]>output_probs[best])best=i; }
    printf("Final Prediction: %s\n", cn[best]);
    fflush(stdout);
#ifdef __riscv
    __asm__ volatile("li a7,93\n\t li a0,0\n\t ecall");
#endif
    return 0;
}

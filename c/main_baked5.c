#include <stdio.h>
#include <string.h>
#include "nn.h"
#include "data5.h"
#define WSZ 21124
int main(void){
    static float w[WSZ]; static float img[IN_CHANNELS][IMG_SIZE][IMG_SIZE]; static float pr[N_CLASSES];
    memcpy(w, WEIGHTS_BLOB, sizeof(WEIGHTS_BLOB));
    const int* hi=(const int*)w; const float* mw=(const float*)w;
    const char* cn[4]={"Round Elliptical","In-between Elliptical","Cigar-shaped Elliptical","Edge-on Disk"};
    for(int s=0;s<5;s++){
        memcpy(img, IMGS[s], IN_CHANNELS*IMG_SIZE*IMG_SIZE*4);
        model_forward(img, pr, mw, hi);
        int b=0; for(int i=0;i<N_CLASSES;i++) if(pr[i]>pr[b])b=i;
        printf("PRED sample_%d %s %.2f%%\n", s, cn[b], pr[b]*100.0f);
    }
    fflush(stdout);
#ifdef __riscv
    __asm__ volatile("li a7,93\n\t li a0,0\n\t ecall");
#endif
    return 0;
}

#ifndef NN_H
#define NN_H

#include <math.h>

#define IMG_SIZE 64
#define SEQ_LEN 4096
#define D_MODEL 64
#define D_STATE 64
#define N_CLASSES 4
#define IN_CHANNELS 1

typedef struct {
    float log_dt[D_MODEL];
    float log_A_real[D_MODEL][D_STATE/2];
    float A_imag[D_MODEL][D_STATE/2];
    float C[D_MODEL][D_STATE/2][2];
    float D[D_MODEL];
} S4DParams;

void hilbert_scan(
    float input[IN_CHANNELS][IMG_SIZE][IMG_SIZE],
    float output[SEQ_LEN][IN_CHANNELS]
);

void linear_uproject(
    float input[SEQ_LEN][IN_CHANNELS],
    float output[SEQ_LEN][D_MODEL],
    float weight[D_MODEL][IN_CHANNELS],
    float bias[D_MODEL]
);

void linear_fc(
    float input[D_MODEL],
    float output[N_CLASSES],
    float weight[N_CLASSES][D_MODEL],
    float bias[N_CLASSES]
);

void gelu(float* x, int size);

void s4d_layer(
    float input[SEQ_LEN][D_MODEL],
    float output[SEQ_LEN][D_MODEL],
    S4DParams* params
);

#endif

#include "nn.h"

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
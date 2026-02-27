#include <stdio.h>
#include "nn.h"

int main() {

    float img[IN_CHANNELS][IMG_SIZE][IMG_SIZE];
    float seq[SEQ_LEN][IN_CHANNELS];

    // Fill image with known pattern
    for (int y = 0; y < IMG_SIZE; y++) {
        for (int x = 0; x < IMG_SIZE; x++) {
            img[0][y][x] = y * IMG_SIZE + x;
        }
    }

    hilbert_scan(img, seq);

    // Print first 20 outputs
    for (int i = 0; i < 20; i++) {
        printf("seq[%d] = %.0f\n", i, seq[i][0]);
    }

    return 0;
}
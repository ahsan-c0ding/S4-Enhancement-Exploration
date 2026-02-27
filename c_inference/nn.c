#include <stdio.h>
#include "nn.h"

static void rot(int s, int* x, int* y, int rx, int ry) {
    if (ry == 0){
        if (rx == 1) {
            *x = s - 1 - *x;
            *y = s - 1 - *y;
        }
        int temp = *x;
        *x = *y;
        *y = temp;
    }
}

static void d2xy (int n, int d, int* x, int* y) {
    int rx, ry, s, t = d;
    *x = 0;
    *y = 0;

    for (s = 1; s < n ; s *= 2) {
        rx = (t/2) & 1;
        ry = (t^rx) & 1;
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

void hilbert_scan(float input[IN_CHANNELS][IMG_SIZE][IMG_SIZE], float output[SEQ_LEN][IN_CHANNELS]) {
    int indices[SEQ_LEN];
    //Generate Hilbert indices
    for (int d = 0 ; d < SEQ_LEN ; d++) {
        int x, y;
        d2xy(IMG_SIZE, d, &x, &y);
        indices[d] = y * IMG_SIZE + x;
    }
    //Flatten input manually and reorder
    for (int d = 0 ; d < SEQ_LEN ; d++) {
        int flat_index = indices[d];
        int y = flat_index/IMG_SIZE;
        int x = flat_index % IMG_SIZE;

        for (int c = 0 ; c < IN_CHANNELS ; c++) {
            output[d][c] = input[c][y][x];
        }
    }
}
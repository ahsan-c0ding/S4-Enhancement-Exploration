#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

// Total size in floats (84496 bytes / 4 bytes per float = 21124 floats)
#define WEIGHTS_SIZE_FLOATS 21124

int main() {
    // 1. ALLOCATING AS FLOATS GUARANTEES 4-BYTE MEMORY ALIGNMENT
    // This prevents the -O2 GCC Auto-Vectorizer from Segfaulting!
    static float raw_weights_aligned[WEIGHTS_SIZE_FLOATS];
    static float input_image[IN_CHANNELS][IMG_SIZE][IMG_SIZE];
    static float output_probs[N_CLASSES];

    FILE *f_weights = fopen("../model_params/model_weights.bin", "rb");
    if (!f_weights) {
        printf(" Error: Could not find model_weights.bin\n");
        return 1;
    }
    // Read into our properly aligned array
    if (fread(raw_weights_aligned, sizeof(float), WEIGHTS_SIZE_FLOATS, f_weights) != WEIGHTS_SIZE_FLOATS) {
        printf(" Warning: Read different amount of bytes than expected.\n");
    }
    fclose(f_weights);

    FILE *f_img = fopen("../test_data/input_image.bin", "rb");
    if (!f_img) {
        printf(" Error: Could not find input_image.bin\n");
        return 1;
    }
    if (fread(input_image, sizeof(float), IN_CHANNELS * IMG_SIZE * IMG_SIZE, f_img) == 0) {
         printf(" Warning: Failed to read image.\n");
    }
    fclose(f_img);

    // 2. Set up pointers
    const int* hilbert_indices = (const int*)raw_weights_aligned;
    const float* model_weights = (const float*)raw_weights_aligned;

    // 3. Run
    printf(" Booting up S4 Galaxy Classifier Engine (No-Math-Lib Edition)...\n");
    printf("Running forward pass (calculating O(L^2) convolutions twice)...\n");
    
    model_forward(input_image, output_probs, model_weights, hilbert_indices);

    // 4. Print Results
    printf("\n====================================\n");
    printf(" Galaxy Class Predictions\n");
    printf("====================================\n");
    
    const char* class_names[4] = {
        "Round Elliptical", 
        "In-between Elliptical", 
        "Cigar-shaped Elliptical", 
        "Edge-on Disk"
    };
    
    int best_class = 0;
    for (int i = 0; i < N_CLASSES; i++) {
        printf("Class %d [%-24s]: %6.2f%%\n", i, class_names[i], output_probs[i] * 100.0f);
        if (output_probs[i] > output_probs[best_class]) {
            best_class = i;
        }
    }
    
    printf("====================================\n");
    printf(" FINAL PREDICTION: %s\n", class_names[best_class]);
    printf("====================================\n");

    return 0;
}
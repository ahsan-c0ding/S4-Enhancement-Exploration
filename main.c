#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

#define WEIGHTS_SIZE_FLOATS 21124

int main(int argc, char *argv[]) { 
    if (argc != 2) {
        printf("Usage: %s <input_image.bin>\n", argv[0]); //Ask for command line input
        return 1;
    }

    static float raw_weights_aligned[WEIGHTS_SIZE_FLOATS];
    static float input_image[IN_CHANNELS][IMG_SIZE][IMG_SIZE];
    static float output_probs[N_CLASSES];

    FILE *f_weights = fopen("model_params/model_weights.bin", "rb");
    if (!f_weights) {
        printf(" Error: Could not find model_weights.bin\n");
        return 1;
    }
    if (fread(raw_weights_aligned, sizeof(float), WEIGHTS_SIZE_FLOATS, f_weights) != WEIGHTS_SIZE_FLOATS) {
        printf(" Warning: Read different amount of bytes than expected.\n");
    }
    fclose(f_weights);

    //  Load the image specified in the command line argument from ../test_data/<file_name>.bin
    FILE *f_img = fopen(argv[1], "rb");
    if (!f_img) {
        printf(" Error: Could not find %s\n", argv[1]);
        return 1;
    }
    if (fread(input_image, sizeof(float), IN_CHANNELS * IMG_SIZE * IMG_SIZE, f_img) == 0) {
         printf(" Warning: Failed to read image.\n");
    }
    fclose(f_img);

    const int* hilbert_indices = (const int*)raw_weights_aligned;
    const float* model_weights = (const float*)raw_weights_aligned;

    printf(" Booting up S4 Galaxy Classifier Engine (No-Math-Lib Edition)...\n");
    printf("Running forward pass (recurrent S4D, O(L*N) per layer)...\n");
    
    model_forward(input_image, output_probs, model_weights, hilbert_indices);

    printf("\nGalaxy Class Predictions\n");

    //C-type string (char*) array
    const char* class_names[4] = {
        "Round Elliptical", 
        "In-between Elliptical", 
        "Cigar-shaped Elliptical", 
        "Edge-on Disk"
    };

    //loop over all classes and find highest probablity
    int best_class = 0;
    for (int i = 0; i < N_CLASSES; i++) {
        printf("Class %d [%-24s]: %6.2f%%\n", i, class_names[i], output_probs[i] * 100.0f);
        if (output_probs[i] > output_probs[best_class]) {
            best_class = i;
        }
    }
    
    printf("\nFinal Prediction: %s\n", class_names[best_class]);

    return 0;
}

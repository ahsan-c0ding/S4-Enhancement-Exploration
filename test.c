#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"

#define WEIGHTS_SIZE_FLOATS 21124

// Helper function to read a binary file into a float array
int load_bin(const char* filepath, float* buffer, int expected_floats) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return 0;
    int read = fread(buffer, sizeof(float), expected_floats, f);
    fclose(f);
    return read == expected_floats;
}

// Computes MSE/MAE and checks against rubric thresholds
int validate_layer(const char* name, float* c_out, float* py_out, int size, double mse_thresh, double mae_thresh) {
    double mse = 0.0, mae = 0.0, max_err = 0.0;
    for(int i = 0; i < size; i++) {
        double diff = (double)c_out[i] - (double)py_out[i];
        double abs_diff = diff < 0.0 ? -diff : diff;
        mse += diff * diff;
        mae += abs_diff;
        if (abs_diff > max_err) max_err = abs_diff;
    }
    mse /= size; 
    mae /= size;
    
    printf("%-18s | MSE: %9.2e | MAE: %9.2e | MaxErr: %9.2e | ", name, mse, mae, max_err);
    if (mse <= mse_thresh && mae <= mae_thresh) {
        printf(" PASS\n");
        return 1;
    } else {
        printf(" FAIL\n");
        return 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <sample_prefix_path>\n", argv[0]);
        return 1;
    }
    
    char filepath[256];
    const char* prefix = argv[1];

    // Allocate memory blocks
    static float raw_weights[WEIGHTS_SIZE_FLOATS];
    static float input_image[IN_CHANNELS][IMG_SIZE][IMG_SIZE];
    
    // Layer buffers
    static float hilbert_out[SEQ_LEN][IN_CHANNELS];
    static float proj_out[SEQ_LEN][D_MODEL];
    static float s4d1_out[SEQ_LEN][D_MODEL];
    static float s4d2_out[SEQ_LEN][D_MODEL];
    static float pooled[D_MODEL];
    static float logits[N_CLASSES];
    static float probs[N_CLASSES];

    // Reusable buffer to load Python reference binaries
    static float py_ref[SEQ_LEN * D_MODEL]; 

    // Load Weights
    FILE *f_w = fopen("model_params/model_weights.bin", "rb");
    if (!f_w) {
        printf("Error: Could not open model_params/model_weights.bin\n");
        return 1;
    }
    if (fread(raw_weights, sizeof(float), WEIGHTS_SIZE_FLOATS, f_w) != WEIGHTS_SIZE_FLOATS) {
        printf("Error reading weights file.\n");
        fclose(f_w);
        return 1;
    }
    fclose(f_w);

    // Load Input Image
    sprintf(filepath, "%s_img.bin", prefix);
    if (!load_bin(filepath, (float*)input_image, IN_CHANNELS * IMG_SIZE * IMG_SIZE)) {
        printf("Failed to load image %s\n", filepath);
        return 1;
    }

    // MAP POINTERS (Exact replica of nn.c mapping)
    int offset = 0;
    const int* hilbert_indices = (const int*)raw_weights;
    offset += 4096 * sizeof(int);
    const float* w = (const float*)((const char*)raw_weights + offset);
    
    const float* uproj_w = w; w += 64 * 1;
    const float* uproj_b = w; w += 64;
    const float* s4_1_dt = w; w += 64;
    const float* s4_1_Ar = w; w += 64 * 32;
    const float* s4_1_Ai = w; w += 64 * 32;
    const float* s4_1_Cr = w; w += 64 * 32 * 2;
    const float* s4_1_Ci = s4_1_Cr + 1; 
    const float* s4_1_D  = w; w += 64;
    const float* s4_2_dt = w; w += 64;
    const float* s4_2_Ar = w; w += 64 * 32;
    const float* s4_2_Ai = w; w += 64 * 32;
    const float* s4_2_Cr = w; w += 64 * 32 * 2;
    const float* s4_2_Ci = s4_2_Cr + 1;
    const float* s4_2_D  = w; w += 64;
    const float* fc_w    = w; w += 4 * 64;
    const float* fc_b    = w; w += 4;

    printf("===================================================================================\n");
    int passed_all = 1;

    // 1. Hilbert
    hilbert_scan(input_image, hilbert_out, hilbert_indices);
    sprintf(filepath, "%s_hilbert.bin", prefix); 
    if (!load_bin(filepath, py_ref, 4096 * 1)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("Hilbert Scan", (float*)hilbert_out, py_ref, 4096 * 1, 1e-12, 1e-12); 

    // 2. UProject
    linear((float*)hilbert_out, (float*)proj_out, uproj_w, uproj_b, SEQ_LEN, IN_CHANNELS, D_MODEL);
    sprintf(filepath, "%s_uproject.bin", prefix); 
    if (!load_bin(filepath, py_ref, 4096 * 64)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("Linear (UProject)", (float*)proj_out, py_ref, 4096 * 64, 1e-8, 1e-6); 

    // 3. S4D 1
    s4d_layer(proj_out, s4d1_out, s4_1_dt, s4_1_Ar, s4_1_Ai, s4_1_Cr, s4_1_Ci, s4_1_D);
    printf("\r"); // Clear the printing carriage return from s4d
    sprintf(filepath, "%s_s4_1.bin", prefix); 
    if (!load_bin(filepath, py_ref, 4096 * 64)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("S4D Layer 1", (float*)s4d1_out, py_ref, 4096 * 64, 1e-7, 5e-4); 

    // 4. GELU 1
    gelu((float*)s4d1_out, 4096 * 64);
    sprintf(filepath, "%s_gelu_1.bin", prefix); 
    if (!load_bin(filepath, py_ref, 4096 * 64)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("GELU 1", (float*)s4d1_out, py_ref, 4096 * 64, 1e-7, 5e-4); 

    // 5. S4D 2 (Accumulated error increases thresholds slightly)
    s4d_layer(s4d1_out, s4d2_out, s4_2_dt, s4_2_Ar, s4_2_Ai, s4_2_Cr, s4_2_Ci, s4_2_D);
    printf("\r");
    sprintf(filepath, "%s_s4_2.bin", prefix); 
    if (!load_bin(filepath, py_ref, 4096 * 64)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("S4D Layer 2", (float*)s4d2_out, py_ref, 4096 * 64, 5e-7, 1e-3); 

    // 6. GELU 2
    gelu((float*)s4d2_out, 4096 * 64);
    sprintf(filepath, "%s_gelu_2.bin", prefix); 
    if (!load_bin(filepath, py_ref, 4096 * 64)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("GELU 2", (float*)s4d2_out, py_ref, 4096 * 64, 5e-7, 1e-3); 

    // 7. Take Last (Inherits GELU 2 error)
    take_last_timestamp(s4d2_out, pooled);
    sprintf(filepath, "%s_takelast.bin", prefix); 
    if (!load_bin(filepath, py_ref, 64)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("Take Last", pooled, py_ref, 64, 5e-7, 1e-3); 

    // 8. FC (Logits)
    linear(pooled, logits, fc_w, fc_b, 1, D_MODEL, N_CLASSES);
    for(int i=0; i<4; i++) probs[i] = logits[i]; // copy for softmax
    sprintf(filepath, "%s_fc.bin", prefix); 
    if (!load_bin(filepath, py_ref, 4)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("FC (Logits)", logits, py_ref, 4, 1e-6, 1e-3); 

    // 9. Softmax (Normalizes everything back down to strict rubric limits!)
    softmax(probs, 4);
    sprintf(filepath, "%s_softmax.bin", prefix); 
    if (!load_bin(filepath, py_ref, 4)) { printf("CRITICAL ERROR: Missing reference file %s\n", filepath); return 1; }
    passed_all &= validate_layer("Softmax", probs, py_ref, 4, 1e-8, 1e-4); 

    printf("===================================================================================\n");
    if (passed_all) {
        printf("STATUS: ALL LAYERS PASSED\n");
        return 0;
    } else {
        printf("STATUS: FAILED\n");
        return 1;
    }
}
#ifndef NN_H
#define NN_H

#define IMG_SIZE 64	//Imput image is 64 x 64 pixels
#define SEQ_LEN 4096	//64 * 64 = 4096 positions after hilbert scan
#define D_MODEL 64	//Hidden dimension
#define D_STATE 64	//s4D state dimension (32 complex pairs)
#define N_CLASSES 4	//4 types of galaxy
#define IN_CHANNELS 1	//1 becuase of grayscale


//Reorders 2D image pixels into 1D sequence following Hilbert curve.
void hilbert_scan(
    float input[IN_CHANNELS][IMG_SIZE][IMG_SIZE],	//input image (C, 64, 64)
    float output[SEQ_LEN][IN_CHANNELS],			//Output sequence (4096, C)
    const int* hilbert_indices			        //Pre-computed indices from weights go here
);

// Generic Linear Layer: handles both sequence and vector inputs.
// For uproject: batch_size=SEQ_LEN, in_features=IN_CHANNELS, out_features=D_MODEL
// For fc:       batch_size=1,       in_features=D_MODEL,     out_features=N_CLASSES
void linear(
    const float* input,    // flat input  [batch_size × in_features]
    float* output,         // flat output [batch_size × out_features]
    const float* weight,   // [out_features × in_features] row-major
    const float* bias,     // [out_features]
    int batch_size,
    int in_features,
    int out_features
);

//S4D Layer: (4096, 64) -> (4096, 64) - Core sequence modeling component
void s4d_layer(
    float input[SEQ_LEN][D_MODEL],
    float output[SEQ_LEN][D_MODEL],
    const float* log_dt,
    const float* log_A_real,
    const float* A_imag,
    const float* C_real,
    const float* C_imag,
    const float* D
);

//GELU Activation: Applies Gaussian Error Linear Unit to input array
void gelu(
    float* x, //Input/Output array (modified in-place)
    int size //Number of elements in the array
);

//Softmax Activation: Converts logits to probability distribution over 4 classes
void softmax(
    float* logits, //Input logits array of size 4 (modified in-place)
    int size //Number of classes (we have 4 classes)
);

//Take Last Timestamp: Extracts final timestep from sequence for classification
void take_last_timestamp(
    float input[SEQ_LEN][D_MODEL], //Input sequence of shape (4096, 64)
    float output[D_MODEL]	//Output is a vector of shape (64) (last position)
);

//Complete model forward pass: Chains all layers together
void model_forward(
    float image[IN_CHANNELS][IMG_SIZE][IMG_SIZE],  //Input image (C, 64, 64)
    float probabilities[N_CLASSES],                 //Output class probabilities
    const float* model_weights,                     //Flat array of all weights
    const int* hilbert_indices                       //Pre-computed Hilbert indices
);
#endif

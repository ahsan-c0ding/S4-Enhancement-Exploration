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

// Generic Linear Layer: handles both sequence and vector inputs
void linear(
    const float* input,    // flat input  [batch_size × in_features]
    float* output,         // flat output [batch_size × out_features]
    const float* weight,   // [out_features × in_features] row-major
    const float* bias,     // [out_features]
    int batch_size,
    int in_features,
    int out_features
);

//Input Projection Layer: (4096, C) -> (4096, 64)
void linear_uproject(
    float input[SEQ_LEN][IN_CHANNELS],			//Input sequence of shape (4096, C) from Hilbert scan
    float output[SEQ_LEN][D_MODEL],			//Output sequence of shape (4096, 64) projected features
    const float weight[D_MODEL][IN_CHANNELS],			//Weight matrix of shape (64, C) stored in row-major order
    const float bias[D_MODEL]					//Bias vector of shape (64)
);

//Final Classification Layer: (64) -> (4)
void linear_fc(
    const float input[D_MODEL],			//Input feature vector of shape (64) after TakeLastTimestamp
    float output[N_CLASSES],			//Output class scores of shape (4) before softmax
    const float weight[N_CLASSES][D_MODEL],		//Weight matrix of shape (4, 64) in row-major order
    const float bias[N_CLASSES]			//Bias vector of shape (4)
);

//S4D Layer: (4096, 64) -> (4096, 64) - Core sequence modeling component
void s4d_layer(
    float input[SEQ_LEN][D_MODEL],		//Input is squence of shape(4096, 64)
    float output[SEQ_LEN][D_MODEL],		//Output is sequence of shape (4096, 64)
    //S4DParams* params (Since struct is not being used anymore we will separately implement components
    const float* log_dt, //Log step size for each channel (64)
    const float* log_A_real, //Real part of log A matrix (64, 32) flattened
    const float* A_imag,  //Imaginary part of matrix A (64, 32)
    const float* C_real, //Imaginary part of matrix C (64, 32)
    const float* C_imag,  //Imaginary part of matrix C
    const float* D // Feedthrough matrix(64)
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

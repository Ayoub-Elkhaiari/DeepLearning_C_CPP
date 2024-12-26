#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// XOR training dataset
double inputs[4][2] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

double outputs[4][1] = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
};

// Sigmoid activation function and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Randomly initialize weights and biases
void initialize_weights_and_biases(double hidden_weights[2][2], double hidden_bias[2], double output_weights[2], double *output_bias) {
    for (int i = 0; i < 2; i++) {
        hidden_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < 2; j++) {
            hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    *output_bias = ((double)rand() / RAND_MAX) * 2 - 1;
    for (int i = 0; i < 2; i++) {
        output_weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

// Perform forward propagation
void forward_propagate(double input[2], double hidden_weights[2][2], double hidden_bias[2], double output_weights[2], double output_bias, double hidden_layer[2], double *output_layer) {
    for (int i = 0; i < 2; i++) {
        hidden_layer[i] = hidden_bias[i];
        for (int j = 0; j < 2; j++) {
            hidden_layer[i] += input[j] * hidden_weights[i][j];
        }
        hidden_layer[i] = sigmoid(hidden_layer[i]);
    }

    *output_layer = output_bias;
    for (int i = 0; i < 2; i++) {
        *output_layer += hidden_layer[i] * output_weights[i];
    }
    *output_layer = sigmoid(*output_layer);
}

// Perform backpropagation and update weights and biases
void backpropagate(double input[2], double hidden_layer[2], double output_layer, double target, double hidden_weights[2][2], double hidden_bias[2], double output_weights[2], double *output_bias, double learning_rate) {
    double output_error = target - output_layer;
    double output_delta = output_error * sigmoid_derivative(output_layer);

    double hidden_error[2], hidden_delta[2];
    for (int i = 0; i < 2; i++) {
        hidden_error[i] = output_delta * output_weights[i];
        hidden_delta[i] = hidden_error[i] * sigmoid_derivative(hidden_layer[i]);
    }

    // Update weights and biases
    for (int i = 0; i < 2; i++) {
        output_weights[i] += learning_rate * output_delta * hidden_layer[i];
    }
    *output_bias += learning_rate * output_delta;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            hidden_weights[i][j] += learning_rate * hidden_delta[i] * input[j];
        }
        hidden_bias[i] += learning_rate * hidden_delta[i];
    }
}

// Train the neural network
void train(int epochs, double learning_rate, double hidden_weights[2][2], double hidden_bias[2], double output_weights[2], double *output_bias) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;
        for (int sample = 0; sample < 4; sample++) {
            double hidden_layer[2], output_layer;

            // Forward propagation
            forward_propagate(inputs[sample], hidden_weights, hidden_bias, output_weights, *output_bias, hidden_layer, &output_layer);

            // Compute error
            double error = outputs[sample][0] - output_layer;
            total_error += error * error;

            // Backpropagation
            backpropagate(inputs[sample], hidden_layer, output_layer, outputs[sample][0], hidden_weights, hidden_bias, output_weights, output_bias, learning_rate);
        }

        // Print error every 1000 epochs
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Error: %f\n", epoch, total_error);
        }
    }
}

// Test the neural network
void test(double hidden_weights[2][2], double hidden_bias[2], double output_weights[2], double output_bias) {
    printf("Testing results:\n");
    for (int sample = 0; sample < 4; sample++) {
        double hidden_layer[2], output_layer;

        // Forward propagation
        forward_propagate(inputs[sample], hidden_weights, hidden_bias, output_weights, output_bias, hidden_layer, &output_layer);

        printf("Input: %.1f %.1f, Predicted: %.4f, Expected: %.1f\n",
               inputs[sample][0], inputs[sample][1], output_layer, outputs[sample][0]);
    }
}

int main() {
    srand(time(NULL));

    // Network parameters
    double hidden_weights[2][2], hidden_bias[2], output_weights[2], output_bias;
    int epochs = 10000;
    double learning_rate = 0.1;

    // Initialize weights and biases
    initialize_weights_and_biases(hidden_weights, hidden_bias, output_weights, &output_bias);

    // Train the network
    train(epochs, learning_rate, hidden_weights, hidden_bias, output_weights, &output_bias);

    // Test the network
    test(hidden_weights, hidden_bias, output_weights, output_bias);

    return 0;
}


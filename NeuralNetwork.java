import java.util.Arrays;
import java.util.Random;

import static java.lang.Math.*;

// A neural network assuming that all the units in
// layer l is connected to all the units in layer l+1.
// It uses back-propagation algorithm to find the optimal weights.
public class NeuralNetwork {

    public static Random random = new Random();
    private Layer[] network;
    private int[] layers;
    private int numLayers;
    private double learningRate;
    private long maxIterations;
    private double allowedError;

    // Layers array indicate the unit size in every layer.
    // Length of the array itself indicates the number of layers.
    // The layers[0] indicates the size of input.
    // The layers[len -1] indicates the output size.
    // There will be len - 2 hidden layers.
    public NeuralNetwork(int[] layers, double learningRate, long maxIterations, double allowedError) {

        numLayers = layers.length - 1;
        this.learningRate = learningRate;
        this.allowedError = allowedError;
        this.maxIterations = maxIterations;
        network = new Layer[layers.length - 1];
        this.layers = layers;
        // No need to create a layer for input
        System.out.println("Constructing the network...");
        for (int i = 1; i < layers.length; ++i) {
            System.out.println("Initializing layer " + i + " with layer size " + layers[i] + " input size = " + layers[i - 1]);
            network[i - 1] = new Layer(layers[i], layers[i - 1]);
        }
    }

    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(new int[]{2, 2, 1}, 0.1, (long) pow(10, 7), 0.01);

        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targetOutput = {{0}, {1}, {1}, {0}};

        neuralNetwork.train(inputs, targetOutput);

        double[][] predicted = neuralNetwork.test(inputs);

        for (int i = 0; i < inputs.length; ++i) {
            String input = Arrays.toString(inputs[i]);
            String actualOut = Arrays.toString(targetOutput[i]);
            String predictedOut = Arrays.toString(predicted[i]);
            System.out.printf("\nInput = %s Actual output = %s Predicted output = %s", input, actualOut, predictedOut);
        }
    }

    // Takes the input from the training set and computes the output.
    private double[] feedForward(double[] inputs) {
        if (inputs.length != network[0].inputs.length) {
            throw new RuntimeException("In-valid input size.");
        }

        double[] previous_output = inputs;
        for (Layer layer : network) {
            layer.inputs = previous_output;
            layer.computeOutput();
            previous_output = layer.output;
        }
        return previous_output;
    }

    public void train(double[][] trainingInputs, double[][] targetOutputs) {
        System.out.println("Started training the network...");
        if (trainingInputs.length != targetOutputs.length) {
            throw new RuntimeException("Inputs and target output lengths are different");
        }

        if (targetOutputs[0].length != layers[numLayers]) {
            throw new RuntimeException("Target output size for a single " +
                    "training case should match the output layer size");
        }

        long count = 0;
        double average_error = 0;
        while (count++ < maxIterations) {

            average_error = 0;
            for (int i = 0; i < trainingInputs.length; ++i) {
                double[] input = trainingInputs[i];
                double[] targetOutput = targetOutputs[i];

                // Output of the network.
                double[] out = feedForward(input);
                double curr_error = 0;
                for (int k = 0; k < out.length; ++k) {
                    curr_error += abs(out[k] - targetOutput[k]);
                }
                curr_error /= out.length;
                average_error += curr_error;
                backPropagate(targetOutput);
            }
            if (average_error < allowedError) {
                break;
            }
        }
        System.out.println("Iteration count = " + count);
        System.out.println("Average error = " + average_error);

    }

    public double[][] test(double[][] trainingInputs) {

        double[][] output = new double[trainingInputs.length][layers[numLayers]];
        for (int i = 0; i < trainingInputs.length; ++i) {
            double[] out = (feedForward(trainingInputs[i]));
            for (int j = 0; j < out.length; ++j) {
                output[i][j] = out[j];
            }
        }
        return output;
    }

    private void backPropagate(double[] targetOutput) {

        // Now for each layer.
        double[] delta_previous = null;
        for (int i = numLayers - 1; i >= 0; --i) {
            Layer currLayer = network[i];
            double[] error = new double[currLayer.numUnits];
            double[] derivatives = new double[currLayer.numUnits];
            double[] delta = new double[currLayer.numUnits];

            Util.derivative(currLayer.output, derivatives);

            // If output layer.
            if (i == numLayers - 1) {
                Util.subtract(targetOutput, currLayer.output, error);
            } else {
                for (int j = 0; j < currLayer.numUnits; ++j) {
                    error[j] = 0;
                    for (int k = 0; k < network[i + 1].numUnits; k++) {
                        error[j] += network[i + 1].weights[k][j] * delta_previous[k];
                    }
                }
            }

            Util.multiply(error, derivatives, delta);

            // Multiply delta with constant.
            double[] ratedDelta = new double[currLayer.numUnits];

            // Multiply delta with learning rate
            Util.multiplyConst(delta, learningRate, ratedDelta);

            //System.out.println("rated delta is " + Arrays.toString(ratedDelta));

            // Adjust bias.
            Util.increment(currLayer.bias, ratedDelta);

            double[] weightInputDelta = new double[currLayer.inputs.length];
            // Adjust weights
            for (int k = 0; k < currLayer.numUnits; ++k) {
                Util.multiplyConst(currLayer.inputs, ratedDelta[k], weightInputDelta);
                Util.increment(currLayer.weights[k], weightInputDelta);
            }
            delta_previous = delta;
        }
    }

    private static class Layer {
        double[] inputs;
        double[][] weights;
        double[] bias;
        double[] output;
        int numUnits;

        public Layer(int layerSize, int inputSize) {
            numUnits = layerSize;
            inputs = new double[inputSize];
            weights = new double[layerSize][inputSize];
            bias = new double[layerSize];
            output = new double[layerSize];

            // Initialize with random weights;
            for (int i = 0; i < layerSize; ++i) {
                bias[i] = abs(random.nextDouble()) % 2 / 100;
                for (int j = 0; j < inputSize; ++j) {
                    weights[i][j] = abs(random.nextDouble()) % 2 / 100;
                }
            }
        }

        public void computeOutput() {
            for (int i = 0; i < numUnits; ++i) {
                output[i] = Util.dot(inputs, weights[i]);
                output[i] += bias[i];
                output[i] = Util.sigmoid(output[i]);
            }
        }
    }

    static class Util {

        public static double dot(final double[] vec1, final double[] vec2) {
            double out = 0;
            for (int i = 0; i < vec2.length; ++i) {
                out += vec1[i] * vec2[i];
            }
            return out;
        }

        public static double sigmoid(double x) {
            return (1 / (1 + exp(-x)));
        }

        public static double derivative(double x) {
            return x * (1 - x);
        }

        public static void subtract(double[] vec1, double[] vec2, double[] out) {
            for (int i = 0; i < vec1.length; ++i) {
                out[i] = vec1[i] - vec2[i];
            }
        }

        public static void derivative(double[] vec1, double[] out) {
            for (int i = 0; i < vec1.length; ++i) {
                out[i] = derivative(vec1[i]);
            }
        }

        public static void multiply(double[] vec1, double[] vec2, double[] out) {
            for (int i = 0; i < vec1.length; ++i) {
                out[i] = vec1[i] * vec2[i];
            }
        }

        public static void increment(double[] out, double[] vec1) {
            for (int i = 0; i < vec1.length; ++i) {
                out[i] += vec1[i];
            }
        }

        public static void multiplyConst(double[] vec1, Double constant, double[] out) {
            for (int i = 0; i < vec1.length; ++i) {
                out[i] = vec1[i] * constant;
            }
        }
    }
}

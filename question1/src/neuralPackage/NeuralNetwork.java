package neuralPackage;

public class NeuralNetwork {
//    private double[][] weightsValue; // Weights between input and hidden layer
//    private double[] biasesValue; // Biases of the hidden layer

    public NeuralNetwork() {
        // Initialize the weights and biases
//        weightsValue = new double[][]{{0.5, 0.5}};       // both inputs have equal weights of 0.5, meaning both inputs are equally important in determining the output
//                                                         // when both inputs are 1, the weighted sum in hidden layer becomes 0.5 * 1 + 0.5 * 1 = 1
//                                                         // if either or both input are 0, the weighted sum becomes less than 1, which causes activation function to return 0 in hidden layer
//        biasesValue = new double[]{-0.7}; // the weighted sum is shifted downward by 0.7, meaning the weighted sum in hidden layer would be
                                            // 0.5 * 1 + 0.5 * 1 - 0.7 = 0.1, when we apply activate function, the output becomes 1
    }

    // if value is positive, return 1
    // if value is negative, return 0
    private int activate(double x) {
        if (x >= 0) {
            return 1;
        } else {
            return 0;
        }
    }

    // Forward propagation to calculate the output
    public int feedForward(int input1, int input2) {
        // Calculate the weighted sum of the hidden layer
        double hiddenSum = 0.5 * input1 + 0.5 * input2 - 0.7; // here we use 0.5 because that is the respected weight
                                                              // we subtract by 0.7 because it is the bias of the hidden layer

        int hiddenLayerOutput = activate(hiddenSum); // determine the output of the hidden layer. If hiddenSum is greater than
                                                // or equal to 0, the activate function returns 1; otherwise, it returns 0
        return hiddenLayerOutput;
    }

}

import neuralPackage.NeuralNetwork;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        // create an instance of NeuralNetwork class
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        Scanner s = new Scanner(System.in); // creates object s to read user input from console

        // prompt the user to enter the number of input pairs
        System.out.println("Please enter the number of input pairs: ");
        int numPairs = s.nextInt();
        int[][] input = new int[numPairs][2];   // store pairs of input values for neural network. Each input pair consists
                                                // of two integer values representing input1 and input2

        // Inside the loop, the program reads two int for each pair
        System.out.println("Please enter the input pairs: ");
        for (int i = 0; i < numPairs; i++) {
//            input[i][0] = s.nextInt();
//            input[i][1] = s.nextInt();
            int num1 = s.nextInt();
            int num2 = s.nextInt();

            // check if user input is 0 or 1, if not, return an error
            if (num1 != 0 && num1 != 1) {
                System.out.println("The input is not valid");
                return;
            }
            if (num2 != 0 && num2 != 1) {
                System.out.println("The input is not valid");
                return;
            }

            // the values that were validated from the if statement (either 0 or 1) is stored in the input array at
            // respective indices
            input[i][0] = num1;
            input[i][1] = num2;
        }

        // Prints the results by iterating over each input pair in the input array using the enhanced for loop.
        // For each pair, the values num1 and num2 are extracted
        System.out.println("The Results are: ");
        for (int[] pair : input) {
            int num1 = pair[0];
            int num2 = pair[1];
            int output = neuralNetwork.feedForward(num1, num2); // call function with num1 and num2 as arguments which calculates
                                                                // the output of the hidden layer which is then stored in output variable
        // Test the neural network with different inputs
//        int[][] input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
//        for (int[] pair : input) {
//            int input1 = pair[0];
//            int input2 = pair[1];
//            int output = neuralNetwork.feedForward(input1, input2);
            System.out.println(num1 + " AND " + num2 + " = " + output);
        }
    }

}


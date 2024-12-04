using System;
using System.Runtime.Intrinsics.X86;

public class Program
{
    public static void Main()
    {
        // Settings
        int input_size  = 2;
        int hidden_size = 4;
        int output_size = 1;

        // Helpers
        Random random = new Random();
        double Rng() => random.NextDouble() * 2 - 1;

        // Sigmoid activation function
        double activate(double x) => 1 / (1 + Math.Exp(-x));
        double dactivate(double x) => x * (1 - x);

        // ReLU activation function
        // double activate(double x) => Math.Max(0, x);
        // double dactivate(double x) => x > 0 ? 1 : 0;


        // Weights and biases initialization
        double[][] Weights = new double[2][]; // 0: input->hidden, 1: hidden->output
        Weights[0] = new double[input_size * hidden_size].Select(x => Rng()).ToArray();
        Weights[1] = new double[hidden_size * output_size].Select((x) => Rng()).ToArray();

        double[][] biases = new double[2][]; // 0: input->hidden, 1: hidden->output
        biases[0] = new double[hidden_size].Select((x) => Rng()).ToArray();
        biases[1] = new double[output_size].Select((x) => Rng()).ToArray();

        // A single layer forward pass
        double[] WeightFoward(double[] inputs, double[] weights, double[] bias)
        {
            double[] output = new double[weights.Length / inputs.Length];
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = 0;
                for (int j = 0; j < inputs.Length; j++)
                {
                    double value = inputs[j] * weights[j + i * inputs.Length];
                    output[i] += value;
                }

                // Bias & Activation
                output[i] = activate(output[i] + bias[i]);
            }
            return output;
        }

        // full network backpropagation
        void Backward(double[] inputs, double[] targets, double learning_rate)
        {
            // Forward pass
            double[] input_to_hidden = WeightFoward(inputs, Weights[0], biases[0]);
            double[] hidden_to_output = WeightFoward(input_to_hidden, Weights[1], biases[1]);

            // Backward pass
            double[] output_errors = new double[output_size];
            for (int i = 0; i < output_size; i++)
            {
                output_errors[i] = targets[i] - hidden_to_output[i];
            }

            double[] hidden_errors = new double[hidden_size];
            for (int i = 0; i < hidden_size; i++)
            {
                hidden_errors[i] = 0;
                for (int j = 0; j < output_size; j++)
                {
                    hidden_errors[i] += output_errors[j] * Weights[1][i + j * hidden_size];
                }
            }

            /// -----------------
            //  Update weights and biases
            /// -----------------

            for (int i = 0; i < output_size; i++)
            {
                for (int j = 0; j < hidden_size; j++)
                {
                    Weights[1][j + i * hidden_size] +=
                        learning_rate * output_errors[i] * dactivate(hidden_to_output[i]) * input_to_hidden[j];
                }
            }

            for (int i = 0; i < hidden_size; i++)
            {
                for (int j = 0; j < input_size; j++)
                {
                    Weights[0][j + i * input_size] +=
                        learning_rate * hidden_errors[i] * dactivate(input_to_hidden[i]) * inputs[j];
                }
            }

            // Update input->hidden biases
            for (int i = 0; i < output_size; i++)
            {
                biases[1][i] += learning_rate * output_errors[i] * dactivate(hidden_to_output[i]);
            }

            // Update hidden->output biases
            for (int i = 0; i < hidden_size; i++)
            {
                biases[0][i] += learning_rate * hidden_errors[i] * dactivate(input_to_hidden[i]);
            }
        }

        double[] Forward(double[] inputs)
        {
            double[] input_to_hidden = WeightFoward(inputs, Weights[0], biases[0]);
            double[] hidden_to_output = WeightFoward(input_to_hidden, Weights[1], biases[1]);
            return hidden_to_output;
        }

        // Training
        double learning_rate = 0.1;

        double[][] inputs = new double[][]
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };

        double[][] targets = new double[][]
        {
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 }
        };

        for (int i = 0; i < 10000; i++)
        {
            Console.SetCursorPosition(0, 0);
            Console.WriteLine($" ===== Epoch {i} =====");
            for (int j = 0; j < inputs.Length; j++)
            {
                Backward(inputs[j], targets[j], learning_rate);

                double[] _output;

                _output = Forward(new double[] { 0, 0 });
                Console.WriteLine($"[0,0] => [{_output[0]:F10}] (Want [{0:F10}]) (ERROR: {Math.Abs(0 - _output[0]):F10})");

                _output = Forward(new double[] { 1, 0 });
                Console.WriteLine($"[1,0] => [{_output[0]:F10}] (Want [{1:F10}]) (ERROR: {Math.Abs(1 - _output[0]):F10})");

                _output = Forward(new double[] { 0, 1 });
                Console.WriteLine($"[0,1] => [{_output[0]:F10}] (Want [{1:F10}]) (ERROR: {Math.Abs(1 - _output[0]):F10})");

                _output = Forward(new double[] { 1, 1 });
                Console.WriteLine($"[1,1] => [{_output[0]:F10}] (Want [{0:F10}]) (ERROR: {Math.Abs(0 - _output[0]):F10})");
            }
        }
    }
}

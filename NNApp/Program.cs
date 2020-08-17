using System;
using System.Collections.Generic;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetworks;

namespace NNApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var input = Matrix<double>.Build.Dense(10, 1, 10);
            var weights = Matrix<double>.Build.Dense(1, 10, 0.001);
            var bias = Matrix<double>.Build.Dense(1, 1, 1);
            
            var node = new Node();
            var Z = node.Linear(input, weights, bias);
            var A = node.Relu(Z);
            
            Console.WriteLine(A);
        }
    }
}
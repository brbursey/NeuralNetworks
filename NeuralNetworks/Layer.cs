using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class Layer
    {
        public Matrix<double> Weight { get; set; }
        public Matrix<double> Bias { get; set; }

        public Layer(int size, int previous)
        {
            Weight = Matrix<double>.Build.Random(size, previous) / 100;
            Bias = Matrix<double>.Build.Dense(size, size, 0);
        }
        
        public Matrix<double> LinearFunction(Matrix<double> input)
        {
            return Weight * input + Bias ;
        }
        
        public Matrix<double> Relu(Matrix<double> z)
        {
            var activation = z.Map(val => Math.Max(0, val));            
            return activation;
        }

        private Matrix<double> Sigmoid(Matrix<double> z)
        {
            var activation = 1 / (1 + z.Map(val => Math.Exp(-val)));
            return activation;
        }
    }
}
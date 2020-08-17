using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
{
    public class Node
    {
        public Matrix<double> Linear(Matrix<double> X, Matrix<double> W, Matrix<double> b)
        {
            return W * X + b ;
        }
        
        public Matrix<double> Relu(Matrix<double> z)
        {
            var activation = z.Map(val => Math.Max(0, val));            
            return activation;
        }

        public Matrix<double> Sigmoid(Matrix<double> z)
        {
            var activation = 1 / (1 + z.Map(val => Math.Exp(-val)));
            return activation;
        }
    }
}
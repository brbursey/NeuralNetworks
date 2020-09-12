using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public static class NeuralNetworkFactory
    {
        public static NeuralNetwork Create(List<int> layers, Matrix<double> input, NeuralNetworkConfiguration config)
        {
            ICost cost;
            NeuralNetwork network;
            switch (config.Cost)
            {
                case Cost.Linear:
                    cost = new LinearRegression();
                    break;
                case Cost.Logistic:
                    cost = new LogisticRegression();
                    break;
                default:
                    throw new ArgumentException("Must use a valid cost function");
            }
            
            return new NeuralNetwork(layers, input, cost);
        }
    }

    public class NeuralNetworkConfiguration
    {
        public Cost Cost { get; set; }
    }

    public enum Cost
    {
        Linear,
        Logistic
    }
}
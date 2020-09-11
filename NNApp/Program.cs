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
            var input = Matrix.Build.Dense(10, 1, 1);
            var layers = new List<int> {3, 2, 1};
            var config = new NeuralNetworkConfiguration()
            {
                Cost = Cost.Linear
            };
            var network = NeuralNetworkFactory.Create(layers, input, config);
            network.Train();
            // var pred = network2.Predict(testX);
        }
    }
}
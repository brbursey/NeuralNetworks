using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetworks;

namespace NNApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var config = new NeuralNetworkConfiguration()
            {
                Cost = Cost.Linear,
                TrainTestRatio = 0.7
            };

            var dataset = DatasetFactory.Create("../../../Examples", config);
            var input = Matrix.Build.Dense(10, 1, 1);
            var layers = new List<int> { 3, 2, 1 };
           
            var network = NeuralNetworkFactory.Create(layers, dataset, config);
            network.Train(epochs: 1);
            // var pred = network.Predict(testX);
        }
    }
}
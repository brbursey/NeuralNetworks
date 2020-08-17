using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
{
    public class Network
    {
        public List<Layer> Layers { get; set; }
        public List<Matrix<double>> Weights { get; set; }
        public List<Matrix<double>> Bias { get; set; }

        public Network()
        {
            Weights = new List<Matrix<double>>();
            Bias = new List<Matrix<double>>();
            Layers = new List<Layer>()
            {
                new Layer()
            };
            foreach (var layer in Layers)
            {
                Weights.Add(layer.Weight);
                Bias.Add(layer.Bias);
            }
        }
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
{
    public class Network
    {
        public List<Layer> Layers { get; set; }
        public List<Matrix<double>> Weights { get; set; }
        public List<Matrix<double>> Bias { get; set; }
        
        public Network(List<int> layers, Matrix<double> input)
        {
            Layers = new List<Layer>();
            Weights = new List<Matrix<double>>();
            Bias = new List<Matrix<double>>();
            
            Layers.Add(new Layer(layers[0], input.RowCount));
            Weights.Add(Layers.First().Weight);
            Bias.Add(Layers.First().Bias);
            
            for (int i = 1; i < layers.Count; i++)
            {
                var layer = new Layer(layers[i], layers[i - 1]);
                Layers.Add(layer);
                Weights.Add(layer.Weight);
                Bias.Add(layer.Bias);
            }
        }

        public double LossFunction(Matrix<double> y, Matrix<double> yHat)
        {
            var m = y.RowCount;
            var logProbs = (Matrix.Log(yHat) * y) + (Matrix.Log(1 - yHat) * (1 - y));
            var loss = -logProbs.RowSums() / m;
            return loss.AsArray().First();
        }
    }
}
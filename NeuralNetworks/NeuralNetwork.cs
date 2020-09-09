using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        public List<Matrix<double>> Weights { get; set; }
        public List<Matrix<double>> Bias { get; set; }

        public NeuralNetwork()
        {
            
        }
        public NeuralNetwork(int layers, string type)
        {
        }

        public double LossFunction(Matrix<double> y, Matrix<double> yHat)
        {
            var m = y.RowCount;
            var logProbs = (Matrix.Log(yHat) * y) + (Matrix.Log(1 - y) * (1 - y));
            var loss = -logProbs.RowSums() / m;
            return loss.AsArray().First();
        }
    }
}
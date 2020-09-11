using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
{
    public interface INetwork
    {
        public void Train();
        public void Predict();
    }
    public class NeuralNetwork : INetwork
    {
        private readonly ICost _costFunction;
        
        public List<Layer> Layers { get; set; }
        public List<Matrix<double>> Weights { get; set; }
        public List<Matrix<double>> Bias { get; set; }

        public NeuralNetwork(List<int> layers, Matrix<double> input, ICost costFunction)
        {
            _costFunction = costFunction;
            InitializeParameters(layers, input);
        }

        private void InitializeParameters(List<int> layers, Matrix<double> input)
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

        public  void Train()
        {
            throw new NotImplementedException();
        }

        public void Predict()
        {
            throw new NotImplementedException();
        }


        private Matrix<double> ForwardPropagation(Matrix<double> X)
        {
            var input = X;
            foreach (var layer in Layers)
            {
                var z = layer.LinearFunction(input);
                var a = layer.Relu(z);
                input = a;
            }

            var output = input;
            return output;
        }
    }
}
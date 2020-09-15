using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
{
    public interface INetwork
    {
        public void Train(int epochs);
        public void Predict(Matrix<double> input);
    }
    public class NeuralNetwork : INetwork
    {
        private readonly ICost _costFunction;
        private readonly Dataset _trainingData;

        public Data X { get; set; }
        public Data Y { get; set; }
        public List<Layer> Layers { get; set; }
        public List<Matrix<double>> Weights { get; set; }
        public List<Matrix<double>> Bias { get; set; }

        public NeuralNetwork(List<int> layers, Dataset trainingData, ICost costFunction)
        {
            _trainingData = trainingData;
            _costFunction = costFunction;
            InitializeParameters(layers, _trainingData);
        }

        private void InitializeParameters(List<int> layers, Dataset trainingData)
        {
            Layers = new List<Layer>();
            Weights = new List<Matrix<double>>();
            Bias = new List<Matrix<double>>();

            var input = trainingData.X.Value.RowCount;
            Layers.Add(new Layer(layers[0], input));
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

        public void Train(int epochs)
        {
           for (int i = 0; i < epochs; i++)
           {
               var X = _trainingData.X.Value;
               var y = _trainingData.Y.Value;
               var yHat = ForwardPropagation(X);
               var loss = ComputeCost(y, yHat);
               // BackwardPropagation(loss);
           }
        }

        public void Predict(Matrix<double> input)
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

        public double ComputeCost(Matrix<double> y, Matrix<double> yHat)
        {
            return _costFunction.ComputeCost(y, yHat);
        }
    }
}
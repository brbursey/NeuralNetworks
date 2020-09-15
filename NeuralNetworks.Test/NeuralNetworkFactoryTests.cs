using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using NUnit.Framework;

namespace NeuralNetworks.Test
{
    [TestFixture]
    public class NeuralNetworkFactoryTests
    {
        private NeuralNetwork _subject;
        private List<int> _layers;
        private Dataset _dataset;
        private NeuralNetworkConfiguration _config;

        [SetUp]
        public void SetUp()
        {
            _dataset = new Dataset
            {
                X = new Data
                {
                    Value = Matrix.Build.Dense(10, 1, 1)
                }
            };
            _layers = new List<int> { 3, 2, 1 };
            _config = new NeuralNetworkConfiguration
            {
                Cost = Cost.Logistic
            };
            _subject = NeuralNetworkFactory.Create(_layers, _dataset, _config);
        }

        [Test]
        public void NeuralNetworkFactory_Create_CreatesNeuralNetwork()
        {
            Assert.That(_subject, Is.TypeOf<NeuralNetwork>());
        }
    }
}
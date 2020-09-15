using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using NUnit.Framework;

namespace NeuralNetworks.Test
{
    [TestFixture]
    public class NeuralNetworkTests
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
        public void NeuralNetwork_InitializeParameters_CreatesANetwork_GivenTheNodesInEachLayer()
        {
            var expected = new List<Layer>
            {
                new Layer(3, 10),
                new Layer(2, 3),
                new Layer(1, 2)
            };
            var result = _subject.Layers.Count;

            Assert.That(result, Is.EqualTo(expected.Count));
        }

        [Test]
        public void NeuralNetwork_InitializeParameters_FirstLayerWeightHasSize3x10()
        {
            var rowExpected = 3;
            var colExpected = 10;
            var rowResult = _subject.Layers.First().Weight.RowCount;
            var colResult = _subject.Layers.First().Weight.ColumnCount;

            Assert.That(rowResult, Is.EqualTo(rowExpected));
            Assert.That(colResult, Is.EqualTo(colExpected));
        }

        [Test]
        public void NeuralNetwork_InitializeParameters_WeightsAreProperSize()
        {
            var weight1 = new List<int> { 3, 10 };
            var weight2 = new List<int> { 2, 3 };
            var weight3 = new List<int> { 1, 2 };
            var weights = new List<List<int>>
            {
                weight1, weight2, weight3
            };
            for (int i = 0; i < _subject.Weights.Count; i++)
            {
                var rowCount = _subject.Weights[i].RowCount;
                var rowExpected = weights[i][0];
                var colCount = _subject.Weights[i].ColumnCount;
                var colExpected = weights[i][1];
                
                Assert.That(rowCount, Is.EqualTo(rowExpected));
                Assert.That(colCount, Is.EqualTo(colExpected));
            }
        }

        // [Test]
        // public void NeuralNetwork_Train_UpdatesParametersAfterCalled()
        // {
        //     var preResult = _subject.Weights
        //         .First()
        //         .Enumerate()
        //         .First();
        //     
        //     _subject.Train(1);
        //
        //     var postResult = _subject.Weights
        //         .First()
        //         .Enumerate()
        //         .First();
        //
        //     Assert.That(preResult, Is.Not.EqualTo(postResult));
        // }
        
        [Test]
        public void NeuralNetwork_LossFunction_ReturnsAnInt_WhenYIs_Nx1_AndYHatIs_1xM()
        {
            //var logistic = new LogisticRegression();
            var y = Matrix.Build.Dense(10, 1);
            var yHat = Matrix.Build.Dense(1, 10);

            var result = _subject.ComputeCost(y, yHat);

            Assert.That(result, Is.TypeOf<double>());
        }
    }
}
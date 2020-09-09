using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetworks;
using NUnit.Framework;

namespace NeuralNetwork.Test
{
    [TestFixture]
    public class NeuralNetworkTests
    {
        private Network subject;

        [SetUp]
        public void SetUp()
        {
            var input = Matrix.Build.Dense(10, 1, 1);
            var layers = new List<int> { 3, 2, 1 };
            subject = new Network(layers, input);
        }

        [Test]
        public void NeuralNetwork_CreatesANetwork_GivenTheNodesInEachLayer()
        {
            var expected = new List<Layer>
            {
                new Layer(3, 10),
                new Layer(2, 3),
                new Layer(1, 2)
            };
            var result = subject.Layers.Count;

            Assert.That(result, Is.EqualTo(expected.Count));
        }

        [Test]
        public void NeuralNetwork_FirstLayerWeightHasSize3x10()
        {
            var rowExpected = 3;
            var colExpected = 10;
            var rowResult = subject.Layers.First().Weight.RowCount;
            var colResult = subject.Layers.First().Weight.ColumnCount;

            Assert.That(rowResult, Is.EqualTo(rowExpected));
            Assert.That(colResult, Is.EqualTo(colExpected));
        }

        [Test]
        public void NeuralNetwork_WeightsIsProperSize()
        {
            var weight1 = new List<int> { 3, 10 };
            var weight2 = new List<int> { 2, 3 };
            var weight3 = new List<int> { 1, 2 };
            var weights = new List<List<int>>
            {
                weight1, weight2, weight3
            };
            for (int i = 0; i < subject.Weights.Count; i++)
            {
                var rowCount = subject.Weights[i].RowCount;
                var rowExpected = weights[i][0];
                var colCount = subject.Weights[i].ColumnCount;
                var colExpected = weights[i][1];
                
                Assert.That(rowCount, Is.EqualTo(rowExpected));
                Assert.That(colCount, Is.EqualTo(colExpected));
            }
        }
        
        [Test]
        public void LossFunction_ReturnsAnInt_WhenYIs_Nx1_AndYHatIs_1xM()
        {
            var y = Matrix.Build.Dense(10, 1);
            var yHat = Matrix.Build.Dense(1, 10);

            var result = subject.LossFunction(y, yHat);

            Assert.That(result, Is.TypeOf<double>());
        }
    }
}
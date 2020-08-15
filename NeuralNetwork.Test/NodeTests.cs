using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetworks;
using NUnit.Framework;
using NUnit.Framework.Internal.Builders;

namespace NeuralNetwork.Test
{
    [TestFixture]
    public class NodeTests
    {
        private Node subject;
        private Matrix<double> input;
        private Matrix<double> weights;
        private Matrix<double> bias;
        
        [SetUp]
        public void SetUp()
        {
            subject = new Node();
            input = Matrix<double>.Build.Random(100, 1);
            weights = Matrix<double>.Build.Random(1, 100);
            bias = Matrix<double>.Build.Random(1, 1);
        }
        
        [Test]
        public void Node_Linear_ReturnsMatrixOfShape1x1_When_InputIs100x1_And_WeightIs1x100()
        {
            var expected = 1;
            var result = subject.Linear(input, weights, bias).Rank();

            Assert.That(result, Is.EqualTo(expected));
        }

        [Test]
        public void Node_Linear_ReturnsMatrixOfShape2x2_When_InputIs100x2And_WeightIs2x100()
        {
            input = Matrix<double>.Build.Random(100, 2);
            weights = Matrix<double>.Build.Random(2, 100);
            bias = Matrix<double>.Build.Random(2, 2);

            var expected = 2;
            var result = subject.Linear(input, weights, bias).Rank();

            Assert.That(result, Is.EqualTo(expected));
        }

        [Test]
        public void Node_Relu_Returns2x2MatrixOf0s_WhenAllValuesAreLessThan0()
        {
            var z = Matrix<double>.Build.Dense(5, 5, -1);

            var expected = Matrix<double>.Build.Dense(5, 5, 0);
            var result = subject.Relu(z);

            Assert.That(result, Is.EqualTo(expected));
        }
        
        [Test]
        public void Node_Relu_ReturnsSameMatrix_WhenAllValuesAreGreaterThan0()
        {
            var z = Matrix<double>.Build.Dense(5, 5, 1);

            var expected = Matrix<double>.Build.Dense(5, 5, 1);
            var result = subject.Relu(z);

            Assert.That(result, Is.EqualTo(expected));
        }

        [Test]
        public void Node_Sigmoid_ReturnsCorrectAnswer_WhenInputIs2x2Matrix()
        {
            input = Matrix<double>.Build.Dense(2, 2, 1);

            var sigmoid = 1 / (1 + Math.Exp(-1));
            var expected = Matrix<double>.Build.Dense(2, 2, sigmoid);
            var result = subject.Sigmoid(input);

            Assert.That(result, Is.EqualTo(expected));
        }
        
        [Test]
        public void Node_Sigmoid_ReturnsCorrectAnswer_WhenInputIs1x10Matrix()
        {
            input = Matrix<double>.Build.Dense(1, 10, 1);

            var sigmoid = 1 / (1 + Math.Exp(-1));
            var expected = Matrix<double>.Build.Dense(1, 10, sigmoid);
            var result = subject.Sigmoid(input);

            Assert.That(result, Is.EqualTo(expected));
        }
    }
}
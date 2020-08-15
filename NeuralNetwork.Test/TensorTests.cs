using System;
using System.Collections.Generic;
using NeuralNetworks;
using NUnit.Framework;

namespace NeuralNetwork.Test
{
    [TestFixture]
    public class TensorTests
    {
        [SetUp]
        public void Setup()
        {
            
        }

        [Test]
        public void Tensor_BaseCase_Creates2x2MatrixOfZeros_WhenShapeIs2x2()
        {
            var shape = new Tuple<int, int>(2, 2);
            var subject = new BaseCaseTensor(shape);
            var expected = new int[,]
            {
                {0, 0}, 
                {0, 0}
            };
            var result = subject.Value;

            Assert.That(result, Is.EqualTo(expected));
        }

        [Test]
        public void Tensor_BaseCase_CreatesA2xNMatrix_WhenGivenValues()
        {
            var values = new List<List<int>>()
            {
                new List<int>() {1, 2}, 
                new List<int>() {3, 4}
            };
            var subject = new BaseCaseTensor(values);

            var expected = new int[,]
            {
                {1, 2},
                {3, 4}
            };
            var result = subject.Value;

            Assert.That(result, Is.EqualTo(expected));
        }
        
    }
}
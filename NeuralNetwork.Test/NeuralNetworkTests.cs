using MathNet.Numerics.LinearAlgebra.Double;
using NUnit.Framework;

namespace NeuralNetwork.Test
{
    [TestFixture]
    public class NeuralNetworkTests
    {
        private NeuralNetworks.NeuralNetwork subject;
        
        public void SetUp()
        {
            subject = new NeuralNetworks.NeuralNetwork();
        }
        
        [Test]
        public void LossFunction_ReturnsAnInt_WhenYIs_Nx1_AndYHatIs_1xM()
        {
            var n = 10;
            var m = 20;
            var y = Matrix.Build.Dense(n, 1);
            var yHat = Matrix.Build.Dense(1, m);

            var result = subject.LossFunction(y, yHat);

            Assert.That(result, Is.TypeOf<double>());
        }
    }
}
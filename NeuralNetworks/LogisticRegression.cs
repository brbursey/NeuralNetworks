using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
{
    public class LogisticRegression : ICost
    {
        public double ComputeCost(Matrix<double> y, Matrix<double> yHat)
        {
            var m = y.RowCount;
            var logProbs = (Matrix.Log(yHat) * y) + (Matrix.Log(1 - yHat) * (1 - y));
            var loss = -logProbs.RowSums() / m;
            return loss.AsArray().First();
        }
    }
}
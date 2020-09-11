using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public interface ICost
    {
        public double ComputeCost(Matrix<double> y, Matrix<double> yHat);
    }
}
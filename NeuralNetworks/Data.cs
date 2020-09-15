using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class Data
    {
        public string Name { get; set; }
        public Matrix<double> Value { get; set; }
        public DatasetType Type { get; set; }
    }

    public class Dataset
    {
        public Data X { get; set; }
        public Data Y { get; set; }
    }
}
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
{
    public class Layer
    {
        public List<Node> Nodes { get; set; }
        public Matrix<double> Weight { get; set; }
        public Matrix<double> Bias { get; set; }

        public Layer()
        {
            Nodes = new List<Node>();
            Nodes.Add(new Node());
            
            Weight = Matrix<double>.Build.Dense(Nodes.Count, 1, 0.01);
            Bias = Matrix<double>.Build.Dense(Nodes.Count, Nodes.Count, 0);
        }
    }
}
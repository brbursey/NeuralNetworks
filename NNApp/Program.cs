using System;
using System.Collections.Generic;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetworks;

namespace NNApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var m = Matrix<double>.Build.Random(500, 500);
            var v = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Random(500);
            var y = m.Solve(v);
            Console.WriteLine(y);
            Console.WriteLine(m);

            var tensor = new Vector4();
        }
    }
}
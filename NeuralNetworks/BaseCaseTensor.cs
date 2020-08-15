using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class BaseCaseTensor
    {
        public int[,] Value { get; set; }

        public BaseCaseTensor(List<List<int>> values)
        {
            Value = CreateMatrix(values);
        }
        
        public BaseCaseTensor(Tuple<int, int> shape)
        {
            var (item1, item2) = shape;
            Value = new int[item1, item2];
        }
        
        //TODO: Add unit test
        private int[,] CreateMatrix(List<List<int>> values)
        {
            var result = new int[2, 2];
            
            var unrolledValues = Unroll(values);
            for (int i = 0; i < unrolledValues.Count; i++)
            {
                result.SetValue(unrolledValues[i], i / result.Rank, i % result.Rank);
            }

            return result;
        }
        
        //TODO: Add unit test
        private List<int> Unroll(List<List<int>> values)
        {
            return values.SelectMany(val => val).ToList();
        }
    }
}
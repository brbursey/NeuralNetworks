using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace NeuralNetworks
{
    public static class DatasetFactory
    {
        public static Dataset Create(string pathToDataFolder, NeuralNetworkConfiguration config)
        {
            // Todo: look into dataframes!
            var dataset = new Dataset();
            
            foreach (var filename in Directory.GetFiles(pathToDataFolder))
            {
                var contents = DelimitedReader.Read<double>(filename, false, ",", true);
                if (contents.ColumnCount > 1)
                {
                    dataset.X = new Data
                    {
                        Name = "X",
                        Value = contents,
                        Type = DatasetType.Input
                    };
                }
                else
                {
                    dataset.Y = new Data
                    {
                        Name = "y",
                        Value = contents,
                        Type = DatasetType.Output
                    };
                }
            }
            
            return dataset;
        }
    }

    public enum DatasetType
    {
        Input, 
        Output
    }
}
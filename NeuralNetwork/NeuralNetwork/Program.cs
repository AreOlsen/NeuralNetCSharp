using System.Globalization;
namespace NeuralNetwork;

using NeuralNetwork.Activations;
using NeuralNetwork.Loss;
using CsvHelper;

public class Program
{
	public static void Main(string[] args)
	{
        LeakyReLU relu = new(0.01d);
        Linear lin = new();
        Network net = new(2,2,2,relu,1,lin);
        MSE mse = new(net);
        var (x,y) = ReadCSV("/home/cockblocker/Desktop/Programming/C#/NeuralNetworkCSharp/NeuralNetwork/NeuralNetwork/train.csv","x1","x2","y");
        List<double> loss = net.Train(1000,0.001d,x,y,mse);
        for(int epoch = 0; epoch < loss.Count; epoch++)
        {
            Console.WriteLine($"Epoch {epoch}, loss:{loss[epoch]}");
        }   
    }

    private static (List<List<double>> x, List<List<double>> y) ReadCSV(String filepath,String y, params String[] x)
    {
        var xData = new List<List<double>>();
        var yData = new List<List<double>>();

        using (var reader = new StreamReader(filepath))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            csv.Read();
            csv.ReadHeader();
            
            while (csv.Read())
            {
                var xRow = new List<double>();
                foreach (var col in x)
                {
                    xRow.Add(csv.GetField<double>(col));
                }
                xData.Add(xRow);
                
                yData.Add(new List<double> { csv.GetField<double>(y) });
            }
        }

        return (xData, yData);
    }
}

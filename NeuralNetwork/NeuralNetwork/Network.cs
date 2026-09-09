namespace NeuralNetwork;

using NeuralNetwork.Activations;
using NeuralNetwork.Layers.Input;
using NeuralNetwork.Layers.Output;
using NeuralNetwork.Layers.Hidden;
using NeuralNetwork.Loss;

public class Network {
    private readonly List<Dense> _dense = new();
    private readonly Input _input;
    private readonly Output _output;
    public int OutputSize { get;  }
    public int InputSize { get;  }

    public Network(int inputNodes, int hiddenLayers, int hiddenNodes, IActivation hiddenActivation,  int outputNodes, IActivation outputActivation)
    {
        InputSize = inputNodes;
        OutputSize = outputNodes;

        Input input = new(inputNodes);
        _input = input;
        _dense.Add(new Dense(input.Nodes,hiddenNodes,hiddenActivation));
        for (int i = 1; i < hiddenLayers; i++){
            Dense dense = new(_dense[i-1].Nodes, hiddenNodes, hiddenActivation);
            _dense.Add(dense);
        }
        _output = new Output(_dense[^1].Nodes, outputNodes, outputActivation);
    }

	public List<double> Predict(List<double> x)
	{
	    _input.SetInputs(x);
		for(int i = 0; i<_dense.Count; i++){
		    _dense[i].Forward();
		}
        _output.Forward();
        return _output.GetValues();
	}

	public List<double> Train(int epochs, double learningRate, List<List<double>> x, List<List<double>> y, ILoss loss){
        List<double> lossHistory = new(epochs);
        for (int epoch = 0; epoch < epochs; epoch++){
            double lossValue = loss.Loss(x, y);
            lossHistory.Add(lossValue);

            for (int iteration = 0; iteration< x.Count; iteration++ ){
                List<double> gradient = loss.LossGradient(x[iteration],y[iteration]);
                _output.SetErrors(gradient);
                _output.Backward(learningRate);
                for (int layer = _dense.Count-1; layer >= 0; layer--){
                    _dense[layer].Backward(learningRate);
                }
            }
        }
        return lossHistory;
    }
}

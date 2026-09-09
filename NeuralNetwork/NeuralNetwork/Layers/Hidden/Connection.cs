namespace NeuralNetwork.Layers.Hidden;

using NeuralNetwork.Layers;
using NeuralNetwork.Layers.Hidden;

public record Connection(Node inputNode, DenseNode outputNode) {
    public double Weight { get; private set; } = 1d;

    public void UpdateWeight(double learningRate){
        Weight -= learningRate * outputNode.Error * inputNode.Value;
    }

	public double GetValue(){
	    return Weight*inputNode.Value;
	}
}

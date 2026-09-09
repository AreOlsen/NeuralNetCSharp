using NeuralNetwork.Layers.Hidden;
using NeuralNetwork.Activations;

namespace NeuralNetwork.Layers.Output;

public class Output : Dense {
    public Output(List<Node> inputNodes, int nodes, IActivation activation) : base(inputNodes,nodes,activation) {}

    public void SetErrors(List<double> errors){
        for (int i = 0; i < errors.Count; i++)
        {
            ((DenseNode)Nodes[i]).Error = errors[i];
        }
    }
}

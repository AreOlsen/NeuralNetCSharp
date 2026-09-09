namespace NeuralNetwork.Layers.Hidden;

using NeuralNetwork.Activations;

public class Dense : Layer {
    public Dense(List<Node> inputNodes, int nodes, IActivation activation){
        for(int i = 0; i < nodes; i++){
            DenseNode node = new(inputNodes, activation);
            Nodes.Add(node);
        }
    }

    public void Forward(){
        foreach(DenseNode node in Nodes){
            node.Forward();
        }
    }

    public void Backward(double learningRate){
        foreach(DenseNode node in Nodes){
            node.Backward(learningRate);
        }
    }
}

namespace NeuralNetwork.Layers.Input;

using NeuralNetwork.Layers;


public class Input : Layer {
    public Input(int nodes){
        for(int i = 0; i < nodes; i++){
            Node node = new Node();
            Nodes.Add(node);
        }
    }

    public void SetInputs(List<double> inputs){
        for (int i = 0; i < Nodes.Count; i++){
            Nodes[i].Value = inputs[i];
        }
    }
}

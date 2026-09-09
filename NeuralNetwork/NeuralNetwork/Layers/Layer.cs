namespace NeuralNetwork.Layers;

public abstract class Layer
{
    public List<Node> Nodes { get; } = new();
    public List<double> GetValues(){
        List<double> values = new();
        for(int i = 0; i<Nodes.Count; i++){
            values.Add(Nodes[i].Value);
        }
        return values;
    }
}

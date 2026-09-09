    namespace NeuralNetwork.Layers.Hidden;

    using NeuralNetwork.Activations;
    using NeuralNetwork.Layers.Hidden;

    public class DenseNode : Node {
        private readonly List<Connection> _inputs = new();
        private readonly IActivation _activation;
        public double Error { get; set; } // Gradient of loss respect to this node output.
        public double NetInput { get; private set; }
        private double _bias;


        public DenseNode(List<Node> inputNodes, IActivation activation){
            _activation=activation;
            foreach(Node input in inputNodes){
                _inputs.Add(new Connection(input,this));
            }
        }

        public void Backward(double learningRate){
            AccumulateError();
            UpdateBias(learningRate);
            UpdateIncomingConnections(learningRate);
            Error = 0;
    }

        private void AccumulateError(){
            foreach(Connection con in _inputs){
                if(con.inputNode is DenseNode input){
                    input.Error += input.ValueDerivative() * Error * con.Weight;
                }
            }
    }

        private void UpdateBias(double learningRate){
            _bias -= learningRate * Error;
        }

        private void UpdateIncomingConnections(double learningRate){
            foreach(Connection con in _inputs){
                con.UpdateWeight(learningRate);
            }
        }

        public void Forward(){
            NetInput = _bias + _inputs.Sum(node => node.GetValue());
            Value = _activation.Calculate(NetInput);
            Error = 0;
        }

        private double ValueDerivative(){
            return _activation.Derivative(NetInput);
        }
    }

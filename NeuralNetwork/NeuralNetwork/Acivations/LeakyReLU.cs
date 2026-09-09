namespace NeuralNetwork.Activations;

public class LeakyReLU(double alpha) : IActivation {
    public double Calculate(double x){
        return Math.Max(0,x)-Math.Min(alpha*x,0);
    }

    public double Derivative(double x) {
        return x>=0 ? 1 : -alpha;
    }
}

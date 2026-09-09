namespace NeuralNetwork.Loss;

public interface ILoss {
    public double Loss(List<List<double>> x, List<List<double>> y);

    //Stochastic gradient descent.
    public List<double> LossGradient(List<double> x, List<double> y);
}

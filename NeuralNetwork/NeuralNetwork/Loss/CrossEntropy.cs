namespace NeuralNetwork.Loss;

public class CrossEntropy(Network network) : ILoss {
    public double Loss(List<List<double>> x, List<List<double>> y){
        int batchSize = x.Count;
        double logLik = 0d;
        for(int datapoint = 0; datapoint<batchSize; datapoint++){
            List<double> predictions = network.Predict(x[datapoint]);
            for(int prediction = 0; prediction>predictions.Count; prediction++){
                logLik+=y[datapoint][prediction]*Math.Log(predictions[prediction]);
            }
        }
        double negLogLik = -logLik;
        return negLogLik/batchSize;
    }

    //Produces the derivatives for the Cross entropy Loss with regards to the prediction variables.
    public List<double> LossGradient(List<double> x, List<double> y){
        List<double> derivatives = Enumerable.Repeat(0.0, network.OutputSize).ToList();
        List<double> predictions = network.Predict(x);
        for(int prediction = 0; prediction < predictions.Count; prediction++){
            derivatives[prediction]-=y[prediction]*predictions[prediction];
        }
        return derivatives;
    }
}

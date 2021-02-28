import numpy as np
import matplotlib.pyplot as plt

class Model(object):

    def __init__(self,num_of_inputs, learning_rate = 0.001, threshold = 100):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_of_inputs + 1)


    def prediction(self,inputs):
        cost = np.dot(inputs,self.weights[1:]) + self.weights[0]

        if cost > 0:
            y_predicted = 1

        else:
            y_predicted = 0

        return y_predicted

    def training ( self, training_inputs, labels):
        b = []
        for j in range(1,self.threshold + 1):
            for x, y in zip(training_inputs,labels):
                predicted = self.prediction(x)
                #a = predicted
                self.weights[1:] = self.weights[1:] + self.learning_rate * ( y - predicted ) * x
                self.weights[0] = self.weights[0] + self.learning_rate * ( y - predicted )

                b.append(self.weights[0])

            print("Epoch" + str(j) + ":")
            print("Value of Bias = ",self.weights[0])
            print("Values of weights = ",self.weights[1:])
            print("---------------------------------------")

        # Plotting Weight Convergence
        plt.plot(self.weights[1:])
        plt.title("Plotting Weight Convergence")
        plt.show()

        # Plotting Bias Correction
        plt.xlim([0,20])
        plt.plot(b)
        plt.title("Plotting Bias Correction")
        plt.show()




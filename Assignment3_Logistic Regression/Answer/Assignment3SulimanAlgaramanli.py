import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Logistic_Regression():
    
    # Constructor
    def __init__(self, dataset_name, learning_rate, no_of_iterations):
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.cost_fun = []  # Cost Array

    # Traning
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()
            cost = self.compute_cost()
            self.cost_fun.append(cost)

        print("Final estimates of b and w are: ", self.b, self.w)

        # Cost function
        print('Initial loss\t:', self.cost_fun[0])
        print('Final loss\t:', self.cost_fun[-1])
        self.plot_cost_fun()

    # Gradient Descent
    def update_weights(self):
        
        z = np.dot(self.X, self.w) + self.b
        predictions = self.sigmoid(z)
        
        # Differentiation
        dw = (1/self.m) * np.dot(self.X.T, (predictions - self.Y))
        db = (1/self.m) * np.sum(predictions - self.Y)
        
        #update
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def sigmoid(self, z):
        z = 1 / (1 + np.exp(-z))
        return z

    # Log Loss
    def compute_cost(self):
        z = np.dot(self.X, self.w) + self.b
        predictions = self.sigmoid(z)
        cost = -(1/self.m) * np.sum(self.Y * np.log(predictions) + (1 - self.Y) * np.log(1 - predictions))
        return cost
    
    def predict(self, X):
        Y_pred = self.sigmoid(np.dot(X, self.w) + self.b)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred

    def load_dataset(self):
        heart_dataset = pd.read_csv(self.dataset_name)
        heart_dataset.famhist.replace(('Present', 'Absent'), (0, 1), inplace=True)
        X = heart_dataset.drop(columns=['row.names', 'chd'], axis=1)
        y = heart_dataset["chd"]
        X = X.values
        y = y.values
        return X, y
    
    def sklearn_LR(self, X_train, X_test, Y_train, Y_test):
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)
        y_pred = lr.predict(X_test)
        testing_data_accuracy = accuracy_score(Y_test, y_pred)
        print('sklearn classifier: Accuracy score of the testing data : ', testing_data_accuracy)

    def plot_cost_fun(self):
        plt.plot(range(1, self.no_of_iterations + 1), self.cost_fun, marker='.')
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()


if __name__ == "__main__":
    dataset_name = "c:\\Users\\sulim\\Downloads\\Heart.csv" # Edit according to your file path
    learning_rate = 0.0001
    no_of_iterations = 200
    classifier = Logistic_Regression(dataset_name, learning_rate, no_of_iterations)
    X, y = classifier.load_dataset()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    testing_data_accuracy = accuracy_score(Y_test, y_pred)
    print('Accuracy score of the testing data : ', testing_data_accuracy)

    classifier.sklearn_LR(X_train, X_test, Y_train, Y_test)

    input_data = [124, 4.00, 12.42, 31.29, 1, 54, 23.23, 2.06, 42]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    standardized_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(standardized_data)

    if prediction[0] == 0:
        print('The person is not coronary heart disease')
    else:
        print('The person has coronary heart disease')

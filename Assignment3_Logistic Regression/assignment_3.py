# importing numpy library
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Logistic_Regression():


  # declaring dataset name learning rate & number of iterations
    def __init__(self, dataset_name,learning_rate, no_of_iterations):
        self.dataset_name=dataset_name
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations



  # fit function to train the model with dataset
    def fit(self, X, Y):

        # number of data points in the dataset (number of rows)  -->  m
        # number of input features in the dataset (number of columns)  --> n
        self.m, self.n = X.shape


        #initiating weight & bias value

        self.w = np.zeros(self.n)
        
        self.b = 0

        self.X = X

        self.Y = Y


    # implementing Gradient Descent for Optimization
        for i in range(self.no_of_iterations):     
            self.update_weights()
        print("Final estimates of b and w are: ", self.b, self.w)

    def sigmoid(self): 
        #    Compute sigmoid function given the input z.
        Z=0
        return Z
    def update_weights(self):
         # Compute gradient logistic regression. 




        #update w and b
        self.w = self.w - 0

        self.b = self.b - 0



  # Sigmoid Equation & Decision Boundary
    def predict(self, X):
        Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) ))     
        Y_pred = np.where( Y_pred > 0.5, 1, 0)
        return Y_pred

    def load_dateset(self):
        #load Heart dataset csv file
        heart_dataset = pd.read_csv(self.dataset_name)
        heart_dataset.famhist.replace(('Present', 'Absent'), (0, 1), inplace=True)
        # separate the x from the y
        X = heart_dataset.drop(columns = ['row.names','chd'], axis=1)
        y = heart_dataset["chd"]
        X = X.values
        y = y.values
        #return features X an Labels Y
        return X,y
    
    def sklearn_LR(self,X_train, X_test, Y_train, Y_test):
        lr=LogisticRegression()
        #training the Logistic Regression Classifier
        lr.fit(X_train,Y_train)
        y_prid=lr.predict(X_test)
        # accuracy score on the testing data
        testing_data_accuracy = accuracy_score( Y_test, y_prid)
        print('sklearn classifier: Accuracy score of the testing data : ', testing_data_accuracy)
 


if __name__ == "__main__":
    # loading the diabetes dataset to a pandas DataFrame
    dataset_name="Heart.csv"
    learning_rate=0.01
    no_of_iterations=1000
    classifier = Logistic_Regression(dataset_name,learning_rate, no_of_iterations)
    X,y=classifier.load_dateset()
    #Scales all the values between [0,1]. It often speeds up the learning process.
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state=2)

    #training the support vector Machine Classifier
    classifier.fit(X_train, Y_train)
    # accuracy score on the training data
    y_prid = classifier.predict(X_test)
    testing_data_accuracy = accuracy_score( Y_test, y_prid)
    print('Accuracy score of the training data : ', testing_data_accuracy)
    ## Logistic_Regression 
    classifier.sklearn_LR(X_train, X_test, Y_train, Y_test)

    #Making a Predictive System
    input_data = [124, 4.00,12.42,31.29,1,54,23.23,  2.06,42]

    # changing the input_data to numpy array
    input_data_as_numpy_array =  np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    print(input_data_reshaped)

    # standardize the input data
    standardized_data = scaler.transform(input_data_reshaped)


    prediction = classifier.predict(standardized_data)

    if (prediction[0] == 0):
        print('The person is not coronary heart disease')
    else:
        print('The person is coronary heart disease')
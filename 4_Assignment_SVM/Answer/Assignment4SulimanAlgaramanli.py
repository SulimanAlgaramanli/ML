# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class SVM_Classifier():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_dataset(self):
        # load Heart dataset csv file
        heart_dataset = pd.read_csv(self.dataset_name)
        heart_dataset.famhist.replace(('Present', 'Absent'), (0, 1), inplace=True)
        X = heart_dataset.drop(columns=['row.names', 'chd'], axis=1)
        y = heart_dataset["chd"]
        X = X.values
        y = y.values
        return X, y

    def scale_features(self, X):
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        return X_scaled

    def train_svm(self, X_train, Y_train, C, gamma, kernel):
        svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
        svm_model.fit(X_train, Y_train)

        return svm_model

    def predict(self, model, X):
        Y_pred = model.predict(X)
        return Y_pred

    def grid_search(self, X_train, Y_train):
        # Create a dictionary called param_grid and fill out some parameters for kernels, C, and gamma
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        
        # Create a GridSearchCV object and fit it to the training data
        
        # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)

        grid.fit(X_train, Y_train)
        
        # Find the optimal parameters
        #print("Optimal Parameters:", grid.best_estimator_)

        return grid.best_estimator_

if __name__ == "__main__":
    dataset_name = "C:\\Users\\sulim\\Downloads\\MyGitHub\\ML\\4_Assignment_SVM\\Answer\\Heart.csv"

    # Load and preprocess the dataset
    svm_classifier = SVM_Classifier(dataset_name)
    X, y = svm_classifier.load_dataset()
    X_scaled = svm_classifier.scale_features(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2, shuffle=True)

    # Hyperparameter Tuning using GridSearchCV
    best_estimator = svm_classifier.grid_search(X_train, Y_train)

    # Training the SVM Classifier with the best hyperparameters
    trained_svm_model = svm_classifier.train_svm(X_train, Y_train, C=best_estimator.C, gamma=best_estimator.gamma, kernel=best_estimator.kernel)

    # Prediction on the testing data
    y_pred = svm_classifier.predict(trained_svm_model, X_test)
    testing_data_accuracy = accuracy_score(Y_test, y_pred)
    
    print("\nOptimal Parameters:",f"C: {best_estimator.C}, Gamma: {best_estimator.gamma}, Kernel: {best_estimator.kernel}")
    print('Accuracy score on the testing data:', testing_data_accuracy)
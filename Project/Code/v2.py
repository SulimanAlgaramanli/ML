import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class TextLowercaseVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (word.lower() for word in analyzer(doc))

class NaiveBayesModel:
    def __init__(self):
        self.model = MultinomialNB()
        self.vectorizer = TextLowercaseVectorizer()

    def train(self, X_train, y_train):
        X_train = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.vectorizer.transform(X_test)
        return self.model.predict(X_test)

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1100)
        self.vectorizer = TextLowercaseVectorizer()

    def train(self, X_train, y_train):
        X_train = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.vectorizer.transform(X_test)
        return self.model.predict(X_test)

class EmailClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.models = {'Naive Bayes': NaiveBayesModel(), 'Logistic Regression': LogisticRegressionModel()}

    def read_data(self):
        self.data = pd.read_csv(self.data_path)

    def clean_data(self):
        self.data = self.data.fillna('')
        self.data = self.data[self.data['text'].notna()]

    def train_models(self):
        results = {}
        for model_name, model in self.models.items():
            accuracies = []
            X = self.data['text']
            y = self.data['label']
            model.train(X, y)
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            accuracies.append(accuracy)
            results[model_name] = accuracies 
        return results

    def print_results(self, results):
        for model_name, accuracies in results.items():
            print(f"\nModel: {model_name}")
            print("Average Accuracy:", sum(accuracies) / len(accuracies))
            X = self.data['text']
            y = self.data['label']
            model = self.models[model_name]
            y_pred = model.predict(X)
            confusion_matrix_result = confusion_matrix(y, y_pred)
            print("Confusion Matrix:")
            print(confusion_matrix_result)

    def plot_models(self):
        for model_name, model in self.models.items():
            X = self.data['text']
            y = self.data['label']
            self.plot_model(X, y, model, model_name)

    def plot_model(self, X, y, model, model_name):
        h = .02  # step size in the mesh

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Plot the decision boundary. Assign a color to each point in the mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Decision Boundary for {model_name}')
        plt.show()

if __name__ == "__main__":
    data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"
    classifier = EmailClassifier(data_path)
    classifier.read_data()
    classifier.clean_data()
    results = classifier.train_models()
    classifier.print_results(results)
    classifier.plot_models()

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

class NaiveBayesModel:
    def __init__(self):
        self.model = MultinomialNB()
        self.vectorizer = CountVectorizer()

    def train(self, X_train, y_train):
        X_train = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.vectorizer.transform(X_test)
        return self.model.predict(X_test)

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1100)
        self.vectorizer = CountVectorizer()

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
        self.models = {'Naive Bayes': NaiveBayesModel(),
                        'Logistic Regression': LogisticRegressionModel()}

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

            print(f"\nConfusion Matrix for {model_name}:")
            print(confusion_matrix(y, y_pred))

        return results

    def predict_single_text(self, text):
        nb_model = self.models['Naive Bayes']
        nb_prediction = nb_model.predict([text])[0]

        lr_model = self.models['Logistic Regression']
        lr_prediction = lr_model.predict([text])[0]

        return nb_prediction, lr_prediction


if __name__ == "__main__":
    data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"
    classifier = EmailClassifier(data_path)
    classifier.read_data()
    classifier.clean_data()
    results = classifier.train_models()
    for model_name, accuracies in results.items():
        print(f"\nModel: {model_name}")
        print("Average Accuracy:", sum(accuracies) / len(accuracies))

    user_input = input("Enter the text to predict: ")
    nb_prediction, lr_prediction = classifier.predict_single_text(user_input)
    print("Naive Bayes Prediction:", nb_prediction)
    print("Logistic Regression Prediction:", lr_prediction)














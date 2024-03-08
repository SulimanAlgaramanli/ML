# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# class NaiveBayesModel:
#     def __init__(self):
#         self.model = MultinomialNB()
#         self.vectorizer = CountVectorizer()

#     def train(self, X_train, y_train):
#         X_train = self.vectorizer.fit_transform(X_train)
#         self.model.fit(X_train, y_train)

#     def predict(self, X_test):
#         X_test = self.vectorizer.transform(X_test)
#         return self.model.predict(X_test)

# class LogisticRegressionModel:
#     def __init__(self):
#         self.model = LogisticRegression()
#         self.vectorizer = CountVectorizer()

#     def train(self, X_train, y_train):
#         X_train = self.vectorizer.fit_transform(X_train)
#         self.model.fit(X_train, y_train)

#     def predict(self, X_test):
#         X_test = self.vectorizer.transform(X_test)
#         return self.model.predict(X_test)

# class EmailClassifier:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.data = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None
#         self.nb_model = NaiveBayesModel()
#         self.lr_model = LogisticRegressionModel()

#     def read_data(self):
#         self.data = pd.read_csv(self.data_path)

#     def clean_data(self):
#         self.data = self.data.fillna('')
#         self.data = self.data[self.data['text'].notna()]

#     def split_data(self):
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['text'], self.data['label'], test_size=0.2, random_state=42)

#     def train_models(self):
#         self.nb_model.train(self.X_train, self.y_train)
#         self.lr_model.train(self.X_train, self.y_train)

#     def evaluate_models(self):
#         models = {'Naive Bayes': self.nb_model, 'Logistic Regression': self.lr_model}
#         for model_name, model in models.items():
#             y_pred = model.predict(self.X_test)
#             accuracy = accuracy_score(self.y_test, y_pred)
#             print(f"\nModel: {model_name}")
#             print("Classification Report:")
#             print(classification_report(self.y_test, y_pred))
#             print("Confusion Matrix:")
#             print(confusion_matrix(self.y_test, y_pred))
#             print("Accuracy:", accuracy)

# if __name__ == "__main__":
#     data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"
#     classifier = EmailClassifier(data_path)
#     classifier.read_data()
#     classifier.clean_data()
#     classifier.split_data()
#     classifier.train_models()
#     classifier.evaluate_models()

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
        self.model = LogisticRegression()
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
        self.models = {'Naive Bayes': NaiveBayesModel(), 'Logistic Regression': LogisticRegressionModel()}

    def read_data(self):
        self.data = pd.read_csv(self.data_path)

    def clean_data(self):
        self.data = self.data.fillna('')
        self.data = self.data[self.data['text'].notna()]

    def train_models(self):
        for model_name, model in self.models.items():
            X = self.data['text']
            y = self.data['label']
            for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"\nModel: {model_name}")
                print("Classification Report:")
                print(classification_report(y_test, y_pred))
                print("Confusion Matrix:")
                print(confusion_matrix(y_test, y_pred))
                print("Accuracy:", accuracy)

if __name__ == "__main__":
    data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"
    classifier = EmailClassifier(data_path)
    classifier.read_data()
    classifier.clean_data()
    classifier.train_models()

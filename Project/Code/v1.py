import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Naive Bayes تعريف النموذج المتعلق بـ 
class NaiveBayesModel:
    def __init__(self):
        self.model = MultinomialNB()  # استخدام نموذج Naive Bayes
        self.vectorizer = CountVectorizer()  # استخدام تقنية Bag-of-Words

    def train(self, X_train, y_train):
        X_train = self.vectorizer.fit_transform(X_train)  # تحويل النصوص إلى تمثيل رقمي
        self.model.fit(X_train, y_train)  # تدريب النموذج

    def predict(self, X_test):
        X_test = self.vectorizer.transform(X_test)  # تحويل النصوص إلى تمثيل رقمي
        return self.model.predict(X_test)  # التنبؤ بالتصنيف

# Logistic Regression تعريف النموذج المتعلق بـ 
class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1100)  # استخدام نموذج Logistic Regression مع زيادة عدد الدورات
        self.vectorizer = CountVectorizer()  # استخدام تقنية Bag-of-Words

    def train(self, X_train, y_train):
        X_train = self.vectorizer.fit_transform(X_train)  # تحويل النصوص إلى تمثيل رقمي
        self.model.fit(X_train, y_train)  # تدريب النموذج

    def predict(self, X_test):
        X_test = self.vectorizer.transform(X_test)  # تحويل النصوص إلى تمثيل رقمي
        return self.model.predict(X_test)  # التنبؤ بالتصنيف

# تعريف الكلاس الرئيسي لتصنيف البريد الإلكتروني
class EmailClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.models = {'Naive Bayes': NaiveBayesModel(), 'Logistic Regression': LogisticRegressionModel()}

    def read_data(self):
        self.data = pd.read_csv(self.data_path)  # قراءة البيانات من المسار المحدد

    def clean_data(self):
        self.data = self.data.fillna('')  # ملء القيم المفقودة بقيم فارغة
        self.data = self.data[self.data['text'].notna()]  # إزالة الصفوف التي تحتوي على قيمة مفقودة في العمود 'text'

    def train_models(self):
        results = {}
        for model_name, model in self.models.items():
            accuracies = []
            X = self.data['text']
            y = self.data['label']
            model.train(X, y)  # استخدام كل البيانات للتدريب
            y_pred = model.predict(X)  # التنبؤ بالتصنيف
            accuracy = accuracy_score(y, y_pred)  # حساب الدقة
            accuracies.append(accuracy)
            results[model_name] = accuracies 
        return results


if __name__ == "__main__":
    data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"  # مسار ملف البيانات
    classifier = EmailClassifier(data_path)
    classifier.read_data()  # قراءة البيانات
    classifier.clean_data()  # تنظيف البيانات

    # قراءة البيانات وتدريب النماذج
    X = classifier.data['text']
    y = classifier.data['label']

    results = classifier.train_models()  # تدريب النماذج باستخدام كل البيانات المتاحة
    for model_name, accuracies in results.items():
        print(f"\nModel: {model_name}")
        print("Average Accuracy:", sum(accuracies) / len(accuracies))  # طباعة متوسط الدقة لكل نموذج
        confusion_matrix_result = confusion_matrix(y, classifier.models[model_name].predict(X))  # حساب مصفوفة الارتباك

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import re

# قراءة مجموعة البيانات
data = pd.read_csv("C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv")

# حذف السطور الفارغة والتعبئة بالقيم الافتراضية
data = data.fillna('')

# تنظيف البيانات بإزالة السطور الفارغة
data = data[data['text'].notna()]

# تقسيم البيانات إلى متغير التحقق والتدريب
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Naive Bayes بناء نموذج الـ
nb_model = MultinomialNB()

# Bag-of-Words تحويل النص إلى متغيرات رقمية باستخدام تقنية  
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# تدريب النموذج
nb_model.fit(X_train, y_train)

# التنبؤ بالتصنيف لمجموعة الاختبار
y_pred = nb_model.predict(X_test)

# التنبؤ بإدخال المستخدم للبريد
def predict_email_category(email_text):
    # تحويل النص إلى متغير رقمي
    email_text_transformed = vectorizer.transform([email_text])
    
    # التنبؤ بالتصنيف
    prediction = nb_model.predict(email_text_transformed)[0]
    
    # تحويل التنبؤ إلى تسمية مقروءة
    if prediction == 0:
        return "Ham"
    elif prediction == 1:
        return "Spam"

# حساب الدقة
accuracy = accuracy_score(y_test, y_pred)

# طباعة تقرير الأداء
print("Classification Report:")
print(classification_report(y_test, y_pred))

# طباعة المصفوفة الواصفة
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# طباعة الدقة
print("Accuracy:", accuracy)

# الدخول من المستخدم
email_text = input("Please enter email text:")

# التنبؤ بالتصنيف
prediction_result = predict_email_category(email_text)
print("Prediction result:", prediction_result)
"""
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# class EmailClassifier:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.data = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None
#         self.vectorizer = CountVectorizer()
#         self.nb_model = MultinomialNB()

#     def read_data(self):
#         self.data = pd.read_csv(self.data_path)

#     def clean_data(self):
#         # حذف السطور الفارغة والتعبئة بالقيم الافتراضية
#         self.data = self.data.fillna('')
#         # تنظيف البيانات بإزالة السطور الفارغة
#         self.data = self.data[self.data['text'].notna()]

#     def prepare_data(self):
#         # تقسيم البيانات إلى متغير التحقق والتدريب
#         X = self.data['text']
#         y = self.data['label']
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # تحويل النص إلى متغيرات رقمية باستخدام تقنية Bag-of-Words
#         self.X_train = self.vectorizer.fit_transform(self.X_train)
#         self.X_test = self.vectorizer.transform(self.X_test)

#     def train_model(self):
#         # تدريب النموذج
#         self.nb_model.fit(self.X_train, self.y_train)

#     def predict(self, email_text):
#         # التنبؤ بالتصنيف
#         email_text_transformed = self.vectorizer.transform([email_text])
#         prediction = self.nb_model.predict(email_text_transformed)[0]
#         return "Ham" if prediction == 0 else "Spam"

#     def evaluate_model(self):
#         # التنبؤ بالتصنيف لمجموعة الاختبار
#         y_pred = self.nb_model.predict(self.X_test)
#         # حساب الدقة
#         accuracy = accuracy_score(self.y_test, y_pred)

#         # طباعة تقرير الأداء
#         print("Classification Report:")
#         print(classification_report(self.y_test, y_pred))

#         # طباعة المصفوفة الواصفة
#         print("Confusion Matrix:")
#         print(confusion_matrix(self.y_test, y_pred))

#         # طباعة الدقة
#         print("Accuracy:", accuracy)

# if __name__ == "__main__":
#     data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"
#     classifier = EmailClassifier(data_path)
#     classifier.read_data()
#     classifier.clean_data()
#     classifier.prepare_data()
#     classifier.train_model()

#     # التنبؤ بإدخال المستخدم للبريد
#     email_text = input("Please enter email text:")
#     prediction_result = classifier.predict(email_text)
#     print("Prediction result:", prediction_result)

#     # تقييم النموذج
#     classifier.evaluate_model()



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
        self.model = LogisticRegression(max_iter=1000)  # زيادة عدد الدورات
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
        results = {}
        for model_name, model in self.models.items():
            accuracies = []
            X = self.data['text']
            y = self.data['label']
            for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            results[model_name] = accuracies
        return results

    def print_results(self, results):
        for model_name, accuracies in results.items():
            print(f"\nModel: {model_name}")
            print("Average Accuracy:", sum(accuracies) / len(accuracies))

if __name__ == "__main__":
    data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"
    classifier = EmailClassifier(data_path)
    classifier.read_data()
    classifier.clean_data()
    results = classifier.train_models()
    classifier.print_results(results)

"""
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# تعريف النموذج المتعلق بـ Naive Bayes
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

# تعريف النموذج المتعلق بـ Logistic Regression
class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)  # استخدام نموذج Logistic Regression مع زيادة عدد الدورات
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
            for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
                X_train, X_test = X[train_index], X[test_index]  # تقسيم البيانات إلى تدريب واختبار
                y_train, y_test = y[train_index], y[test_index]
                model.train(X_train, y_train)  # تدريب النموذج
                y_pred = model.predict(X_test)  # التنبؤ بالتصنيف
                accuracy = accuracy_score(y_test, y_pred)  # حساب الدقة
                accuracies.append(accuracy)
            results[model_name] = accuracies
        return results

    def print_results(self, results):
        for model_name, accuracies in results.items():
            print(f"\nModel: {model_name}")
            print("Average Accuracy:", sum(accuracies) / len(accuracies))  # طباعة متوسط الدقة لكل نموذج

if __name__ == "__main__":
    data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"  # مسار ملف البيانات
    classifier = EmailClassifier(data_path)
    classifier.read_data()  # قراءة البيانات
    classifier.clean_data()  # تنظيف البيانات
    results = classifier.train_models()  # تدريب النماذج
    classifier.print_results(results)  # طباعة النتائج
"""
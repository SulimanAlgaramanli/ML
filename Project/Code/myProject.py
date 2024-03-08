import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
            X = self.data['text']
            y = self.data['label']
            model = self.models[model_name]
            confusion_matrices = []  # قائمة لتخزين مصفوفات الارتباك لكل تجربة
            for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
                X_train, X_test = X[train_index], X[test_index]  # تقسيم البيانات إلى تدريب واختبار
                y_train, y_test = y[train_index], y[test_index]
                model.train(X_train, y_train)  # تدريب النموذج
                y_pred = model.predict(X_test)  # التنبؤ بالتصنيف
                confusion_matrix_result = confusion_matrix(y_test, y_pred)  # حساب مصفوفة الارتباك
                confusion_matrices.append(confusion_matrix_result)
            average_confusion_matrix = sum(confusion_matrices) / len(confusion_matrices)
            print("Average Confusion Matrix:")
            print(average_confusion_matrix)  # طباعة متوسط مصفوفات الارتباك لكل نموذج

if __name__ == "__main__":
    data_path = "C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\combined_data.csv"  # مسار ملف البيانات
    classifier = EmailClassifier(data_path)
    classifier.read_data()  # قراءة البيانات
    classifier.clean_data()  # تنظيف البيانات
    results = classifier.train_models()  # تدريب النماذج
    classifier.print_results(results)  # طباعة النتائج

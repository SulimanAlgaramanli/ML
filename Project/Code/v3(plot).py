import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

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

    def plot_roc_curve(self, X_test, y_test):
        X_test = self.vectorizer.transform(X_test)  # تحويل النصوص إلى تمثيل رقمي
        y_score = self.model.predict_proba(X_test)[:, 1]  # احتمالات الفئة الإيجابية
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.show()

# تعريف النموذج المتعلق بـ Logistic Regression
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

    def plot_roc_curve(self, X_test, y_test):
        X_test = self.vectorizer.transform(X_test)  # تحويل النصوص إلى تمثيل رقمي
        y_score = self.model.predict_proba(X_test)[:, 1]  # احتمالات الفئة الإيجابية
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.show()

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

            # طباعة مصفوفة الارتباك
            print(f"\nConfusion Matrix for {model_name}:")
            print(confusion_matrix(y, y_pred))

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

        # Plot ROC curve for each model
        model = classifier.models[model_name]
        model.plot_roc_curve(X, y)  # رسم بيان الفترة (ROC curve)


           
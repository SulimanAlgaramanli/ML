import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import re

# قراءة مجموعة البيانات
data = pd.read_csv("C:\\Users\\sulim\\Downloads\\DataSets_for_Spam-Emails\\Complete messages.csv")

# حذف السطور الفارغة والتعبئة بالقيم الافتراضية
data = data.fillna('')

# تنظيف البيانات بإزالة السطور الفارغة
data = data[data['text'].notna()]

# تقسيم البيانات إلى متغير التحقق والتدريب
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# بناء نموذج الـNaive Bayes
nb_model = MultinomialNB()

# تحويل النص إلى متغيرات رقمية باستخدام تقنية Bag-of-Words
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



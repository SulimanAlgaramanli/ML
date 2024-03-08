#اضافة المكتبات اللازمة 
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.word_probabilities = {}
        self.classes = []

    def train(self, X_train, y_train):
        # حساب احتمالات الفئات
        total_samples = len(y_train)
        self.classes, class_counts = np.unique(y_train, return_counts=True)
        for i, class_ in enumerate(self.classes):
            self.class_probabilities[class_] = class_counts[i] / total_samples
        
        # حساب احتمالات الكلمات لكل فئة
        for class_ in self.classes:
            # جمع جميع النصوص التابعة لهذه الفئة
            class_text = ' '.join(X_train[y_train == class_])
            # تقسيم النصوص إلى كلمات
            words = class_text.split()
            # حساب تكرار كل كلمة
            word_counts = Counter(words)
            # حساب احتمالات الكلمات
            total_words = len(words)
            self.word_probabilities[class_] = {word: count / total_words for word, count in word_counts.items()}

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            max_probability = -1
            predicted_class = None
            for class_ in self.classes:
                # احتمال الفئة
                class_probability = self.class_probabilities[class_]
                # احتمال النص بالنسبة لهذه الفئة
                text_probability = 1
                for word in sample.split():
                    if word in self.word_probabilities[class_]:
                        text_probability *= self.word_probabilities[class_][word]
                # حساب احتمال الفئة بالنسبة للنص
                total_probability = class_probability * text_probability
                # التحقق مما إذا كان هذا الاحتمال هو الأكبر حتى الآن
                if total_probability > max_probability:
                    max_probability = total_probability
                    predicted_class = class_
            predictions.append(predicted_class)
        return predictions

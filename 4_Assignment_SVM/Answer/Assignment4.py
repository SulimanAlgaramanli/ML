# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, Y_train)
        
        # Find the optimal parameters
        print("Optimal Parameters:", grid.best_estimator_)
        
        return grid.best_estimator_

if __name__ == "__main__":
    dataset_name = "C:\\Users\\sulim\\Downloads\\MyGitHub\\ML\\4_Assignment_SVM\\Answer\\Heart.csv"

    # Load and preprocess the dataset
    svm_classifier = SVM_Classifier(dataset_name)
    X, y = svm_classifier.load_dataset()
    X_scaled = svm_classifier.scale_features(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

    # Hyperparameter Tuning using GridSearchCV
    best_estimator = svm_classifier.grid_search(X_train, Y_train)

    # Training the SVM Classifier with the best hyperparameters
    trained_svm_model = svm_classifier.train_svm(X_train, Y_train, C=best_estimator.C, gamma=best_estimator.gamma, kernel=best_estimator.kernel)

    # Prediction on the testing data
    y_pred = svm_classifier.predict(trained_svm_model, X_test)
    testing_data_accuracy = accuracy_score(Y_test, y_pred)
    print('Accuracy score on the testing data:', testing_data_accuracy)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred))

"""
# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class SVM_Classifier():
    def __init__(self, dataset_name, C, gamma, kernel):
        self.dataset_name = dataset_name
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

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

    def train_svm(self, X_train, Y_train):
        svm_model = SVC(C=self.C, gamma=self.gamma, kernel=self.kernel)
        svm_model.fit(X_train, Y_train)
        return svm_model

    def predict(self, model, X):
        Y_pred = model.predict(X)
        return Y_pred

    def grid_search(self, X_train, Y_train):
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, Y_train)
        print("Optimal Parameters:", grid.best_estimator_)
        return grid.best_estimator_

if __name__ == "__main__":
    dataset_name = "C:\\Users\\sulim\\Downloads\\MyGitHub\\ML\\4_Assignment_SVM\\Answer\\Heart.csv"
    C = 1.0
    gamma = 'scale'  # or you can choose a specific value
    kernel = 'rbf'  # or 'linear', 'poly', 'sigmoid', etc.

    svm_classifier = SVM_Classifier(dataset_name, C, gamma, kernel)
    X, y = svm_classifier.load_dataset()
    X_scaled = svm_classifier.scale_features(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

    # Hyperparameter Tuning using GridSearchCV
    best_estimator = svm_classifier.grid_search(X_train, Y_train)

    # Training the SVM Classifier
    trained_svm_model = svm_classifier.train_svm(X_train, Y_train)

    # Prediction on the testing data
    y_pred = svm_classifier.predict(trained_svm_model, X_test)
    testing_data_accuracy = accuracy_score(Y_test, y_pred)
    print('Accuracy score on the testing data:', testing_data_accuracy)

    # Making a Predictive System
    input_data = [124, 4.00, 12.42, 31.29, 1, 54, 23.23, 2.06, 42]

    # Standardizing the input data
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    standardized_data = svm_classifier.scale_features(input_data_reshaped)

    # Making prediction
    prediction = svm_classifier.predict(trained_svm_model, standardized_data)

    if prediction[0] == 0:
        print('The person is not coronary heart disease.')
    else:
        print('The person is coronary heart disease.')

"""
#
"""
# استيراد الحزم اللازمة
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class SVMClassifier:
    def __init__(self, dataset_name, kernel='rbf', C=1, gamma='scale'):
        self.dataset_name = dataset_name
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def load_dataset(self):
        heart_dataset = pd.read_csv(self.dataset_name)
        heart_dataset.famhist.replace(('Present', 'Absent'), (0, 1), inplace=True)
        X = heart_dataset.drop(columns=['row.names', 'chd'], axis=1)
        y = heart_dataset["chd"]
        X = X.values
        y = y.values
        return X, y

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def grid_search(self, X_train, y_train, param_grid):
        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, y_train)
        print("Optimal Parameters:", grid.best_params_)
        return grid

# تحميل البيانات وتقسيمها
# يمكنك استخدام هذا التكويد خارج الكلاس
svm_classifier = SVMClassifier(dataset_name='C:\\Suliman_Algaramanli\\study\\CS461\\HW\\4\\Heart.csv')  # يجب استبدال 'path_to_Heart.csv' بالمسار الفعلي
X, y = svm_classifier.load_dataset()

# تقسيم البيانات إلى مجموعات التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
svm_classifier.train(X_train, y_train)

# قم بباقي الخطوات حسب الحاجة (توقعات النموذج و GridSearchCV)

# # استخدام النموذج المُعدل للتنبؤ وطباعة التقرير
# grid_predictions = grid_model.predict(X_test)
# print(confusion_matrix(y_test, grid_predictions))
# print(classification_report(y_test, grid_predictions))
"""
#
"""
# استيراد الحزم اللازمة
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def grid_search(self, X_train, y_train, param_grid):
        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, y_train)
        print("Optimal Parameters:", grid.best_params_)
        return grid

# تحميل البيانات وتقسيمها
# يجب استبدال 'path_to_Heart.csv' بمسار الملف الفعلي لبيانات أمراض القلب
# يفضل استخدام pandas لتحميل البيانات
data = pd.read_csv('C:\\Suliman_Algaramanli\\study\\CS461\\HW\\4\\Heart.csv')

# تحقق من أسماء الأعمدة في بياناتك باستخدام data.columns
# افترض أن اسم العمود الذي يحتوي على البيانات المستهدفة هو 'target'
# يمكنك استبدال 'target' بالاسم الصحيح إذا كان مختلفًا في بياناتك
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# 1. تجربة قيم مختلفة لمعلمة C
C_values = [1, 50, 100]
for C_val in C_values:
    svm_model = SVMClassifier(C=C_val)
    svm_model.train(X_train, y_train)
    predictions = svm_model.predict(X_test)
    accuracy = svm_model.model.score(X_test, y_test)
    print(f"Accuracy with C={C_val}: {accuracy}")

# 2. تقرير الدقة باستخدام أنواع مختلفة من النواة
kernel_values = ['linear', 'rbf']
for kernel_val in kernel_values:
    svm_model = SVMClassifier(kernel=kernel_val)
    svm_model.train(X_train, y_train)
    predictions = svm_model.predict(X_test)
    accuracy = svm_model.model.score(X_test, y_test)
    print(f"Accuracy with kernel={kernel_val}: {accuracy}")

# 3. ضبط معلمات SVM باستخدام GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
svm_model = SVMClassifier()
grid_model = svm_model.grid_search(X_train, y_train, param_grid)

# استخدام النموذج المُعدل للتنبؤ وطباعة التقرير
grid_predictions = grid_model.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
"""
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
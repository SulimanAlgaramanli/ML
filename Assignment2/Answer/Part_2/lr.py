import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/sulim/Downloads/hours_grades.csv')

def read_dataset():
    x = np.array(df.hours)
    y = np.array(df.grade)
    return x,y

def predict_using_sklearn():
    r = LinearRegression()
    r.fit(df[['hours']],df.grade)
    return r.coef_, r.intercept_

Cost_fun = []
def gradient_descent(x,y):
    b_curr = 0      #theta_0    a    c    Intercept
    m_curr = 0      #theta_1    b     w    Coefficient
    
    #iterations = 2000000
    #learning_rate = 0.0004223

    iterations = 400000
    learning_rate = 0.0004223
    n = len(x)

    for i in range(iterations):
        # توقعات النموذج باستخدام المعاملات الحالية
        h_x = b_curr + m_curr*x
        #  حساب متوسط مربعات الخطأ (MSE)
        mse = 1 / (2 * n) * np.sum((h_x - y)**2)
        Cost_fun.append(mse)
        #حساب المشتقات الجزئية لمعاملات الانحدار
        derivative_b_curr = (1/n) * np.sum(h_x - y)
        derivative_m_curr = (1/n) * np.sum((h_x - y) * x)
        #تحديث المعاملات باستخدام معدل التعلم والمشتقات الحالية
        b_curr -= learning_rate * derivative_b_curr
        m_curr -= learning_rate * derivative_m_curr
    return b_curr,m_curr

if __name__ == "__main__":
    x,y=read_dataset()
    #print(x,y)
    m_sklearn, b_sklearn = predict_using_sklearn()
    #print()
    print('Using sklearn :\t\tCoef {} \t Intercept {}'.format(m_sklearn,[b_sklearn]))
    b, m = gradient_descent(x,y)
    print('Using gradient descent: Coef {} Intercept{}'.format([m], [b]))
    print()
    # predict the grade for a new student
    new_x= 89

    #predict the grade with Using sklearn
    Prediction_Model = b_sklearn + m_sklearn*new_x
    print ("#Using sklearn#:\t\t If Hours: ",new_x,'\tGrade: ', Prediction_Model)

    #predict the grade with Using gradient descent
    Prediction_Model = b + m*new_x
    print ("#Using gradient descent#:\t If Hours: ",new_x,'\tGrade: ', [Prediction_Model])

    #Cost function
    print('Initial loss\t:', Cost_fun[0])
    print('Final loss\t:', Cost_fun[-1])


    #لرسم المعادلة وتمثيل البيانات
    plt.figure(figsize=(8,6))
    plt.title('Data distribution')
    plt.scatter(x, y)
    plt.xlabel('Hours')
    plt.ylabel('Grade')
    #plt.show()

    x_line = np.linspace(0,100,100)
    y_line = b + m*x_line
    plt.figure(figsize=(8,6))
    plt.title('Data distribution')
    plt.plot(x_line, y_line, c='r')
    plt.scatter(x, y, s=25)
    plt.xlabel('Hours')
    plt.ylabel('Grade')
    #plt.show()

    plt.figure(figsize=(8,6))
    plt.title('Cost values')
    plt.plot(Cost_fun[0:])
    plt.ylabel('Cost j')
    plt.xlabel('Number of iteration')
    plt.show()
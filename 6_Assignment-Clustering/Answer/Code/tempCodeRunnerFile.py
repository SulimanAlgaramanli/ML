



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

class DataPlotter:
    """
    DataPlotter class: مسؤولة عن رسم جوانب مختلفة من مجموعة البيانات.
    """
    def __init__(self, df):
        """
        تهيئة فئة DataPlotter مع DataFrame.
        """
        self.df = df

    def plot_dataset_before_clustering(self):
        """
        رسم مجموعة البيانات قبل التجميع.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.df['X'], self.df['Y'], color='red')  # تغيير اللون إلى الأحمر
        plt.title('مجموعة البيانات قبل التجميع')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def plot_distance_matrix(self, distance_matrix):
        """
        رسم مصفوفة المسافات.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(distance_matrix, cmap='viridis')  # تغيير اللون إلى الأخضر
        plt.colorbar(label='المسافة')
        plt.title('مصفوفة المسافات')
        plt.xticks(np.arange(len(self.df)), self.df['Point'])
        plt.yticks(np.arange(len(self.df)), self.df['Point'])

        for i in range(len(self.df)):
            for j in range(len(self.df)):
                plt.text(j, i, f'{distance_matrix[i, j]:.2f}', ha='center', va='center', color='darkblue')  # تغيير اللون إلى الأزرق الداكن

        plt.show()

    def plot_clusters(self, centroids, labels, n_clusters):
        """
        رسم التجمعات بعد تطبيق تجميع k-means.
        """
        for iteration in range(3):
            plt.figure(figsize=(8, 6))
            for cluster in range(n_clusters):
                cluster_points = self.df.iloc[labels == cluster]
                plt.scatter(cluster_points['X'], cluster_points['Y'], label=f'التجمع {cluster + 1}')
            plt.scatter(centroids[:, 0], centroids[:, 1], color='blue', marker='X', label='مراكز التجميع')  # تغيير اللون إلى الأزرق
            plt.title(f'تجميع k-means التكرار {iteration + 1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()

            plt.show()

    def plot_hierarchical_clustering(self, Z, method):
        """
        رسم شجرة التجميع الهرمي.
        """
        plt.figure(figsize=(8, 6))
        dendrogram(Z)
        plt.title(f'تجميع هرمي ({method.capitalize()} الربط)')
        plt.xlabel('فهرس العينة')
        plt.ylabel('المسافة')
        plt.show()

class ClusteringCalculator:
    """
    ClusteringCalculator class: تتعامل مع حسابات التجميع.
    """
    def __init__(self, df):
        """
        تهيئة فئة ClusteringCalculator مع DataFrame.
        """
        self.df = df

    def calculate_distance_matrix(self):
        """
        حساب مصفوفة المسافات.
        """
        distance_matrix = np.zeros((len(self.df), len(self.df)))

        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        for i, row1 in self.df.iterrows():
            for j, row2 in self.df.iterrows():
                distance_matrix[i, j] = euclidean_distance((row1['X'], row1['Y']), (row2['X'], row2['Y']))

        return distance_matrix

    def kmeans_clustering(self, n_clusters, centroids, n_iterations=3, convergence_threshold=1e-4):
        """
        تطبيق تجميع k-means.
        """
        for iteration in range(n_iterations):
            kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
            labels = kmeans.fit_predict(self.df[['X', 'Y']])
            new_centroids = kmeans.cluster_centers_

            if np.allclose(centroids, new_centroids, atol=convergence_threshold):
                print(f"تم الانتهاء بعد {iteration + 1} تكرارات.")
                break

            centroids = new_centroids

        return centroids, labels

    def hierarchical_clustering(self, method):
        """
        تطبيق تجميع هرمي.
        """
        X = self.df[['X', 'Y']].values
        Z = linkage(X, method=method)
        return Z

# برنامج التشغيل
if __name__ == "__main__":
    # تحميل مجموعة البيانات
    file_path = 'C:\\Users\\sulim\\Downloads\\MyGitHub\\ML\\6_Assignment-Clustering\\Answer\\Code\\myData.csv'
    df = pd.read_csv(file_path)

    # إنشاء مثيلات لـ DataPlotter و ClusteringCalculator
    plotter = DataPlotter(df)
    calculator = ClusteringCalculator(df)

    # رسم مجموعة البيانات قبل التجميع
    plotter.plot_dataset_before_clustering()

    # حساب ورسم مصفوفة المسافات
    distance_matrix = calculator.calculate_distance_matrix()
    plotter.plot_distance_matrix(distance_matrix)

    # تطبيق تجميع k-means بمعلمات محددة
    n_clusters = 3
    initial_centroids = df.loc[df['Point'].isin(['K1', 'K4', 'K7']), ['X', 'Y']].values
    centroids, labels = calculator.kmeans_clustering(n_clusters, initial_centroids)

    # رسم التجميعات
    plotter.plot_clusters(centroids, labels, n_clusters)

    # تطبيق تجميع هرمي بطرق مختلفة
    methods = ['single', 'complete', 'average']
    for method in methods:
        Z = calculator.hierarchical_clustering(method)
        plotter.plot_hierarchical_clustering(Z, method)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

#يتضمن وظائف لتنفيذ عمليات التجميع ورسم البيانات والنتائج
class Clustering: 
    
    def __init__(self, df):        
        self.df = df

    # رسم مجموعة البيانات قبل التجميع
    def plot_dataset_before_clustering(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.df['X'], self.df['Y'], color='red')
        plt.title('Dataset Before Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    # رسم مصفوفة المسافات
    def plot_distance_matrix(self, distance_matrix):
        plt.figure(figsize=(8, 6))
        plt.imshow(distance_matrix, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.title('Distance Matrix')
        plt.xticks(np.arange(len(self.df)), self.df['Point'])
        plt.yticks(np.arange(len(self.df)), self.df['Point'])

        for i in range(len(self.df)):
            for j in range(len(self.df)):
                plt.text(j, i, f'{distance_matrix[i, j]:.2f}', ha='center', va='center', color='darkblue')  
        plt.show()

    # k-means رسم التجميعات بعد تطبيق تجميع  
    def plot_clusters(self, centroids, labels, n_clusters):
        for iteration in range(3):
            plt.figure(figsize=(8, 6))
            for cluster in range(n_clusters):
                cluster_points = self.df.iloc[labels == cluster]
                plt.scatter(cluster_points['X'], cluster_points['Y'], label=f'Cluster {cluster + 1}')
            plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', label='Centroids')
            plt.title(f'K-Means Clustering Iteration {iteration + 1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.show()

    # رسم شجرة التجميع الهرمي
    def plot_hierarchical_clustering(self, Z, method):
        plt.figure(figsize=(8, 6))
        dendrogram(Z)
        plt.title(f'Hierarchical Clustering ({method.capitalize()} Linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

    # حساب مصفوفة المسافات
    def calculate_distance_matrix(self):        
        distance_matrix = np.zeros((len(self.df), len(self.df)))

        # في الفضاء p1 و p2 حساب المسافة اليوروكليدية بين نقطتين 
        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        for i, row1 in self.df.iterrows():
            for j, row2 in self.df.iterrows():
                distance_matrix[i, j] = euclidean_distance((row1['X'], row1['Y']), (row2['X'], row2['Y']))

        return distance_matrix

    # k-means تطبيق تجميع
    def kmeans_clustering(self, n_clusters, centroids, n_iterations=3, convergence_threshold=1e-4):

        for iteration in range(n_iterations):
            kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
            labels = kmeans.fit_predict(self.df[['X', 'Y']])
            new_centroids = kmeans.cluster_centers_

            if np.allclose(centroids, new_centroids, atol=convergence_threshold):
                print(f"Converged after {iteration + 1} iterations.")
                break

            centroids = new_centroids

        return centroids, labels

    # تطبيق تجميع هرمي
    def hierarchical_clustering(self, method):
        X = self.df[['X', 'Y']].values
        Z = linkage(X, method=method)
        return Z
    
    # elbow للعثور على العدد الأمثل للمجموعات باستخدام طريقة elbow curve حساب
    def calculate_elbow(self, max_clusters=10):
        # clusters قائمة لتخزين التكاليف لعدد مختلف من 
        costs = []  
        
        for num_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(self.df[['X', 'Y']])
            #لها cluster هو مجموع المسافات المربعة للعينات إلى أقرب مركز Inertia 
            costs.append(kmeans.inertia_)  
            
        # elbow رسم منحنى 
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), costs, marker='o')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Cost')
        plt.xticks(range(1, max_clusters + 1))
        plt.grid(True)
        plt.show()

# البرنامج الرئيسي
if __name__ == "__main__":
    # Load dataset
    file_path = 'C:\\Users\\sulim\\Downloads\\MyGitHub\\ML\\6_Assignment-Clustering\\Answer\\Code\\myData.csv'
    df = pd.read_csv(file_path)

    clustering = Clustering(df)

    # رسم مجموعة البيانات قبل التجميع
    clustering.plot_dataset_before_clustering()

    # حساب ورسم مصفوفة المسافات
    distance_matrix = clustering.calculate_distance_matrix()
    clustering.plot_distance_matrix(distance_matrix)

    # تطبيق تجميع k-means بمعلمات محددة
    n_clusters = 3
    initial_centroids = df.loc[df['Point'].isin(['K1', 'K4', 'K7']), ['X', 'Y']].values
    centroids, labels = clustering.kmeans_clustering(n_clusters, initial_centroids)

    # رسم التجميعات
    clustering.plot_clusters(centroids, labels, n_clusters)

    # تطبيق تجميع هرمي بطرق مختلفة
    methods = ['single', 'complete', 'average']
    for method in methods:
        Z = clustering.hierarchical_clustering(method)
        clustering.plot_hierarchical_clustering(Z, method)

    # elbow حساب ورسم منحنى 
    clustering.calculate_elbow(max_clusters=8)

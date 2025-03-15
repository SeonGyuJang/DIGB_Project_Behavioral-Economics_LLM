## 최적 군집 개수 : 4개로 확인됨.
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 임베딩값 불러오기(JSON)
with open("/Users/jangseongyu/Desktop/DIGB_Project/persona_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings = [item["embedding"] for item in data]
X = np.array(embeddings)  # (샘플 수, 임베딩 차원)

# 2. Elbow & Silhouette 값 계산
sum_of_squared_distances = [] 
silhouette_scores = []        
K_range = range(2, 11)   

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    sum_of_squared_distances.append(kmeans.inertia_)

    labels = kmeans.labels_
    sil_score = silhouette_score(X, labels)
    silhouette_scores.append(sil_score)

# 3. Elbow Plot 시각화
plt.figure(figsize=(6, 4))
plt.plot(list(K_range), sum_of_squared_distances, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (Sum of squared distances)")
plt.title("Elbow Method for Optimal k")
plt.savefig(r"elbow_plot.png")
plt.show()

# 4. Silhouette Plot 시각화
plt.figure(figsize=(6, 4))
plt.plot(list(K_range), silhouette_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal k")
plt.savefig(r"silhouette_plot.png")
plt.show()

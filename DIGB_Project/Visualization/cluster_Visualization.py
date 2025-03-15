import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from adjustText import adjust_text

# 파일 경로 설정
file_paths = {
    "PCA": r"C:\Users\dsng3\Desktop\DIGB_Project\PCA\persona_clusters_pca.json",
    "TSNE": r"C:\Users\dsng3\Desktop\DIGB_Project\TSNE\persona_clusters_tsne.json",
    "UMAP": r"C:\Users\dsng3\Desktop\DIGB_Project\UMAP\persona_clusters_umap.json"
}

# 군집 키워드 정의
cluster_keywords = {
    "PCA": {
        0: "Education, Technology, Students, Language, AI",
        1: "Energy, Materials, Water, Gas, Technology",
        2: "History, Historian, Ancient, Medieval, Geography",
        3: "Animals, Species, Environment, Science, Biology"
    },
    "TSNE": {
        0: "Education, Technology, Students, Language, AI",
        1: "Energy, Materials, Water, Gas, Technology",
        2: "History, Historian, Ancient, Medieval, Geography",
        3: "Animals, Species, Environment, Science, Biology"
    },
    "UMAP": {
        0: "Species, Animals, Biology, Impact, Genetic",
        1: "Energy, Materials, Gas, Drilling, Production",
        2: "Health, Education, Students, Nutrition, Monitoring",
        3: "History, Historian, Medieval, Language, Ancient"
    }
}

# 색상 설정
colors = ["red", "blue", "green", "orange"]
cmap = ListedColormap(colors)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

data_results = {}

for i, (method, path) in enumerate(file_paths.items()):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pca1_values, pca2_values, cluster_labels = [], [], []
    
    for persona in data:
        if method == "PCA":
            pca1_values.append(persona["pca1"])
            pca2_values.append(persona["pca2"])
        elif method == "TSNE":
            pca1_values.append(persona["tsne1"])
            pca2_values.append(persona["tsne2"])
        elif method == "UMAP":
            pca1_values.append(persona["umap1"])
            pca2_values.append(persona["umap2"])
        cluster_labels.append(persona["cluster_label"])
    
    pca1_values = np.array(pca1_values)
    pca2_values = np.array(pca2_values)
    cluster_labels = np.array(cluster_labels)
    
    silhouette_avg = silhouette_score(np.column_stack((pca1_values, pca2_values)), cluster_labels)
    data_results[method] = silhouette_avg
    
    ax = axes[i]
    ax.set_title(f"{method} (Silhouette Score: {silhouette_avg:.2f})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    
    # KNN을 활용한 군집 경계 시각화
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(np.column_stack((pca1_values, pca2_values)), cluster_labels)
    x_min, x_max = pca1_values.min() - 0.5, pca1_values.max() + 0.5
    y_min, y_max = pca2_values.min() - 0.5, pca2_values.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.2)
    
    for cluster_id in np.unique(cluster_labels):
        idx = cluster_labels == cluster_id
        ax.scatter(pca1_values[idx], pca2_values[idx], c=colors[cluster_id], label=f"Cluster {cluster_id} ({sum(idx)})", edgecolors="k")
    
    texts = []
    for cluster_id in np.unique(cluster_labels):
        cluster_idx = np.where(cluster_labels == cluster_id)
        cluster_center_x = np.mean(pca1_values[cluster_idx])
        cluster_center_y = np.mean(pca2_values[cluster_idx])
        keywords = cluster_keywords[method][cluster_id]
        text = ax.text(cluster_center_x, cluster_center_y, keywords, fontsize=10, ha="center", color=colors[cluster_id], bbox=dict(facecolor='white', alpha=0.7))
        texts.append(text)
    
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    ax.legend()
    ax.grid(True, linestyle='dotted')

plt.suptitle("Cluster Visualization using PCA, TSNE, and UMAP with Silhouette Scores", fontsize=14)
plt.tight_layout()
plt.savefig("cluster_comparison.png")
plt.show()
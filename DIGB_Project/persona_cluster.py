# python persona_cluster.py --method pca --n_clusters 4
# python persona_cluster.py --method tsne --n_clusters 4
# python persona_cluster.py --method umap --n_clusters 4

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(input_path):
    """데이터 로드"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = [item["embedding"] for item in data]
    return data, np.array(embeddings)

def reduce_dimensions(X, method='pca', n_components=2):
    """방법론 정의"""
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    return X_reduced

def perform_clustering(X_reduced, n_clusters):
    """K-means 군집화 수행"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_reduced)
    return labels

def update_data(data, labels, X_reduced, method):
    """데이터 업데이트"""
    for i, item in enumerate(data):
        item["cluster_label"] = int(labels[i])
        item[f"{method}1"] = float(X_reduced[i, 0])
        item[f"{method}2"] = float(X_reduced[i, 1])
    return data

def save_results(data, output_path, X_reduced, labels, n_clusters, method, plot_path):
    """결과 저장 및 시각화"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    silhouette_avg = silhouette_score(X_reduced, labels)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.title(f"K-means Clustering ({method.upper()} 2D, k={n_clusters})")
    
    legend_text = [f"Silhouette Score: {silhouette_avg:.3f}"]
    plt.legend(legend_text, loc="best", fontsize=10, frameon=True)

    plt.colorbar(label="Cluster Label")
    plt.savefig(plot_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Persona Clustering with Dimension Reduction')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne', 'umap'],
                        help='Dimension reduction method (pca, tsne, or umap)')
    parser.add_argument('--n_clusters', type=int, default=4,
                        help='Number of clusters for K-means')
    parser.add_argument('--input_path', type=str, default='persona_embeddings.json',
                        help='Input JSON file path')
    
    args = parser.parse_args()
    
    method = args.method.lower()
    output_path = f"persona_clusters_{method}.json"
    plot_path = f"cluster_plot_{method}.png"
    
    # 데이터 로드
    data, X = load_data(args.input_path)
    
    # 차원 축소
    X_reduced = reduce_dimensions(X, method=method)
    
    # 군집화
    labels = perform_clustering(X_reduced, args.n_clusters)
    
    # 데이터 업데이트
    data = update_data(data, labels, X_reduced, method)
    
    # 결과 저장
    save_results(data, output_path, X_reduced, labels, args.n_clusters, method, plot_path)
    
    print(f"군집화가 완료되었습니다. 결과가 {output_path}에 저장되었습니다.")
    print(f"시각화 결과가 {plot_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()

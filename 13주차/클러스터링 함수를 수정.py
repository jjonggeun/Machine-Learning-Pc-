import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_means_clustering(data, K):
    np.random.shuffle(data)
    u = data[np.random.choice(data.shape[0], K, replace=False)]
    
    iteration_count = 0

    while True:
        iteration_count += 1
        clusters = [[] for _ in range(K)]

        for point in data:
            distances = []
            for j in range(K):
                distances.append(euclidean_distance(point, u[j]))
            closest_center = np.argmin(distances)
            clusters[closest_center].append(point)

        for i in range(K):
            clusters[i] = np.array(clusters[i])

        new_u = np.empty_like(u)
        for i in range(K):
            if len(clusters[i]) > 0:
                new_u[i] = np.mean(clusters[i], axis=0)
            else:
                new_u[i] = u[i]

        if np.allclose(u, new_u):
            break

        u = new_u

    return clusters, u, iteration_count

# 데이터 불러오기
fold_dir = "C:\\Users\\pc\\Desktop\\3학년\\1학기\\머러실\\Clustering_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()

# 원하는 K 값 설정
K = 4

# K-means Clustering 실행
clusters, final_centers, iteration_count = k_means_clustering(temp_data, K)

# 초기 중심에 대한 클러스터링 결과 계산
dists_initial = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_initial[:, i] = [euclidean_distance(point, final_centers[i]) for point in temp_data]
closest_initial = np.argmin(dists_initial, axis=1)

# 최종 클러스터링 결과 계산
dists_final = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_final[:, i] = [euclidean_distance(point, final_centers[i]) for point in temp_data]
closest_final = np.argmin(dists_final, axis=1)

# 결과 시각화
plt.figure(figsize=(18, 6))

# 초기 데이터 및 중심 시각화
plt.subplot(1, 3, 1)
plt.scatter(temp_data[:, 0], temp_data[:, 1], c='blue', s=20, label='Data Points')
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='red', marker='x', s=100, label='Final Centers')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Data Points and Initial Centers')
plt.legend()
plt.grid()

# 초기 중심에 대한 클러스터링 결과 시각화
colors = plt.cm.rainbow(np.linspace(0, 1, K))
plt.subplot(1, 3, 2)
for k in range(K):
    mask = closest_initial == k
    plt.scatter(temp_data[mask, 0], temp_data[mask, 1], c=colors[k], s=20, label=f'Cluster {k+1}')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Initial Clustered Data Points')
plt.legend()
plt.grid()

# 최종 클러스터링 결과 시각화
plt.subplot(1, 3, 3)
for k in range(K):
    mask = closest_final == k
    plt.scatter(temp_data[mask, 0], temp_data[mask, 1], c=colors[k], s=20, label=f'Cluster {k+1}')
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='black', marker='x', s=100, label='Final Centers')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Final Clustered Data Points')
plt.legend()
plt.grid()

plt.show()

print(f'K={K}, Number of iterations: {iteration_count}')

###############################################################################################
def elbow_method(data):
    em = []
    for k in range(1, 11): 
        clusters, _, _ = k_means_clustering(data, k)
        max_variance = 0
        for cluster in clusters:
            if len(cluster) > 0:
                variance = np.var(cluster, axis=0).max() 
                if variance > max_variance:
                    max_variance = variance
        em.append(max_variance)
    return em

# 엘보우 메소드를 사용하여 최적의 K 찾기
em_values = elbow_method(temp_data)

# 최적의 k 값 찾기
optimal_k = np.argmin(np.diff(em_values)) + 1

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), em_values, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Max Variance')
plt.title('Elbow Method for Optimal K using Max Variance')
plt.grid()
plt.show()


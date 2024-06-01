import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# L1 Norm (Manhattan 거리) 함수 정의
def l1_norm(point1, point2):
    return np.sum(np.abs(point1 - point2))

# L2 Norm (Euclidean 거리) 함수 정의
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))

def k_means_clustering(data, K):
    # 데이터를 무작위로 섞어줍니다.
    np.random.shuffle(data)

    # 초기 중심 설정: 데이터를 무작위로 섞은 후 처음 K개의 데이터를 초기 중심으로 선택합니다.
    center = data[:K]
    # 초기 중심을 u에
    u = center.copy()

    iteration_count = 0

    while True:
        # 반복이 몇번인지 계산
        iteration_count += 1
        # 각 클러스터를 저장할 빈 리스트
        clusters = [[] for _ in range(K)]

        # 각 포인트를 가장 가까운 중심에 할당합니다.
        for point in data:
            # 각 중심과의 거리를 계산, euclidean_distance
            distances = [euclidean_distance(point, u[j].reshape(1, -1)) for j in range(K)]
            # 각 중심과의 거리를 계산, l1_norm
            distances = [euclidean_distance(point.reshape(1, -1), u[j].reshape(1, -1)) for j in range(K)]
            # 가장 가까운 중심의 인덱스를 찾습니다.
            closest_center = np.argmin(distances)
            # 해당 중심에 포인트를 할당합니다.
            clusters[closest_center].append(point)

        
        for i in range(K):
            clusters[i] = np.array(clusters[i])

        # 새로운 중심
        new_u = np.empty_like(u)
        for i in range(K):
            if len(clusters[i]) > 0:
                # 클러스터의 평균을 새로운 중심
                new_u[i] = np.mean(clusters[i], axis=0)
            else:
                new_u[i] = u[i]

        # 중심이 변화하지 않으면 종료
        # 만약 k값이 작게 되면 그 값이 생각보다 일찍 종료될 수 있는데 이 때문에 그 결과가 달라질 수 있음. 
        if np.allclose(u, new_u):
            break

        # 새로운 중심으로 업데이트
        u = new_u

    # 최종 클러스터와 중심, 초기 중심, 반복 횟수를 반환합니다.
    return clusters, u, center, iteration_count

fold_dir = "C:\\Users\\pc\\Desktop\\3학년\\1학기\\머러실\\Clustering_data.csv"
temp_data = pd.read_csv(fold_dir)
temp_data = temp_data.to_numpy()


# 원하는 K 값 설정
K = 2

# K-means Clustering 실행
clusters, final_centers, initial_centers, iteration_count = k_means_clustering(temp_data, K)

# 초기 중심에 대한 클러스터링 결과 계산, 유클리드
dists_initial = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_initial[:, i] = euclidean_distance(temp_data, initial_centers[i].reshape(1, -1))
closest_initial = np.argmin(dists_initial, axis=1)

# 초기 중심에 대한 클러스터링 결과 계산, l1_norm
dists_initial = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_initial[:, i] = l1_norm(temp_data, initial_centers[i].reshape(1, -1))
closest_initial = np.argmin(dists_initial, axis=1)

# 최종 클러스터링 결과 계산, 유클리드
dists_final = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_final[:, i] = euclidean_distance(temp_data, final_centers[i].reshape(1, -1))
closest_final = np.argmin(dists_final, axis=1)

# 최종 클러스터링 결과 계산, l1_norm
dists_final = np.zeros((temp_data.shape[0], K))
for i in range(K):
    dists_final[:, i] = l1_norm(temp_data, final_centers[i].reshape(1, -1))
closest_final = np.argmin(dists_final, axis=1)

# 결과 시각화
plt.figure(figsize=(18, 6))

# 초기 데이터 및 중심 시각화
plt.subplot(1, 3, 1)
plt.scatter(temp_data[:, 0], temp_data[:, 1], c='blue', s=20, label='Data Points')
plt.scatter(initial_centers[:, 0], initial_centers[:, 1], c='red', marker='x', s=100, label='Initial Centers')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title(f'K={K}: Data Points')
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
        clusters, _, _, _ = k_means_clustering(data, k)
        max_variance = 0
        for cluster in clusters:
            if len(cluster) > 0:
                variance = np.var(cluster, axis=0).max()
                if variance > max_variance:
                    max_variance = variance
        em.append(max_variance)
    return em

# Elbow Method for finding the optimal K
em_values = elbow_method(temp_data)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), em_values, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Max Variance')
plt.title('Elbow Method for Optimal K using Max Variance')
plt.grid()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
fold_dir = "C:\\Users\\pc\\Desktop\\3학년\\1학기\\머러실\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()

# 데이터 분리
Wei = temp_data[:, 0]  # 무게 데이터
Len = temp_data[:, 1]  # 길이 데이터

# 그래프 그리기
plt.figure(figsize=(10, 6))

# 데이터 증강 및 노이즈 추가
def augment_data(data, factor=20, noise_range=0.3):
    augmented_data = []
    for i in range(len(data)):
        for _ in range(factor):
            augmented_value = data[i] + np.random.normal(0, noise_range)
            augmented_data.append(augmented_value)
    return np.array(augmented_data)

augmented_Wei = augment_data(Wei)
augmented_Len = augment_data(Len)

# 데이터 증강된 것을 먼저 그래프에 추가
plt.scatter(augmented_Wei, augmented_Len, color='red', alpha=0.5, marker='o', s=20, label='Augmented Data')  

# 원본 데이터 그리기
plt.scatter(Wei, Len, color='blue', label='Original Data', s=30)

plt.xlabel('Weight')
plt.ylabel('Length')
plt.title('Original Data & Augmented Data')
plt.legend()
plt.grid(True)
plt.show()

def split_data(data, train_ratio, val_ratio, test_ratio):
    # 데이터 사이즈 계산
    data_size = len(data)
    
    # Training set, Validation set, Test set 크기 계산
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)
    test_size = data_size - train_size - val_size

    # 데이터 인덱스를 랜덤하게 섞음
    shuffled_indices = np.random.permutation(data_size)

    # 데이터를 분할
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    # 분할된 데이터 생성
    train_data = data[train_indices]
    val_data = data[val_indices]
    test_data = data[test_indices]

    return train_data, val_data, test_data

# 데이터 분할
augmented_data = np.column_stack((augmented_Wei, augmented_Len))
train_set, val_set, test_set = split_data(augmented_data, 0.8, 0, 0.2)

# 그래프 그리기
plt.figure(figsize=(10, 6))

# Training set 플로팅
plt.scatter(train_set[:, 0], train_set[:, 1], color='blue',s=20, label='Training Set')

# Validation set 플로팅
plt.scatter(val_set[:, 0], val_set[:, 1], color='red',s=20, label='Validation Set')

# Test set 플로팅
plt.scatter(test_set[:, 0], test_set[:, 1], color='green',s=20, label='Test Set')

plt.xlabel('Weight')
plt.ylabel('Length')
plt.title('Split Dataset')
plt.legend()
plt.grid(True)
plt.show()

# 가우시안 기저 함수 정의
def gaussian_basis_function(X, K, k):
    x_min = X.min()  # 데이터의 최솟값
    x_max = X.max()  # 데이터의 최댓값
    mu = x_min + ((x_max - x_min) / (K - 1)) * k  # 각 가우시안 함수의 평균 계산
    
    v = (x_max - x_min) / (K - 1)  # 모든 가우스 함수의 분산
    
    simple = (X - mu) / v
    G = np.exp((-1/2) * (simple ** 2))
    
    return G

# 가중치 계산 함수
def calculate_weights(X, Y, K): 
    # k의 배열 생성
    k_values = np.arange(K).reshape(-1, 1)
    # K에 따른 가우시안 기저 함수 계산
    X_b = np.column_stack([gaussian_basis_function(X, K, k) for k in k_values])
    # bias 추가
    X_b = np.hstack([X_b, np.ones((len(X), 1))])
    # 가중치 계산 (K+1개의 가중치)
    weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y
    return weights

# MSE 계산 함수 수정
def mse(X, Y, K):
    # k의 배열 생성
    k_values = np.arange(K).reshape(-1, 1)
    # K에 따른 가우시안 기저 함수 계산
    X_b = np.column_stack([gaussian_basis_function(X, K, k) for k in k_values])
    # bias 추가
    X_b = np.hstack([X_b, np.ones((len(X), 1))])
    # 가중치 계산
    weights = calculate_weights(X, Y, K)
    # MSE 계산
    mse_value = np.mean(((X_b @ weights) - Y) ** 2)
    return mse_value

# K_values에 따른 MSE 계산
K_values = np.arange(1, 100)  # K 값 범위 설정
mse_values_training = [mse(train_set[:, 0], train_set[:, 1], K) for K in K_values]
mse_values_test = [mse(test_set[:, 0], test_set[:, 1], K) for K in K_values]

# MSE 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(K_values, mse_values_training, label='Training MSE')
plt.plot(K_values, mse_values_test, label='Test MSE')
plt.xlabel('Number of Basis Functions (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs. Number of Basis Functions')
plt.legend()
plt.grid(True)
plt.show()

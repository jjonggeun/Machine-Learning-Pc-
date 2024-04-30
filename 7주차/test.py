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

# 데이터 증강
augmented_Wei = []
augmented_Len = []
for i in range(len(Wei)):
    for _ in range(20):
        augmented_Wei.append(Wei[i] + np.random.normal(0, 0.3))  # 주변에 노이즈 추가
        augmented_Len.append(Len[i] + np.random.normal(0, 0.3))  # 주변에 노이즈 추가
        
# 증강된 데이터를 numpy 배열로 변환
augmented_data = np.column_stack((augmented_Wei, augmented_Len))

# 가우시안 기저 함수 정의
def gaussian_basis_function(X, K, k):
    x_min = X.min()  # 데이터의 최솟값
    x_max = X.max()  # 데이터의 최댓값
    mu = x_min + ((x_max - x_min) / (K - 1)) * k  # 각 가우시안 함수의 평균 계산
    
    v = (x_max - x_min) / (K - 1)  # 모든 가우스 함수의 분산
    
    simple = (X - mu) / v
    G = np.exp((-1/2) * (simple ** 2))
    
    return G

def split_augmented_data(augmented_data, train_ratio, val_ratio, test_ratio):

    # 비율의 합이 1인지 확인
    assert train_ratio + val_ratio + test_ratio == 1.0, "비율의 합이 1이어야 합니다."

    # 데이터의 총 개수
    total_samples = len(augmented_data)
    
    # 각 세트의 크기 계산
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # 데이터를 랜덤하게 섞음
    np.random.shuffle(augmented_data)

    # 데이터 분할
    train_set = augmented_data[:train_size]
    val_set = augmented_data[train_size:train_size + val_size]
    test_set = augmented_data[train_size + val_size:]

    return train_set, val_set, test_set

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

# 데이터 분할
train_set, val_set, test_set = split_augmented_data(augmented_data, 0.8, 0, 0.2)

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

# MSE 그래프 추가
K_values = np.arange(3, 11)  # K 값 범위 설정
train_mse_values = [mse(train_set[:, 0], train_set[:, 1], K) for K in K_values]
val_mse_values = [mse(val_set[:, 0], val_set[:, 1], K) for K in K_values]
test_mse_values = [mse(test_set[:, 0], test_set[:, 1], K) for K in K_values]

plt.figure(figsize=(10, 6))
plt.plot(K_values, train_mse_values, marker='o', label='Training Set MSE')
plt.plot(K_values, val_mse_values, marker='o', label='Validation Set MSE')
plt.plot(K_values, test_mse_values, marker='o', label='Test Set MSE')
plt.xlabel('Number of Basis Functions (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE for Different Number of Basis Functions')
plt.legend()
plt.grid(True)
plt.show()

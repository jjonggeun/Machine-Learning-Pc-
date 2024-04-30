import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
fold_dir = "C:\\Users\\pc\\Desktop\\3학년\\1학기\\머러실\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()   

x_data = temp_data[:, 0].reshape(-1, 1)
y_data = temp_data[:, 1].reshape(-1, 1)

# 노이즈 추가 과정
noise_range = 0.3
x_data_noise = []
y_data_noise = []

for x, y in zip(x_data, y_data) :
    for _ in range(19):
        noise_x =np.random.randn() * noise_range
        noise_y = np.random.randn() * noise_range
        x_data_noise.append(x + noise_x)
        y_data_noise.append(y + noise_y)

# 리스트를 NumPy 배열로 변환
x_data_noise = np.array(x_data_noise).reshape(-1, 1)
y_data_noise = np.array(y_data_noise).reshape(-1, 1)

plt.figure(figsize = [10,6])
plt.grid(True)
plt.scatter(x_data,y_data, c='r', label = 'origianl')
plt.scatter(x_data_noise,y_data_noise,alpha = 0.1, c='b', label = 'augmented')
plt.legend()

def split_data(x_data, y_data, train_ratio, val_ratio, test_ratio):
    # 데이터 사이즈 계산
    data_size = len(x_data)
    
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
    x_train = x_data[train_indices].reshape(-1, 1)
    y_train = y_data[train_indices].reshape(-1, 1)
    x_val = x_data[val_indices].reshape(-1, 1)
    y_val = y_data[val_indices].reshape(-1, 1)
    x_test = x_data[test_indices].reshape(-1, 1)
    y_test = y_data[test_indices].reshape(-1, 1)

    return x_train, y_train, x_val, y_val, x_test, y_test

# 데이터 분할 비율 설정
train_ratio = 0.8
val_ratio = 0.0
test_ratio = 0.2

# 데이터 분할
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_data_noise, y_data_noise, train_ratio, val_ratio, test_ratio)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, c='r', label='Training Set')
plt.scatter(x_val, y_val, c='g', label='Validation Set')
plt.scatter(x_test, y_test, c='b', label='Test Set')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Split')
plt.legend()
plt.grid(True)
plt.show()



# 가우시안 기저 함수 정의 #x, uk, 분포
def gaussian_basis_function(x, uk, mean):
    return np.exp(-0.5 * ((x - uk) / mean) ** 2)

# 평균 제곱 오차(MSE) 계산 함수
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

K_range = range(2, 50, 1)
MSE_train = []
MSE_test = []

for K in K_range:
    k = np.arange(0, K, 1)
    std = (max(x_data) - min(x_data)) / (K - 1)
    uk = np.zeros(K)
    
    for i in range(K):
        uk[i] = min(x_data) + ((max(x_data) - min(x_data)) / (K - 1)) * k[i]
            
    # 훈련 세트를 사용하여 모델 훈련
    Phi_train = np.zeros((len(x_train), K))
    dummy_data = np.ones((len(x_train), 1))
    Phi_train = np.column_stack((Phi_train, dummy_data))

    for i in range(len(x_train)):
        for j in range(K):
            Phi_train[i, j] = gaussian_basis_function(x_train[i, 0], uk[j], std)

    # Analytic Solution
    w = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ y_train
    
    # 훈련 세트에 대한 예측값 계산
    y_pred_train = Phi_train @ w
    
      # 훈련 세트에 대한 MSE 계산 및 저장
    mse_train = calculate_mse(y_train, y_pred_train)
    MSE_train.append(mse_train)
    
    # 테스트 세트에 대한 예측값 계산
    Phi_test = np.zeros((len(x_test), K))
    dummy_data = np.ones((len(x_test), 1))
    Phi_test = np.column_stack((Phi_test, dummy_data))

    for i in range(len(x_test)):
        for j in range(K):
            Phi_test[i, j] = gaussian_basis_function(x_test[i, 0], uk[j], std)

    # 테스트 세트에 대한 예측값 계산
    y_pred_test = Phi_test @ w
    
      # 테스트 세트에 대한 MSE 계산 및 저장
    mse_test = calculate_mse(y_test, y_pred_test)
    MSE_test.append(mse_test)

plt.figure(figsize=(10, 6))
plt.plot(K_range, MSE_train, label='Train MSE')
plt.plot(K_range, MSE_test, label='Test MSE')
plt.xlabel('K')
plt.ylabel('MSE')
plt.title('MSE for Training and Test Sets')
plt.legend()
plt.show()
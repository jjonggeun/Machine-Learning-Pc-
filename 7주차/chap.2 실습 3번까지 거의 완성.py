import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
fold_dir = "C:\\Users\\pc\\Desktop\\3학년\\1학기\\머러실\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
temp_data = temp_data.to_numpy()

# 데이터 분리
Wei = temp_data[:, 0]  # 무게 데이터
Len = temp_data[:, 1]  # 길이 데이터

# 데이터 증강
augmented_Wei = []
augmented_Len = []
for i in range(len(Wei)):
    for _ in range(20):
        augmented_Wei.append(Wei[i] + np.random.normal(0, 0.3))  # 주변에 노이즈 추가
        augmented_Len.append(Len[i] + np.random.normal(0, 0.3))  # 주변에 노이즈 추가
        
# 증강된 데이터를 numpy 배열로 변환
augmented_data = np.column_stack((augmented_Wei, augmented_Len))

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

def gaussian_basis_regression(X, y, k):
    # Ridge regression 모델 생성
    model = make_pipeline(Ridge(alpha=1.0))
    
    # 가우시안 기저 함수를 사용하여 특징을 변환
    X_transformed = np.column_stack([np.exp(-((X - mu) ** 2) / (2.0 * (0.3 ** 2))) for mu in np.linspace(min(X), max(X), k)])
    
    # 모델 피팅
    model.fit(X_transformed, y)
    
    return model

# k값 범위 설정
k_values = range(1, 100)

# training set과 test set에 대한 오류 저장할 리스트
train_errors = []
test_errors = []

# 모델의 각 k값에 대한 학습 및 테스트 오류 계산
for k in k_values:
    # 데이터 분할
    train_set, val_set, test_set = split_augmented_data(augmented_data, 0.5, 0.3, 0.2)
    
    # 모델 피팅
    model = gaussian_basis_regression(train_set[:, 0], train_set[:, 1], k)
    
    # Training set 예측 및 오류 계산
    train_predicted = model.predict(np.column_stack([np.exp(-((train_set[:, 0] - mu) ** 2) / (2.0 * (0.3 ** 2))) for mu in np.linspace(min(train_set[:, 0]), max(train_set[:, 0]), k)]))
    train_error = mean_squared_error(train_set[:, 1], train_predicted)
    train_errors.append(train_error)
    
    # Test set 예측 및 오류 계산
    test_predicted = model.predict(np.column_stack([np.exp(-((test_set[:, 0] - mu) ** 2) / (2.0 * (0.3 ** 2))) for mu in np.linspace(min(test_set[:, 0]), max(test_set[:, 0]), k)]))
    test_error = mean_squared_error(test_set[:, 1], test_predicted)
    test_errors.append(test_error)

# 최적의 k 값 찾기
optimal_k = k_values[np.argmin(test_errors)]

# 그래프 그리기
plt.figure(figsize=(10, 6))

# training error 그래프
plt.plot(k_values, train_errors, label='Training Error', color='blue')

# test error 그래프
plt.plot(k_values, test_errors, label='Test Error', color='red')

plt.xlabel('Number of Basis Functions (k)')
plt.ylabel('MSE')
plt.title('Hold-Out Validation')
plt.axvline(x=optimal_k, color='green', linestyle='--', label='Optimal k')
plt.legend()
plt.grid(True)
plt.show()

print("Optimal k:", optimal_k)

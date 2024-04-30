import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 데이터 불러오기
fold_dir = "C:\\Users\\pc\\Desktop\\3학년\\1학기\\머러실\\lin_regression_data_01.csv"
temp_data = pd.read_csv(fold_dir, header=None)
Wei, Len = temp_data[0].values, temp_data[1].values

# 데이터 증강
augmented_data = []
for i in range(len(Wei)):
    for _ in range(20):
        augmented_data.append([Wei[i] + np.random.normal(0, 0.3), Len[i] + np.random.normal(0, 0.3)])

augmented_data = np.array(augmented_data)

def gaussian_basis_regression(X, y, k):
    model = make_pipeline(Ridge(alpha=1.0))
    X_transformed = np.column_stack([np.exp(-((X - mu) ** 2) / (2.0 * (0.3 ** 2))) for mu in np.linspace(min(X), max(X), k)])
    model.fit(X_transformed, y)
    return model

# k값 범위 설정
k_values = range(1, 100)

# training set과 test set에 대한 오류 저장할 리스트
train_errors = []
test_errors = []

# 모델의 각 k값에 대한 학습 및 테스트 오류 계산
for k in k_values:
    train_set, test_set = train_test_split(augmented_data, test_size=0.2)
    model = gaussian_basis_regression(train_set[:, 0], train_set[:, 1], k)
    train_predicted = model.predict(np.column_stack([np.exp(-((train_set[:, 0] - mu) ** 2) / (2.0 * (0.3 ** 2))) for mu in np.linspace(min(train_set[:, 0]), max(train_set[:, 0]), k)]))
    train_error = mean_squared_error(train_set[:, 1], train_predicted)
    train_errors.append(train_error)
    test_predicted = model.predict(np.column_stack([np.exp(-((test_set[:, 0] - mu) ** 2) / (2.0 * (0.3 ** 2))) for mu in np.linspace(min(test_set[:, 0]), max(test_set[:, 0]), k)]))
    test_error = mean_squared_error(test_set[:, 1], test_predicted)
    test_errors.append(test_error)

# 최적의 k 값 찾기
optimal_k = k_values[np.argmin(test_errors)]

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_errors, label='Training Error', color='blue')
plt.plot(k_values, test_errors, label='Test Error', color='red')
plt.xlabel('Number of Basis Functions (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training and Test Errors vs. Number of Basis Functions')
plt.axvline(x=optimal_k, color='green', linestyle='--', label='Optimal k')
plt.legend()
plt.grid(True)
plt.show()

print("Optimal k:", optimal_k)

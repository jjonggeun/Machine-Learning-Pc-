import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 시그모이드 함수 선언
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 순전파 함수
def forward_propagation(x_with_dummy, v, w):
    A = v @ x_with_dummy.T
    b = sigmoid(A)
    b_with_dummy = np.vstack([b, np.ones([1, len(x_with_dummy)])])
    B = w @ b_with_dummy
    y_hat = sigmoid(B)
    return A, b, b_with_dummy, B, y_hat

# 역전파 함수
def backward_propagation(x_with_dummy, y_one_hot, A, b, b_with_dummy, B, y_hat, v, w):
    error = y_hat - y_one_hot.T
    wmse = (error * sigmoid_derivative(B)) @ b_with_dummy.T / len(x_with_dummy)
    vmse = ((w[:, :-1].T @ (error * sigmoid_derivative(B))) * sigmoid_derivative(A)) @ x_with_dummy / len(x_with_dummy)
    return wmse, vmse

# 데이터 분할 함수
def aug_data(data, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    np.random.shuffle(data)
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set, test_set

# confusion matrix 계산 함수
def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        row_index = int(y_pred[i])
        col_index = int(y_true[i])
        confusion_matrix[row_index, col_index] += 1
    return confusion_matrix

def preprocess_data(df):
    df['gender'] = df['gender'].map({'female': 1, 'male': 2})
    df['a neurological disorder'] = df['a neurological disorder'].map({'yes': 1, 'no': 0})
    df['heart disease'] = df['heart disease'].map({'yes': 1, 'no': 0})
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
    return df

# 데이터 불러오기
fold_dir = "C:\\Users\\pc\\Desktop\\3학년\\1학기\\머러실\\heart_disease_new.csv"
temp_data = pd.read_csv(fold_dir)

# 전처리
temp_data = preprocess_data(temp_data)

# Numpy 배열로 변환
temp_data = temp_data.to_numpy()

# 데이터 분포 확인
print("Class distribution in the dataset:", np.bincount(temp_data[:, -1].astype(int)))

# 데이터 분할
train_data, test_data = aug_data(temp_data, 0.7, 0.3)

# 데이터 분리
x_train = train_data[:, :-1]
y_train = train_data[:, -1].reshape(-1, 1)

x_test = test_data[:, :-1]
y_test = test_data[:, -1].reshape(-1, 1)

# 입력 속성 수와 출력 클래스 수 추출
M = x_train.shape[1]
output_size = len(np.unique(y_train))

# hidden layer의 노드 수
hidden_size = 5

# weight 초기화 (작은 값으로 설정)
v = np.random.randn(hidden_size, M + 1) * 0.01
w = np.random.randn(output_size, hidden_size + 1) * 0.01

# 학습 파라미터 설정
learning_rate = 0.1
epochs = 100

# One-Hot Encoding
y_train_one_hot = np.zeros((len(y_train), output_size))
for i in range(len(y_train)):
    y_train_one_hot[i, int(y_train[i])] = 1

# 데이터에 더미 변수 추가
x_train_with_dummy = np.hstack((x_train, np.ones((len(x_train), 1))))
x_test_with_dummy = np.hstack((x_test, np.ones((len(x_test), 1))))
total_samples = len(x_train)

# 정확도와 MSE를 저장할 리스트 초기화
accuracy_list = []
mse_list = []

# 최적의 가중치를 저장할 변수 초기화
best_accuracy = 0
best_v = v
best_w = w

# 학습
for epoch in range(epochs):
    for step in range(total_samples):
        A, b, b_with_dummy, B, y_hat = forward_propagation(x_train_with_dummy[step:step+1], v, w)
        wmse, vmse = backward_propagation(x_train_with_dummy[step:step+1], y_train_one_hot[step:step+1], A, b, b_with_dummy, B, y_hat, v, w)
        w -= learning_rate * wmse
        v -= learning_rate * vmse
    
    # 테스트 데이터에 대해 정확도 계산
    A_test, b_test, b_with_dummy_test, B_test, y_hat_test = forward_propagation(x_test_with_dummy, v, w)
    y_hat_test_index = np.argmax(y_hat_test, axis=0)
    test_accuracy = np.mean(y_hat_test_index == y_test.flatten())
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_v = np.copy(v)
        best_w = np.copy(w)

    A_train, b_train, b_with_dummy_train, B_train, y_hat_train = forward_propagation(x_train_with_dummy, v, w)
    predicted_labels = np.argmax(y_hat_train, axis=0)
    accuracy = np.mean(predicted_labels == y_train.flatten())
    accuracy_list.append(accuracy)
    
    mse = np.mean((y_hat_train - y_train_one_hot.T) ** 2)
    mse_list.append(mse)
    


# 최적의 가중치로 모델 업데이트
v = best_v
w = best_w

# confusion matrix 계산
confusion_matrix = compute_confusion_matrix(y_test.flatten(), y_hat_test_index, output_size)

print("Confusion Matrix:")
print(confusion_matrix)

# 그래프 출력
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), accuracy_list, label='Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), mse_list, label='MSE', color='red')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over epochs')
plt.legend()
plt.grid()
plt.ylim(0, 1)

plt.show()

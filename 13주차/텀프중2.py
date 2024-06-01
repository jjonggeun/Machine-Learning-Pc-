import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(df):
    # gender에서 female을 1, male을 2로 변경
    df['gender'] = df['gender'].map({'female': 1, 'male': 2})
    
    # a neurological disorder에서 yes를 1, no를 0으로 변경
    df['a neurological disorder'] = df['a neurological disorder'].map({'yes': 1, 'no': 0})
    
    # heart disease에서 yes를 1, no를 0으로 변경
    df['heart disease'] = df['heart disease'].map({'yes': 1, 'no': 0})
    
    # 각 열에 대한 nan값을 해당 열의 평균으로 대체
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

# 결과 확인
print(temp_data[:5])

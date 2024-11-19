import pandas as pd
import numpy as np
import os


def fisher_score(X, y):
    class_labels = y.unique()
    fisher_scores = []

    for column in X.columns:  # 遍历每个特征列
        feature_values = X[column]
        feature_scores = []

        for label in class_labels:
            # 计算每个类别的均值和方差
            feature_values_class = feature_values[y == label]
            mean = np.mean(feature_values_class)
            var = np.var(feature_values_class)
            feature_scores.append(len(feature_values_class) * (mean - np.mean(feature_values)) ** 2 / var)

        fisher_scores.append(sum(feature_scores))

    # 创建包含特征和对应得分的 DataFrame
    scores_df = pd.DataFrame({'Feature': X.columns, 'Fisher_Score': fisher_scores})

    # 按 Fisher score 从高到低排序
    scores_df = scores_df.sort_values(by='Fisher_Score', ascending=False)

    return scores_df


data = pd.read_csv(r'D:\jupyter\new_funtion\2024_7_9\7_30\mRNA_LncRNA_Multiple_Sclerose_data.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 计算 Fisher 得分
fisher_scores_df = fisher_score(X, y)

# 将结果保存到 CSV 文件
output_directory = r'D:\jupyter\new_funtion\2024_7_9\7_30\duibi'
os.makedirs(output_directory, exist_ok=True)
fisher_scores_df.to_csv(os.path.join(output_directory, 'fsc_Multiple_Sclerose_data.csv'), index=False)

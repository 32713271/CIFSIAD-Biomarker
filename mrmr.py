import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

import time

start_time = time.time()


def plot_score(score):
    plt.figure(figsize=(16, 8))
    plt.plot(score)
    plt.xlabel('Number of feature')
    plt.ylabel('score')
    plt.show()


def add_max_score_to_list(temp_scores, current_score, selected_indices, selected_indices_list):
    max_score_index = np.argmax(np.array(temp_scores))
    current_score.append(temp_scores[max_score_index])
    selected_indices.add(max_score_index)
    selected_indices_list.append(max_score_index)


def FCQ(X, y, k):
    num_features = len(X[0])
    f_test_scores = [f_oneway(X[:, i], y)[0] for i in range(num_features)]

    start_feature_index = random.randint(0, num_features - 1)
    selected_indices = set()
    selected_indices_list = []
    selected_indices.add(start_feature_index)
    selected_indices_list.append(start_feature_index)

    pearson_score_matrix = np.zeros((num_features, num_features))
    current_score = []
    for _ in range(k - 1):
        temp_scores = []
        for i in range(num_features):
            if i in selected_indices:
                temp_scores.append(-float('inf'))
            else:
                f_test_score = f_test_scores[i]
                q = 0
                for j in selected_indices:
                    # pearson score
                    if j > i:
                        if pearson_score_matrix[i][j] == 0:
                            pearson_score_matrix[i][j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
                        q += pearson_score_matrix[i][j]
                    else:
                        if pearson_score_matrix[j][i] == 0:
                            pearson_score_matrix[j][i] = np.corrcoef(X[:, i], X[:, j])[0, 1]
                        q += pearson_score_matrix[j][i]
                temp_scores.append(f_test_score / (q / len(selected_indices)))
        add_max_score_to_list(temp_scores, current_score, selected_indices, selected_indices_list)
    plot_score(current_score)
    # 获取特征名称
    feature_names = data.columns[:-1]
    # 获取选定索引对应的特征名称
    selected_feature_names = [feature_names[i] for i in selected_indices_list[:-1]]
    # 创建一个新的 DataFrame，包含特征名称和对应的分数
    result_df = pd.DataFrame({'Name': selected_feature_names, 'Score': current_score})
    # 按照 Score 列的值降序排序
    result_df = result_df.sort_values(by='Score', ascending=False)
    # 将结果写入新的 CSV 文件
    result_df.to_csv(r'D:\jupyter\new_funtion\2024_7_9\7_30\duibi\mrmr_Carcinoma_data.csv', index=False)

    return selected_indices_list


def FCD(X, y, k):
    num_features = len(X[0])
    f_test_scores = [f_oneway(X[:, i], y)[0] for i in range(num_features)]

    start_feature_index = random.randint(0, num_features - 1)
    selected_indices = set()
    selected_indices_list = []
    selected_indices.add(start_feature_index)
    selected_indices_list.append(start_feature_index)

    pearson_score_matrix = np.zeros((num_features, num_features))

    current_score = []
    for _ in range(k - 1):
        temp_scores = []
        for i in range(num_features):
            if i in selected_indices:
                temp_scores.append(-float('inf'))
            else:
                f_test_score = f_test_scores[i]
                diff = 0
                for j in selected_indices:
                    # pearson score
                    if j > i:
                        if pearson_score_matrix[i][j] == 0:
                            pearson_score_matrix[i][j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
                        diff += pearson_score_matrix[i][j]
                    else:
                        if pearson_score_matrix[j][i] == 0:
                            pearson_score_matrix[j][i] = np.corrcoef(X[:, i], X[:, j])[0, 1]
                        diff += pearson_score_matrix[j][i]
                    # diff += np.corrcoef(X[:,i], X[:,j])[0, 1]
                temp_scores.append(f_test_score - diff / len(selected_indices))
        add_max_score_to_list(temp_scores, current_score, selected_indices, selected_indices_list)
    print("ok")
    plot_score(current_score)
    print(len(current_score), len(selected_indices_list))
    # 获取特征名称
    feature_names = data.columns[:-1]
    # 获取选定索引对应的特征名称
    selected_feature_names = [feature_names[i] for i in selected_indices_list[:-1]]
    # 创建一个新的 DataFrame，包含特征名称和对应的分数
    result_df = pd.DataFrame({'Name': selected_feature_names, 'Score': current_score})
    # 按照 Score 列的值降序排序
    result_df = result_df.sort_values(by='Score', ascending=False)
    # 将结果写入新的 CSV 文件
    result_df.to_csv(r'D:\jupyter\new_funtion\2024_7_9\7_30\duibi\mrmr_Carcinoma_data.csv', index=False)
    return selected_indices_list


data = pd.read_csv(r'D:\jupyter\new_funtion\2024_7_9\7_30\mRNA_LncRNA_Non_Small_Cell_Lung_Carcinoma_data.csv')

X = data.drop('Label', axis=1).to_numpy()


y = data['Label'].to_numpy()
FCD(X, y, len(data.drop('Label', axis=1).columns))
# FCQ(X, y, num)
end_time = time.time()
print("消耗的时间", end_time - start_time)

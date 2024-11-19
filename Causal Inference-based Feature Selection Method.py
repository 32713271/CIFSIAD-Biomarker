import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import math
import warnings

warnings.filterwarnings("ignore")


def load_data(data_path, label_column='Label'):
    """加载数据并进行标准化"""
    data = pd.read_csv(data_path)
    labels = data[label_column]
    features = data.drop(columns=[label_column])
    scaled_features = StandardScaler().fit_transform(features)
    return pd.DataFrame(scaled_features, columns=features.columns), labels


def load_gene_interactions(file_path):
    """加载基因交互数据，返回字典"""
    gene_dict = defaultdict(set)
    with open(file_path, 'r') as file:
        next(file)  # 跳过第一行
        for line in file:
            data = line.strip().split('\t')
            if len(data) == 8:
                gene1, gene2 = data[1], data[4]
                gene_dict[gene1].add(gene2)
                gene_dict[gene2].add(gene1)
    return gene_dict


def calculate_ace(data, labels, gene_dict, output_path):
    """计算 ACE 值并保存结果"""
    ace_values = []
    feature_columns = data.columns

    for column_name in feature_columns:
        dependent_variable = data[column_name]
        independent_variables = data.drop(columns=[column_name])
        related_genes = gene_dict.get(column_name, set())
        existing_genes = [gene for gene in related_genes if gene in independent_variables.columns]

        # 使用线性回归或 Lasso 回归计算 ACE
        if existing_genes:
            X = independent_variables[existing_genes]
            y = dependent_variable
            model = LinearRegression()
            model.fit(X, y)
            coefs = model.coef_
        else:
            best_alpha = 0.01
            model = Lasso(alpha=best_alpha).fit(independent_variables, dependent_variable)
            non_zero_coefs = model.coef_ != 0
            X = independent_variables.loc[:, non_zero_coefs]
            coefs = model.coef_[non_zero_coefs]

        f = (X * coefs).sum(axis=1)
        f_s = (f * coefs).sum(axis=0)
        variance_product = X.var().prod()
        e_s = ((f - f_s) ** 2) / (2 * variance_product)
        e_f = (1 / (math.sqrt(2 * math.pi) * abs(variance_product))) * np.exp(np.clip(e_s, -np.inf, 700))
        ace = np.mean(np.abs(labels - e_f))

        ace_values.append({"特征": column_name, "ACE": ace})

    ace_df = pd.DataFrame(ace_values).sort_values(by="ACE", ascending=True)
    ace_df.to_csv(output_path, index=False)
    return ace_df


def evaluate_models(data, labels, features, max_features=500):
    """使用多个分类器逐步评估特征子集"""
    classifiers = {
        "SVC": SVC(),
        "NaiveBayes": GaussianNB(),
        "XGBoost": XGBClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }
    results = []

    for clf_name, classifier in classifiers.items():
        print(f"\n正在测试分类器: {clf_name}")
        for num_features in range(1, min(max_features, len(features)) + 1):
            selected_features = features[:num_features]
            X = data[selected_features]
            y = labels

            cross_val_scores = cross_val_score(classifier, X, y, cv=10, scoring='accuracy')
            cross_val_predictions = cross_val_predict(classifier, X, y, cv=10)
            f1 = f1_score(y, cross_val_predictions, average='weighted')
            accuracy_mean = np.mean(cross_val_scores)

            results.append({'Classifier': clf_name, 'NumFeatures': num_features, 
                            'Accuracy': accuracy_mean, 'F1Score': f1})

            print(f"分类器 {clf_name} 使用 {num_features} 个特征: 准确度: {accuracy_mean}, F1 分数: {f1}")

    return pd.DataFrame(results)


def main():
    # 配置文件路径
    data_path = r"D:\jupyter\new_funtion\smote_and_reduce\新建文件夹\原始\o_reduce.csv"
    gene_interactions_path = r"D:\jupyter\new_funtion\download.txt"
    ace_output_path = r"D:\jupyter\new_funtion\Reduce\reduce_o\ace_sorted_result.csv"

    # 数据加载
    data, labels = load_data(data_path)
    gene_dict = load_gene_interactions(gene_interactions_path)

    # 计算 ACE
    print("\n计算 ACE 值...")
    ace_df = calculate_ace(data, labels, gene_dict, ace_output_path)
    print(f"ACE 计算完成，结果已保存到 {ace_output_path}")

    # 模型评估
    print("\n开始模型评估...")
    features = ace_df['特征']
    results_df = evaluate_models(data, labels, features)
    
    # 找到最高 F1 分数对应的特征数量和模型
    results_cleaned = results_df.dropna(subset=['F1Score'])
    max_f1_index = results_cleaned['F1Score'].idxmax()
    best_result = results_cleaned.loc[max_f1_index]
    print(f"\n最高 F1 Score: {best_result['F1Score']} 对应分类器: {best_result['Classifier']} "
          f"使用特征数量: {best_result['NumFeatures']}")


if __name__ == "__main__":
    main()

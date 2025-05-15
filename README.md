# Fake News Detection

本项目使用数据挖掘与文本分析方法，对新闻文本进行真假识别。我们通过 TF-IDF 特征提取及多种分类器组合，探索如何高效地检测虚假新闻。

## 📁 项目结构

fake-news-detection.ipynb     # 主代码文件
Fake\_first\_10000.csv          # 虚假新闻示例数据
True\_first\_10000.csv          # 真实新闻示例数据

## ⚙️ 使用的工具与库

Notebook 中使用的主要 Python 库包括：

- `pandas`：数据加载与处理
- `numpy`：数值计算
- `sklearn`：
  - `feature_extraction.text.TfidfVectorizer`：TF-IDF 特征提取
  - `model_selection.train_test_split`：训练集与测试集划分
  - `linear_model.LogisticRegression`、`tree.DecisionTreeClassifier`、`ensemble.RandomForestClassifier`、`ensemble.GradientBoostingClassifier`：多个分类模型
  - `metrics`：模型评估指标（准确率、召回率、F1 分数等）
- `matplotlib.pyplot` 与 `seaborn`：数据可视化
- `string`, `re`：文本清洗（正则表达式与标点处理）

## 🗂 数据说明

项目中使用了两个数据集：

- `Fake_first_10000.csv`: 包含10000条虚假新闻文本
- `True_first_10000.csv`: 包含10000条真实新闻文本

每个数据集均包含以下字段：

- `title`: 新闻标题
- `text`: 新闻正文

## 🚀 运行方式

### 1. 安装依赖

推荐使用虚拟环境（如 `venv` 或 Conda）：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. 下载项目并启动 Jupyter

```bash
jupyter notebook
```

打开 `fake-news-detection.ipynb`，按顺序逐单元格运行。

### 3. 数据路径修改（如有需要）

默认数据路径为：

```python
/workspaces/Fake-News/input/fake-news-detection/Fake_first_10000.csv
/workspaces/Fake-News/input/fake-news-detection/True_first_10000.csv
```

若在本地运行，请修改为相对路径：

```python
./Fake_first_10000.csv
./True_first_10000.csv
```

## 📊 模型评估

项目中训练并比较以下模型：

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**
* **Gradient Boosting**

每种模型都会输出：

* 混淆矩阵
* 准确率、精确率、召回率、F1 分数
* 可视化对比图表

## ✅ 项目亮点

* 统一使用 TF-IDF 向量作为输入特征，便于模型横向比较
* 模块化结构清晰，便于扩展或替换模型
* 支持大规模文本处理与中文本清洗（部分预留逻辑）

## 🧠 潜在扩展方向

* 引入深度学习模型（如 LSTM、BERT）
* 增加情感分析或新闻来源特征
* 多语言支持与跨域泛化评估
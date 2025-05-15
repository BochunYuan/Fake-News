# Fake News Detection

This project uses data mining and text analysis methods to identify fake news. By employing TF-IDF feature extraction and a combination of multiple classifiers, we explore efficient ways to detect false news.

## 📁 Project Structure

fake-news-detection.ipynb     # Main code file
Fake_first_10000.csv          # Fake news sample data
True_first_10000.csv          # True news sample data

## ⚙️ Tools and Libraries Used

The main Python libraries used in the Notebook include:

- `pandas`: Data loading and processing
- `numpy`: Numerical computation
- `sklearn`:
  - `feature_extraction.text.TfidfVectorizer`: TF-IDF feature extraction
  - `model_selection.train_test_split`: Train-test split
  - `linear_model.LogisticRegression`, `tree.DecisionTreeClassifier`, `ensemble.RandomForestClassifier`, `ensemble.GradientBoostingClassifier`: Multiple classification models
  - `metrics`: Model evaluation metrics (accuracy, precision, recall, F1 score, etc.)
- `matplotlib.pyplot` and `seaborn`: Data visualization
- `string`, `re`: Text cleaning (regular expressions and punctuation handling)

## 🗂 Data Description

The project uses two datasets:

- `Fake_first_10000.csv`: Contains 10,000 fake news texts
- `True_first_10000.csv`: Contains 10,000 true news texts

Each dataset includes the following fields:

- `title`: News title
- `text`: News body

## 🚀 Running the Project

### 1. Install Dependencies

It is recommended to use a virtual environment (such as `venv` or Conda):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Download the Project and Launch Jupyter

```bash
jupyter notebook
```

Open `fake-news-detection.ipynb` and run the cells in sequence.

### 3. Modify Data Paths (if necessary)

The default data paths are:

```python
/workspaces/Fake-News/input/fake-news-detection/Fake_first_10000.csv
/workspaces/Fake-News/input/fake-news-detection/True_first_10000.csv
```

If running locally, modify them to relative paths:

```python
./Fake_first_10000.csv
./True_first_10000.csv
```

## 📊 Model Evaluation

The project trains and compares the following models:

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**
* **Gradient Boosting**

Each model outputs:

* Confusion matrix
* Accuracy, precision, recall, F1 score
* Visualization comparison charts

## ✅ Project Highlights

* Uniformly using TF-IDF vectors as input features for easy model comparison
* Modular structure for easy expansion or model replacement
* Support for large-scale text processing and Chinese text cleaning (some logic reserved)

## 🧠 Potential Expansion Directions

* Introducing deep learning models (such as LSTM, BERT)
* Adding sentiment analysis or news source features
* Multilingual support and cross-domain generalization evaluation
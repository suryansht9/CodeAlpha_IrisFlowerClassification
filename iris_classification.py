# Iris Flower Classification
# CodeAlpha Internship Project
# Author: Your Name

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset (from local file inside repo)
data = pd.read_csv("iris.csv")   # <-- keep iris.csv in same repo folder

print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:\n", data.head())

# 3. Exploratory Data Analysis (EDA)
print("\nDataset Info:\n")
print(data.info())

print("\nClass Distribution:\n")
print(data['species'].value_counts())

# Visualize class distribution
sns.countplot(x='species', data=data)
plt.title("Count of Each Iris Flower Species")
plt.show()

# Pairplot for feature relationships
sns.pairplot(data, hue="species")
plt.show()

# 4. Data Preprocessing
X = data.drop("species", axis=1)   # features
y = data["species"]               # target

# Encode species labels (setosa=0, versicolor=1, virginica=2)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Example Prediction
example = np.array([[5.1, 3.5, 1.4, 0.2]])  # sepal_length, sepal_width, petal_length, petal_width
pred = models["Random Forest"].predict(example)
print("\nPrediction for input [5.1, 3.5, 1.4, 0.2]:", encoder.inverse_transform(pred)[0])

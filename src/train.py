from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

from utils import print_evaluation

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in df.columns]  # Clean column names

# Prepare features and target
X = df.drop(columns='target')
y = iris.target_names[df['target']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Pipeline with scaling + logistic regression (C=10 is our best)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(C=10.0, penalty='l2', max_iter=200))
])

# Train and evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print_evaluation(y_test, y_pred, "Final Logistic Regression Model (C=10)")
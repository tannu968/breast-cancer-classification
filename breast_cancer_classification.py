import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

# Predict the labels of the test set
y_predict = model.predict(X_test)

# Display accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))

# Display cross-tabulation of actual vs predicted labels
print(pd.crosstab(y_test, y_predict))

# Initialize another model with hyperparameter tuning
tuned_model = XGBClassifier(
    max_depth=2,
    subsample=0.2,
    n_estimators=200,
    learning_rate=0.05,
    random_state=5,
    min_child_weight=1,
    reg_alpha=0,
    reg_lambda=1
)

# Train and evaluate the tuned model
tuned_model.fit(X_train, y_train)
tuned_y_predict = tuned_model.predict(X_test)

# Display accuracy for train and test sets
print('Train accuracy:', accuracy_score(y_train, model.predict(X_train)))
print('Test accuracy:', accuracy_score(y_test, tuned_y_predict))

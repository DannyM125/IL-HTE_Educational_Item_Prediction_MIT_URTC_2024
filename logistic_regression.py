import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import joblib

df = pd.read_csv('item_response_public.csv')
feature_columns = ['s_white_num', 's_black_num', 's_asian_num', 's_hispanic_num', 
                   's_male_num', 's_lep_num', 's_iep_num', 's_ses_low', 's_ses_med', 's_homelang_eng', 's_q_num', 's_itt_consented', 's_maprit_1819w_std']
label_columns = ['s_correct']

# Extract features and labels
X = df[feature_columns].copy()
y = df[label_columns]

# Replace infinite values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN values
X = X.dropna()
y = y.loc[X.index]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline model: Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
lr_model.fit(X_train_scaled, y_train['s_correct'])
lr_predictions = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test['s_correct'], lr_predictions):.4f}")
print(f"F1 Score: {f1_score(y_test['s_correct'], lr_predictions):.4f}")
print(f"AUC ROC: {roc_auc_score(y_test['s_correct'], lr_proba):.4f}")

# Feature importance for Logistic Regression
lr_importances = np.abs(lr_model.coef_[0])
importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': lr_importances})

# Reorder the DataFrame to match the feature_columns order
importance_df = importance_df.set_index('Feature').reindex(feature_columns).reset_index()

# Plot feature importance for Logistic Regression with the same y-axis order as CNN
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='green')
plt.xlabel('Impact on Prediction (Absolute Coefficient Value)')
plt.ylabel('Feature')
plt.title('Feature Importance from Logistic Regression')
plt.gca().invert_yaxis()  # To display the highest values on top
plt.show()

# SHAP values for Logistic Regression
explainer = shap.Explainer(lr_model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# SHAP summary plot
# Reorder SHAP values to match the feature_columns order
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_columns)
plt.title('SHAP Summary Plot for Logistic Regression')
plt.close()
plt.show()

# Save the trained Logistic Regression model
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')  # Save the model to a file

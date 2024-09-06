import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import visualkeras
from PIL import ImageFont

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

# Split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate DNN model
print("\nTraining DNN model...")
dnn_model = create_dnn_model(X_train_scaled.shape[1])
dnn_history = dnn_model.fit(X_train_scaled, y_train['s_correct'], epochs=50, batch_size=64, validation_split=0.2, verbose=0)

dnn_predictions = (dnn_model.predict(X_test_scaled) > 0.5).astype(int)
dnn_proba = dnn_model.predict(X_test_scaled)

print("DNN Model Results:")
print(f"Accuracy: {accuracy_score(y_test['s_correct'], dnn_predictions):.4f}")
print(f"F1 Score: {f1_score(y_test['s_correct'], dnn_predictions):.4f}")
print(f"AUC ROC: {roc_auc_score(y_test['s_correct'], dnn_proba):.4f}")

# Feature Importance for DNN using SHAP DeepExplainer
explainer = shap.DeepExplainer(dnn_model, X_train_scaled[:100])
shap_values = explainer.shap_values(X_test_scaled[:100])

# Convert SHAP values to 2D array for summary plot
shap_values_2d = np.array(shap_values).reshape(-1, len(feature_columns))

shap.summary_plot(shap_values_2d, X_test_scaled[:100], feature_names=feature_columns)

# Bar graph
shap_abs_mean = np.mean(np.abs(shap_values_2d), axis=0)
plt.figure(figsize=(10, 6))
plt.barh(feature_columns, shap_abs_mean, color='red')
plt.xlabel('Impact on Prediction (Average Absolute SHAP Value)')
plt.title('Feature Importance Based on SHAP Values (DNN)')
plt.gca().invert_yaxis()  # To display the highest values on top
plt.show()

# Save the trained DNN model
dnn_model.save('models/dnn_model.h5')

font = ImageFont.truetype("arial.ttf", 32)
visualkeras.layered_view(dnn_model, to_file='dnn_model.png', legend=True, font=font, font_color= 'black', spacing=20, draw_volume=True, scale_xy=1, scale_z=1, max_z= 1200)
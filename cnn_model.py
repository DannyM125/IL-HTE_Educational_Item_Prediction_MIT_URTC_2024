import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import visualkeras
from PIL import ImageFont

df = pd.read_csv('item_response_public.csv')
feature_columns = ['s_white_num', 's_black_num', 's_asian_num', 's_hispanic_num', 
                   's_male_num', 's_lep_num', 's_iep_num', 's_ses_low', 's_ses_med', 's_homelang_eng', 's_q_num', 's_itt_consented', 's_maprit_1819w_std']
label_columns = ['s_correct']

X = df[feature_columns].copy()
y = df[label_columns]

# Replace infinite values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN values
X = X.dropna()
y = y.loc[X.index]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for CNN
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv1D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate CNN model
print("\nTraining CNN model...")
cnn_model = create_cnn_model((X_train_scaled.shape[1], 1))
cnn_history = cnn_model.fit(X_train_cnn, y_train['s_correct'], epochs=50, batch_size=64, validation_split=0.2, verbose=0)

cnn_predictions = (cnn_model.predict(X_test_cnn) > 0.5).astype(int)
cnn_proba = cnn_model.predict(X_test_cnn)

print("CNN Model Results:")
print(f"Accuracy: {accuracy_score(y_test['s_correct'], cnn_predictions):.4f}")
print(f"F1 Score: {f1_score(y_test['s_correct'], cnn_predictions):.4f}")
print(f"AUC ROC: {roc_auc_score(y_test['s_correct'], cnn_proba):.4f}")

# Feature Importance for CNN using SHAP KernelExplainer
explainer = shap.KernelExplainer(lambda x: cnn_model.predict(x.reshape(x.shape[0], x.shape[1], 1)), X_train_scaled[:100])
shap_values = explainer.shap_values(X_test_scaled[:100], nsamples=100)

# SHAP summary plot
shap_values_2d = np.array(shap_values).reshape(-1, len(feature_columns))
shap.summary_plot(shap_values_2d, X_test_scaled[:100], feature_names=feature_columns)

# Bar graph
shap_abs_mean = np.mean(np.abs(shap_values_2d), axis=0)
plt.figure(figsize=(10, 6))
plt.barh(feature_columns, shap_abs_mean, color='blue')
plt.xlabel('Impact on Prediction (Average Absolute SHAP Value)')
plt.title('Feature Importance Based on SHAP Values (CNN)')
plt.gca().invert_yaxis()  # To display the highest values on top
plt.show()

# Save the trained CNN model
cnn_model.save('models/cnn_model.h5')

font = ImageFont.truetype("arial.ttf", 32)
visualkeras.layered_view(cnn_model, to_file='cnn_model.png', legend=True, font=font, font_color= 'black', spacing=20, draw_volume=True, scale_xy=1, scale_z=1, max_z= 1200)
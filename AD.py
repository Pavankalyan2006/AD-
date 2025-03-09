import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.metrics import r2_score
import joblib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv(r"C:\Users\pavan\OneDrive\Desktop\AD\Ecommerce_customers(1).csv", usecols=['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent'])
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]
from scipy.stats import zscore
data = data[(np.abs(zscore(data)) < 3).all(axis=1)]
X = data.drop(columns=['Yearly Amount Spent'])
y = data['Yearly Amount Spent']
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
power = PowerTransformer()
X_scaled = scaler.fit_transform(X_poly)
X_transformed = power.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.15, random_state=42)
rf = RandomForestRegressor(n_estimators=500, max_depth=25, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
param_grid_xgb = {
    'n_estimators': [400, 500,600],
    'max_depth': [10, 12, 14],
    'learning_rate': [0.01, 0.03],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.7, 0.8]
}
grid_search_xgb = GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror', n_jobs=-1),
                               param_grid=param_grid_xgb,
                               cv=3,
                               scoring='r2',
                               verbose=0)
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
dnn = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
optimizer = Adam(learning_rate=0.001)
dnn.compile(optimizer=optimizer, loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
dnn.fit(X_train, y_train, epochs=200, batch_size=256, validation_split=0.2, verbose=0, callbacks=[early_stopping])
y_pred_dnn = dnn.predict(X_test).flatten()
r2_dnn = r2_score(y_test, y_pred_dnn)
final_pred = (y_pred_rf * 0.4) + (y_pred_xgb * 0.4) + (y_pred_dnn * 0.2)
r2_ensemble = r2_score(y_test, final_pred)
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(power, 'power_transformer.pkl')
joblib.dump(poly, 'polynomial_features.pkl')
joblib.dump(rf, 'random_forest.pkl')
joblib.dump(best_xgb, 'xgboost.pkl')
dnn.save('deep_learning_model.h5')
print(f'Random Forest R2: {r2_rf * 100:.2f}%')
print(f'XGBoost R2: {r2_xgb * 100:.2f}%')
print(f'Deep Learning Model R2: {r2_dnn * 100:.2f}%')
print(f'Ensemble Model R2: {r2_ensemble * 100:.2f}%')
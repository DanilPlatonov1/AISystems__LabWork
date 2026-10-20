import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import time
import warnings
warnings.filterwarnings('ignore')

# Шаг 1. Загрузка данных
X_train = pd.read_csv('X_train_reg.csv')
X_test = pd.read_csv('X_test_reg.csv')
y_train = pd.read_csv('y_train_reg.csv').values.ravel()
y_test = pd.read_csv('y_test_reg.csv').values.ravel()

print(f"\nРазмер X_train до удаления: {X_train.shape}")
print(f"Размер X_test до удаления:  {X_test.shape}")
print(f"Размер y_train: {y_train.shape}")
print(f"Размер y_test:  {y_test.shape}")



# Шаг 2. Линейная регрессия
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
print("-"*60)
print(f"MAE:  {mae_lr:.3f}")
print(f"RMSE: {rmse_lr:.3f}")
print(f"R²:   {r2_lr:.3f}")

coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nТоп-5 важных признаков (по модулю коэффициента):")
print(coef_df.head(5).to_string(index=False))



# Шаг 3. Случайный лес
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print("-"*60)
print(f"MAE:  {mae_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")
print(f"R²:   {r2_rf:.3f}")

rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f"\nR² на кросс-валидации (5 фолдов):")

print(f"Среднее: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nТоп-5 важных признаков (Random Forest):")
print(feature_importance_rf.head(5).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance_rf.head(10),
    x='importance',
    y='feature',
    palette='viridis',
    color='steelblue'
)
plt.title('Важность признаков (Random Forest)', fontsize=14)
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.tight_layout()
plt.savefig('diograms/rf_feature_importance.png')
plt.show()



# Шаг 4. XGBoost
xgb_base = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_base.fit(X_train, y_train)
y_pred_xgb_base = xgb_base.predict(X_test)
r2_xgb_base = r2_score(y_test, y_pred_xgb_base)
print("-"*60, f"\nXGBoost (базовый) R²: {r2_xgb_base:.3f}")

print("\nЗапуск GridSearchCV")
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1), param_grid, cv=3,
    scoring='r2', n_jobs=-1, verbose=0,
)

start_time = time.time()
xgb_grid.fit(X_train, y_train)
print(f"Время подбора: {time.time() - start_time:.1f} сек.")
print(f"Лучшие параметры: {xgb_grid.best_params_}")
print(f"Лучший R² на CV: {xgb_grid.best_score_:.3f}")

xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"MAE:  {mae_xgb:.3f}")
print(f"RMSE: {rmse_xgb:.3f}")
print(f"R²:   {r2_xgb:.3f}")



# Шаг 6. Сравнительный анализ и визуализация
results_df = pd.DataFrame({
    'Модель': ['Линейная регрессия', 'Случайный лес', 'XGBoost'],
    'MAE': [mae_lr, mae_rf, mae_xgb],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
    'R²': [r2_lr, r2_rf, r2_xgb]
})
print("-"*60)
print(results_df.to_string(index=False))
results_df.to_csv('regression_results.csv', index=False)

best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Модель']
best_model_pred = {
    'Линейная регрессия': y_pred_lr,
    'Случайный лес': y_pred_rf,
    'XGBoost': y_pred_xgb,
}[best_model_name]

plt.figure(figsize=(8, 8))
plt.scatter(y_test, best_model_pred, alpha=0.4, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Реальные значения BMI')
plt.ylabel('Предсказанные значения BMI')
plt.title(f'Предсказано vs Реально (Лучшая модель: {best_model_name})')
plt.tight_layout()
plt.savefig('diograms/predicted_vs_actual.png')
plt.show()

# 6.3. Анализ остатков
residuals = y_test - best_model_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Гистограмма остатков
axes[0].hist(residuals, bins=30, edgecolor='black', color='coral')
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_title('Распределение остатков')
axes[0].set_xlabel('Ошибка (Реальное - Предсказанное)')

# Q-Q plot
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q plot остатков')

plt.tight_layout()
plt.savefig('diograms/residuals_analysis.png')
plt.show()

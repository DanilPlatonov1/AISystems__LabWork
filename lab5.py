import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ШАГ 1. Загрузка данных
X_train_reg = pd.read_csv('models/X_train_reg.csv')
X_test_reg = pd.read_csv('models/X_test_reg.csv')
y_train_reg = pd.read_csv('models/y_train_reg.csv').values.ravel()
y_test_reg = pd.read_csv('models/y_test_reg.csv').values.ravel()

X_train_clf = pd.read_csv('models/X_train_clf.csv')
X_test_clf = pd.read_csv('models/X_test_clf.csv')
y_train_clf = pd.read_csv('models/y_train_clf.csv').values.ravel()
y_test_clf = pd.read_csv('models/y_test_clf.csv').values.ravel()

lr_reg = joblib.load('models/model_lr_reg.pkl')
rf_reg = joblib.load('models/model_rf_reg.pkl')
xgb_reg = joblib.load('models/model_xgb_reg.pkl')

lr_clf = joblib.load('models/model_lr_clf.pkl')
rf_clf = joblib.load('models/model_rf_clf.pkl')
xgb_clf = joblib.load('models/model_xgb_clf.pkl')



# ШАГ 2. FI для регрессии
importance_rf_reg = pd.DataFrame({
    'feature': X_train_reg.columns,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

importance_xgb_reg = pd.DataFrame({
    'feature': X_train_reg.columns,
    'importance': xgb_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("-"*60,"\nРЕГРЕССИЯ: Топ-5 важных признаков (Random Forest)")
print(importance_rf_reg.head(5).to_string(index=False))

print("\nРЕГРЕССИЯ: Топ-5 важных признаков (XGBoost)")
print(importance_xgb_reg.head(5).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=importance_rf_reg.head(10), x='importance', y='feature',
            ax=axes[0], color='steelblue')
axes[0].set_title('Random Forest (Регрессия BMI)')
sns.barplot(data=importance_xgb_reg.head(10), x='importance', y='feature',
            ax=axes[1], color='darkorange')
axes[1].set_title('XGBoost (Регрессия BMI)')
plt.tight_layout()
plt.savefig('diograms/feature_importance_regression.png')



# ШАГ 3. FI для классификации
importance_rf_clf = pd.DataFrame({
    'feature': X_train_clf.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

importance_xgb_clf = pd.DataFrame({
    'feature': X_train_clf.columns,
    'importance': xgb_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("-"*60,"\nКЛАССИФИКАЦИЯ: Топ-5 важных признаков (Random Forest) ===")
print(importance_rf_clf.head(5).to_string(index=False))

print("\nКЛАССИФИКАЦИЯ: Топ-5 важных признаков (XGBoost) ===")
print(importance_xgb_clf.head(5).to_string(index=False))

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=importance_rf_clf.head(10), x='importance', y='feature',
            ax=axes[0], color='green')
axes[0].set_title('Random Forest (Классификация: Инсульт)')
sns.barplot(data=importance_xgb_clf.head(10), x='importance', y='feature',
            ax=axes[1], color='purple')
axes[1].set_title('XGBoost (Классификация: Инсульт)')
plt.tight_layout()
plt.savefig('diograms/feature_importance_classification_lab5.png')



# ШАГ 4. Сравнение важности признаков между моделями
print("-"*60,"\nСравнение важности признаков (РЕГРЕССИЯ)")
compare_reg = pd.merge(importance_rf_reg, importance_xgb_reg, on='feature', suffixes=('_RF', '_XGB'))
compare_reg['diff'] = abs(compare_reg['importance_RF'] - compare_reg['importance_XGB'])
print(compare_reg.sort_values('diff', ascending=False).head(5).to_string(index=False))

print("\nСравнение важности признаков (КЛАССИФИКАЦИЯ)")
compare_clf = pd.merge(importance_rf_clf, importance_xgb_clf, on='feature', suffixes=('_RF', '_XGB'))
compare_clf['diff'] = abs(compare_clf['importance_RF'] - compare_clf['importance_XGB'])
print(compare_clf.sort_values('diff', ascending=False).head(5).to_string(index=False))
plt.show()



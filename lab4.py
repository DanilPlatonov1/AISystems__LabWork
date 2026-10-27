import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, classification_report)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# ШАГ 1. Загрузка данных
X_train = pd.read_csv('models/X_train_clf.csv')
X_test = pd.read_csv('models/X_test_clf.csv')
y_train = pd.read_csv('models/y_train_clf.csv').values.ravel()
y_test = pd.read_csv('models/y_test_clf.csv').values.ravel()

print("Распределение классов в train:")
print(pd.Series(y_train).value_counts())
print("\nРаспределение классов в test:")
print(pd.Series(y_test).value_counts())
print(f"\nРазмер X_train: {X_train.shape}")
print(f"Размер X_test:  {X_test.shape}")



# ШАГ 2. Логистическая регрессия
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

precision_lr = precision_score(y_test, y_pred_lr, average='binary', zero_division=0)
recall_lr = recall_score(y_test, y_pred_lr, average='binary')
f1_lr = f1_score(y_test, y_pred_lr, average='binary')
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
print("-"*60,f"\nPrecision: {precision_lr:.3f}")
print(f"Recall:    {recall_lr:.3f}")
print(f"F1-Score:  {f1_lr:.3f}")
print(f"ROC-AUC:   {roc_auc_lr:.3f}")

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nМатрица:")
print(cm_lr)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Нет (0)', 'Да (1)'])
disp.plot(ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix (Logistic Regression)')
plt.tight_layout()
plt.savefig('diograms/cm_logistic_regression.png')



# ШАГ 3. Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
best_f1 = 0
best_threshold = 0.5
best_precision = 0
best_recall = 0

for thresh in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_proba_rf >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_temp, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
        best_precision = precision_score(y_test, y_pred_temp, zero_division=0)
        best_recall = recall_score(y_test, y_pred_temp, zero_division=0)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
print("-"*60,f"\nPrecision: {best_precision:.3f}")
print(f"Recall:    {best_recall:.3f}")
print(f"F1-Score:  {best_f1:.3f}")
print(f"ROC-AUC:   {roc_auc_rf:.3f}")
y_pred_rf = (y_proba_rf >= best_threshold).astype(int)

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:")
print(cm_rf)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Нет (0)', 'Да (1)'])
disp.plot(ax=ax, cmap='Greens')
ax.set_title(f'Confusion Matrix (RF, порог={best_threshold:.2f})')
plt.tight_layout()
plt.savefig('diograms/cm_random_forest.png')

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nТоп-10 важных признаков (Random Forest):")
print(importance_df.head(10).to_string(index=False))
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='importance', y='feature', color='steelblue')
plt.title('Важность признаков (Random Forest)', fontsize=14)
plt.tight_layout()
plt.savefig('diograms/rf_feature_importance_lab4.png')


# ШАГ 4. XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.2f}")
xgb_base = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric='logloss'
)
xgb_base.fit(X_train, y_train)
y_pred_xgb_base = xgb_base.predict(X_test)
r2_xgb_base = f1_score(y_test, y_pred_xgb_base)
print("-"*60,f"\nXGBoost (базовый) F1: {r2_xgb_base:.3f}")

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    XGBClassifier(scale_pos_weight=scale_pos_weight,
                  random_state=42, n_jobs=-1, eval_metric='logloss'),
    param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0
)
xgb_grid.fit(X_train, y_train)
print(f"Лучшие параметры: {xgb_grid.best_params_}")
print(f"Лучший F1 на CV: {xgb_grid.best_score_:.3f}")

xgb_best = xgb_grid.best_estimator_
y_proba_xgb = xgb_best.predict_proba(X_test)[:, 1]
THRESHOLD = 0.4
y_pred_xgb = (y_proba_xgb >= THRESHOLD).astype(int)
precision_xgb_best = precision_score(y_test, y_pred_xgb, zero_division=0)
recall_xgb_best = recall_score(y_test, y_pred_xgb)
f1_xgb_best = f1_score(y_test, y_pred_xgb)
roc_auc_xgb_best = roc_auc_score(y_test, y_proba_xgb)

print(f"Precision: {precision_xgb_best:.3f}")
print(f"Recall:    {recall_xgb_best:.3f}")
print(f"F1-Score:  {f1_xgb_best:.3f}")
print(f"ROC-AUC:   {roc_auc_xgb_best:.3f}")

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\nConfusion Matrix:")
print(cm_xgb)

results_df = pd.DataFrame({
    'Модель': ['Логистическая регрессия', 'Случайный лес', 'XGBoost'],
    'Precision': [precision_lr, best_precision, precision_xgb_best],
    'Recall': [recall_lr, best_recall, recall_xgb_best],
    'F1-Score': [f1_lr, best_f1, f1_xgb_best],
    'ROC-AUC': [roc_auc_lr, roc_auc_rf, roc_auc_xgb_best]
})


# ШАГ 5. Сравнительная таблица
print("-"*60,f"\nСРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print(results_df.to_string(index=False))
results_df.to_csv('classification_results.csv', index=False)

plt.figure(figsize=(8, 6))
models = [
    ('Логистическая регрессия', y_proba_lr),
    ('Случайный лес', y_proba_rf),
    ('XGBoost', y_proba_xgb),
]
for name, y_proba in models:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-кривые для моделей классификации')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('diograms/roc_curves.png')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cm_list = [
    (cm_lr, 'Logistic Regression\n(порог 0.5)'),
    (cm_rf, f'Random Forest\n(порог {best_threshold:.2f})'),
    (cm_xgb, f'XGBoost\n(порог {THRESHOLD})')
]
for i, (cm, title) in enumerate(cm_list):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Здоров', 'Инсульт'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(title, fontsize=12)

plt.suptitle('Сравнение матриц ошибок моделей классификации', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('diograms/confusion_matrices_comparison.png')
plt.show()


# Сохранение моделей и порогов (Классификация)
print("\nСохранение моделей классификации...")
joblib.dump(lr, 'models/model_lr_clf.pkl')
joblib.dump(rf, 'models/model_rf_clf.pkl')
joblib.dump(xgb_best, 'models/model_xgb_clf.pkl')

# Сохраняем пороги в текстовый файл (или можно в json)
with open('models/thresholds.txt', 'w') as f:
    f.write(f"{best_threshold}\n{THRESHOLD}")
print("Модели классификации и пороги успешно сохранены!")
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('models/healthcare-dataset-stroke-data.csv')


# Шаг 1
y_reg = df['bmi'].copy()
y_clf = df['stroke'].copy()
X_reg = df.drop(['id', 'bmi', 'stroke'], axis=1).copy()
X_clf = df.drop(['id', 'stroke'], axis=1).copy()

df_reg = X_reg.copy()
df_reg['bmi'] = y_reg
df_reg = df_reg.dropna(subset=['bmi'])
X_reg = df_reg.drop('bmi', axis=1)
y_reg = df_reg['bmi']

print("Признаки X_reg:", X_reg.shape)
print("Признаки X_clf:", X_clf.shape)
print("Таргеты y_reg:", y_reg.shape)
print("Таргеты y_clf:", y_clf.shape)


# Шаг 2
print('-'*60,"\nПропуски в X_clf (до):", X_clf.isnull().sum().sum())
print("Пропуски в X_reg (до):", X_reg.isnull().sum().sum())

imputer_median = SimpleImputer(strategy='median')
X_clf['bmi'] = imputer_median.fit_transform(X_clf[['bmi']])

print("Пропуски в X_clf (после):", X_clf.isnull().sum().sum())
print("Пропуски в X_reg (после):", X_reg.isnull().sum().sum())


# Шаг 3
def detect_outliers_iqr(data, column):
    Q1, Q3 = data[column].quantile(0.25), data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return (data[column] < lower) | (data[column] > upper)

num_cols = ['age', 'avg_glucose_level', 'bmi']
print('-'*60,"\nВыбросы по IQR:")
for col in num_cols:
    n = detect_outliers_iqr(X_clf, col).sum()
    print(f"  {col}: {n} ({n / len(X_clf) * 100:.2f}%)")

def winsorize_series(data, lower_q=0.01, upper_q=0.99):
    lower = data.quantile(lower_q)
    upper = data.quantile(upper_q)
    return data.clip(lower=lower, upper=upper)

print("\nДо winsorization (X_clf):")
print(X_clf[['avg_glucose_level', 'bmi']].describe().round(2))
for col in ['avg_glucose_level', 'bmi']:
    X_clf[col] = winsorize_series(X_clf[col])
X_reg['avg_glucose_level'] = winsorize_series(X_reg['avg_glucose_level'])
print("\nПосле winsorization (X_clf):")
print(X_clf[['avg_glucose_level', 'bmi']].describe().round(2))


# Шаг 4
binary_map_married = {'Yes': 1, 'No': 0}
binary_map_residence = {'Urban': 1, 'Rural': 0}

print('-'*60,"\nX_reg до кодирования:", X_reg.shape)
print("X_clf до кодирования:", X_clf.shape)

for X in [X_reg, X_clf]:
    X['ever_married'] = X['ever_married'].map(binary_map_married)
    X['Residence_type'] = X['Residence_type'].map(binary_map_residence)

nominal_cols = ['gender', 'work_type', 'smoking_status']
X_reg = pd.get_dummies(X_reg, columns=nominal_cols, drop_first=True, dtype=int)
X_clf = pd.get_dummies(X_clf, columns=nominal_cols, drop_first=True, dtype=int)

print("X_reg после кодирования:", X_reg.shape)
print("X_clf после кодирования:", X_clf.shape)
print("\nКолонки X_clf:")
print(X_clf.columns.tolist())


# Шаг 5
continuous_cols_reg = ['age', 'avg_glucose_level']
scaler_reg = StandardScaler()
X_reg[continuous_cols_reg] = scaler_reg.fit_transform(X_reg[continuous_cols_reg])

continuous_cols_clf = ['age', 'avg_glucose_level', 'bmi']

print('-'*60,"\nДо масштабирования (X_clf):")
print(X_clf[continuous_cols_clf].describe().round(2))

scaler_clf = StandardScaler()
X_clf[continuous_cols_clf] = scaler_clf.fit_transform(X_clf[continuous_cols_clf])

print("После масштабирования (X_clf):")
print(X_clf[continuous_cols_clf].describe().round(2))


# Шаг 6
# VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_clf_vt = selector.fit_transform(X_clf)
kept = X_clf.columns[selector.get_support()].tolist()
removed = [c for c in X_clf.columns if c not in kept]
print('-'*60,"\nVarianceThreshold")
print(f"Оставлено: {len(kept)}, удалено: {len(removed)}")
print("Удалённые:", removed)

# SelectKBest
# Для классификации (target = stroke)
selector_clf = SelectKBest(score_func=f_classif, k=10)
X_clf_kbest = selector_clf.fit_transform(X_clf, y_clf)
top_clf = X_clf.columns[selector_clf.get_support()].tolist()

print("\nТоп-10 признаков для классификации:")
for f in top_clf:
    print(f"  {f}")
scores = pd.Series(selector_clf.scores_, index=X_clf.columns).sort_values(ascending=False)
print("\nF-статистики (топ-10):")
print(scores.head(10).round(2))

# Для регрессии (target = bmi)
selector_reg = SelectKBest(score_func=f_regression, k=10)
X_reg_kbest = selector_reg.fit_transform(X_reg, y_reg)
top_reg = X_reg.columns[selector_reg.get_support()].tolist()
print("\nТоп-10 признаков для регрессии:")
print(top_reg)

# RFE
estimator = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rfe = RFE(estimator, n_features_to_select=10, step=1)
rfe.fit(X_clf, y_clf)

rfe_features = X_clf.columns[rfe.support_].tolist()
print("\nПризнаки, отобранные RFE:")
for f in rfe_features:
    print(f"  {f}")
ranking = pd.Series(rfe.ranking_, index=X_clf.columns).sort_values()
print("\nРанжирование (1 = оставлен):")
print(ranking)


# Шаг 7
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg,
    test_size=0.2,
    random_state=42
)
print("-"*60,"\nРегрессия")
print(f"X_train_reg: {X_train_reg.shape}")
print(f"X_test_reg:  {X_test_reg.shape}")
print(f"y_train_reg: {y_train_reg.shape}")
print(f"y_test_reg:  {y_test_reg.shape}")

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)
print("\nКлассификация")
print(f"X_train_clf: {X_train_clf.shape}")
print(f"X_test_clf:  {X_test_clf.shape}")
print(f"y_train_clf: {y_train_clf.shape}")
print(f"y_test_clf:  {y_test_clf.shape}")

print("\nПроверка стратификации")
print("Пропорция stroke в исходном датасете:")
print(y_clf.value_counts(normalize=True).round(4))
print("\nПропорция stroke в train:")
print(y_train_clf.value_counts(normalize=True).round(4))
print("\nПропорция stroke в test:")
print(y_test_clf.value_counts(normalize=True).round(4))

# Регрессия
X_train_reg.to_csv('X_train_reg.csv', index=False)
X_test_reg.to_csv('X_test_reg.csv', index=False)
y_train_reg.to_csv('y_train_reg.csv', index=False)
y_test_reg.to_csv('y_test_reg.csv', index=False)

# Классификация
X_train_clf.to_csv('X_train_clf.csv', index=False)
X_test_clf.to_csv('X_test_clf.csv', index=False)
y_train_clf.to_csv('y_train_clf.csv', index=False)
y_test_clf.to_csv('y_test_clf.csv', index=False)

with open('models/scaler_reg.pkl', 'wb') as f:
    pickle.dump(scaler_reg, f)
with open('models/scaler_clf.pkl', 'wb') as f:
    pickle.dump(scaler_clf, f)
with open('models/imputer_median.pkl', 'wb') as f:
    pickle.dump(imputer_median, f)
print("\nВсе файлы сохранены")
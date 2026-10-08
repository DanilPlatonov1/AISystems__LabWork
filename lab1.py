import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Шаг 1. Загрузка данных
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Шаг 2. Информация о данных
print("-"*60,"\nИнформация о данных:")
print(df.info())
print("-"*60,"\nПроверка на пропуски:")
print(df.isnull().sum())
print("-"*60,"\nДубликаты:", df.duplicated().sum())

# Шаг 3. Описательная статистика
print("-"*60,"\nОписательная статистика числовых признаков:")
print(df.describe())

print("-"*60,"\nУникальные значения категориальных признаков:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

print("-"*60,"\nАсимметрия:")
print(skew_vals := df.select_dtypes(include=[np.number]).skew())
print("-"*60,"\nЭксцесс:")
print(kurt_vals := df.select_dtypes(include=[np.number]).kurtosis())

summary = pd.DataFrame({
    'Среднее': df.mean(numeric_only=True),
    'Медиана': df.median(numeric_only=True),
    'Std': df.std(numeric_only=True),
    'Мин': df.min(numeric_only=True),
    'Макс': df.max(numeric_only=True),
    'Асимметрия': skew_vals,
    'Эксцесс': kurt_vals
})
print("-"*60,"\nИтоговая таблица")
print(summary.round(2))

# Шаг 4. Графики
# 4.1. Гистограммы
# df.hist(figsize=(15, 4), bins=30)
# plt.suptitle('Распределение непрерывных признаков', fontsize=16)
# plt.tight_layout()
# plt.savefig('diograms/histograms.png')
# plt.show()

# 4.2. Boxplot
# num_cols = df.select_dtypes(include=[np.number]).columns
# fig, axes = plt.subplots(nrows=(len(num_cols) + 1) // 2, ncols=2,
#                           figsize=(14, 3 * ((len(num_cols) + 1) // 2)))
# axes = axes.flatten()
# for i, col in enumerate(num_cols):
#     sns.boxplot(y=df[col], ax=axes[i])
#     axes[i].set_title(f'Boxplot for {col}')
# for j in range(i + 1, len(axes)):
#     axes[j].set_visible(False)
# plt.tight_layout()
# plt.savefig('diograms/boxplots.png')
# plt.show()
#
# # 4.3. Корреляционная матрица
# plt.figure(figsize=(12, 10))
# sns.heatmap(df.select_dtypes(include=[np.number]).corr(),
#             annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Корреляционная матрица', fontsize=14)
# plt.tight_layout()
# plt.savefig('diograms/correlation_matrix.png')
# plt.show()
#
# # 4.4. Pairplot
# correlations = df.corr(numeric_only=True)['stroke'].abs().sort_values(ascending=False)
# top_features = correlations.index[1:6]
# sns.pairplot(df[list(top_features) + ['stroke']],
#              hue='stroke',
#              diag_kind='kde',
#              plot_kws={'alpha': 0.5, 's': 15})
# plt.suptitle('Pairplot топ-5 признаков по корреляции со stroke', y=1.02)
# plt.savefig('diograms/pairplot.png')
# plt.show()

# 4.5. Анализ целевой переменной
# fig, ax = plt.subplots(figsize=(7, 7))
# counts = df['stroke'].value_counts()
# ax.pie(counts, labels=['Нет инсульта', 'Инсульт'],
#        autopct='%1.2f%%', colors=['#66b3ff', '#ff6666'],
#        startangle=90)
# ax.set_title('Доля классов')
# plt.tight_layout()
# plt.savefig('diograms/target_distribution.png')
# plt.show()
#
# print("Распределение классов:")
# print(df['stroke'].value_counts())
# print(f"\nДисбаланс: {df['stroke'].value_counts()[1] / len(df) * 100:.2f}% — инсульт")
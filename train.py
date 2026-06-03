import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
from collections import Counter

# БЛОК 1: ПОДГОТОВКА ДАННЫХ

# 1. Загрузка данных
# Столбец 'text' — это признаки (X), столбец 'lang' — целевая переменная (y)
df = pd.read_csv('sampled_sentences.csv', sep='\t', header=None, names=["text", "lang"])
df = df[~df['lang'].isin(['ber', 'epo', 'kab'])]
print("Распределение языков:")
print(df['lang'].value_counts())

# 2. Разделение на обучающую (80%) и финальную тестовую (20%)
X = df['text']
y = df['lang']

X_train_val, X_final_test, y_train_val, y_final_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# БЛОК 2: ОБУЧЕНИЕ И ДИАГНОСТИКА ДВУХ МОДЕЛЕЙ

# Модель 1: Логистическая регрессия
pipe_lr = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 6),
        max_features=120000,
        lowercase=True,
        strip_accents='unicode',
        min_df=2
    )),
    ('clf', LogisticRegression(max_iter=1000, C=12, solver='lbfgs'))
])

# Модель 2: Наивный Байес
pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 6),
        max_features=120000,
        lowercase=True,
        strip_accents='unicode',
        min_df=2
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

models = {
    'LogisticRegression': pipe_lr,
    'NaiveBayes': pipe_nb
}

# Кросс-валидация (5 фолдов, стратифицированная)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ (кросс-валидация) ===\n")
best_model_name = None
best_mean_acc = 0

for name, model in models.items():
    scores = cross_val_score(model, X_train_val, y_train_val, cv=cv, scoring='accuracy')
    print(f"{name}: средняя точность = {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Дополнительно: precision/recall на одном фолде для диагностики
    # (обучим на всех данных, кроме последнего фолда, для примера)
    for train_idx, val_idx in cv.split(X_train_val, y_train_val):
        X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        
        model_clone = Pipeline(model.steps)
        model_clone.fit(X_tr, y_tr)
        y_pred_val = model_clone.predict(X_val)
        
        # Диагностика ошибок: какие классы путаются
        print(f"\nДиагностика для {name} (один фолд):")
        print(classification_report(y_val, y_pred_val))
        
        # Визуализация главной ошибки (confusion matrix)
        cm = confusion_matrix(y_val, y_pred_val, labels=model_clone.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=model_clone.classes_, yticklabels=model_clone.classes_)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        break  # показываем только для первого фолда
    
    if scores.mean() > best_mean_acc:
        best_mean_acc = scores.mean()
        best_model_name = name

print(f"\nЛучшая модель по кросс-валидации: {best_model_name} (acc={best_mean_acc:.4f})")

# БЛОК 3: ФИНАЛЬНЫЙ ОТБОР И СОХРАНЕНИЕ

# Обучаем лучшую модель на всех тренировочных+валидационных данных
best_model = models[best_model_name]
best_model.fit(X_train_val, y_train_val)

# Проверка на финальном тесте
y_final_pred = best_model.predict(X_final_test)
final_acc = accuracy_score(y_final_test, y_final_pred)

# Анализ ошибок на финальном тесте
report = classification_report(y_final_test, y_final_pred)
cm_final = confusion_matrix(y_final_test, y_final_pred)

# Находим пары классов, которые путаются чаще всего
class_names = best_model.classes_
confusion_df = pd.DataFrame(cm_final, index=class_names, columns=class_names)
# Убираем диагональ
for c in class_names:
    confusion_df.loc[c, c] = 0
most_confused = confusion_df.stack().idxmax()
print(f"\n=== ИТОГОВЫЙ ОТЧЁТ ===")
print(f"Лучшая модель — {best_model_name}")
print(f"Её ключевая метрика на новых данных (accuracy): {final_acc:.4f}")
print(f"Чаще всего она путает: {most_confused[0]} → {most_confused[1]}")
print("\nПолный classification report:")
print(report)

# Сохраняем модель
joblib.dump(best_model, 'models/language_detector_best.pkl', compress=3)
print("Модель сохранена")
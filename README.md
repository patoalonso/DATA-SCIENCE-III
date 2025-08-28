# NLP + Deep Learning en sinopsis de Goodreads

**Autora:** Alonso Castillo Patricia  
**Comisión:** 67400

Proyecto de fin de cursada: clasificación binaria de “sentimiento” de libros a partir de sus **sinopsis** (`book_details`). Usamos `average_rating` como señal (≥ 4 → positivo), un pipeline NLP de preprocesamiento y un clasificador **MLP** (baseline y versión regularizada mejorada).

---

## 📦 Dataset

- **Fuente:** [Books Dataset — Goodreads (May 2024)](https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024)  
- **Tamaño:** ~16k libros, ~15 columnas (título, autor, géneros, `book_details`, `average_rating`, `num_reviews`, `num_ratings`, etc.).  
- En este trabajo se usa principalmente **`book_details`** (texto libre) y **`average_rating`** (para construir el target).

---

## 🎯 Objetivo

Construir un clasificador binario que prediga **sentimiento**:
- `1` si `average_rating ≥ 4.0` (positivo)
- `0` si `average_rating < 4.0`

---

## 🧪 Metodología (resumen)

1. **EDA:** distribución de `average_rating`, géneros más frecuentes, relación `num_ratings`–`num_reviews` (log–log), métricas léxicas del corpus (TTR, hapax, % stopwords).
2. **Preprocesamiento NLP:**  
   - minúsculas + limpieza de puntuación (conservando apóstrofes),  
   - **stopwords** en inglés (se preservan negaciones),  
   - **lemmatización con POS** (NLTK + WordNet).
3. **Vectorización:** **TF-IDF** con `max_features=5000`, `min_df=5`, `max_df=0.5`, `sublinear_tf=True`, `token_pattern='\\b[a-z]{3,}\\b'`.
4. **Split:** 80/20 estratificado (fit de TF-IDF **solo en train**).
5. **Modelos:**
   - **Baseline MLP:** `Dense(64, relu) → Dense(1, sigmoid)`.
   - **Mejorado:** `Dense(128) → Dropout → Dense(64) → Dropout → Dense(1)` con **L2**, **Dropout**, **EarlyStopping**, **ReduceLROnPlateau**.
6. **Umbral de decisión:** elegido en **validación** maximizando **macro-F1 / balanced accuracy** (sin usar test).
7. **Evaluación:** accuracy, macro-F1, balanced accuracy, matriz de confusión.

---

## ✅ Resultados (test)

| Modelo                             | Accuracy | Macro-F1 | Balanced Acc. |
|-----------------------------------|:-------:|:--------:|:-------------:|
| **MLP baseline** (t = 0.50)       | 0.592   | 0.590    | 0.590         |
| **MLP mejorado** (t = 0.56, val)  | **0.610** | **0.610** | **0.613**     |

**Conclusión breve:** la versión regularizada + selección de umbral en validación **equilibra mejor las clases** y mejora todas las métricas frente a la línea base.

---

## 🗂️ Estructura sugerida del repo


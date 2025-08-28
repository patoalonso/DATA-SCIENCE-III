# NLP + Deep Learning en sinopsis de Goodreads

- **Tarea:** Clasificar sentimiento (positivo si `average_rating ≥ 4`).
- **Dataset:** Goodreads (≈16k libros).  
- **Pipeline NLP:** minúsculas, limpieza, stopwords, lematización (POS).  
- **Vectorización:** TF-IDF (5k features, `min_df=5`, `max_df=0.5`).  
- **Modelos:** MLP baseline vs. MLP regularizado + selección de umbral en validación.

## Resultados (test)
| Modelo | Accuracy | Macro-F1 | Balanced Acc. |
|---|---:|---:|---:|
| Baseline (t=0.50) | 0.592 | 0.590 | 0.590 |
| Mejorado (t=0.56, val) | **0.610** | **0.610** | **0.613** |

## Reproducibilidad
- Python 3.x, TensorFlow 2.x.
- NLTK: `stopwords`, `wordnet`, `averaged_perceptron_tagger`.
- (Opcional) spaCy `en_core_web_sm` para la demo de árbol.

## Estructura
- `notebook.ipynb` — desarrollo completo.
- `data/` — CSV (o enlace a Kaggle).

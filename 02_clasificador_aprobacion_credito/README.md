# Clasificador de aprobación de crédito con árboles de decisión

Proyecto final de **Aprendizaje de Máquina** (Maestría). Todo el desarrollo (EDA, selección de variables, pipelines, optimización, comparación estadística, t-SNE y conclusiones) está en una sola notebook de Google Colab: [`02_clasificador_aprobacion_credito.ipynb`](./02_clasificador_aprobacion_credito.ipynb).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JaviZoom/ProyectoFinalMachineLearning/blob/main/02_clasificador_aprobacion_credito/02_clasificador_aprobacion_credito.ipynb)

## Problema

Predecir si una solicitud de tarjeta de crédito fue **aprobada (+)** o **rechazada (-)** a partir de las características del solicitante. Es un problema de clasificación binaria donde, además de acertar, interesa poder **explicar la decisión**, por eso el modelo central es un árbol de decisión.

## Dataset

[Credit Approval (UCI, id = 27)](https://archive.ics.uci.edu/dataset/27/credit+approval)

- 690 solicitudes y 15 variables (`A1`–`A15`), anonimizadas, mezcla de numéricas (6) y categóricas (9).
- Variable objetivo `A16` (+ / -), con clases bastante balanceadas (44.5 % aprobadas).
- Valores faltantes en 7 variables (afectan al 5.4 % de las filas).

Se descarga automáticamente dentro de la notebook con `ucimlrepo`.

## Metodología

| Sección | Contenido |
|---|---|
| Introducción | Motivación e importancia del problema |
| EDA | Balance de clases, faltantes, distribuciones, relación con el objetivo y correlaciones |
| Feature extraction | No aplica (datos tabulares, sin transfer learning) |
| Feature selection | Filtro con información mutua (`SelectKBest`) |
| Pipelines | `ColumnTransformer` + selección + clasificador, todo dentro de `Pipeline` de scikit-learn |
| Optimización | `GridSearchCV` con `RepeatedStratifiedKFold` (5×3), curvas de validación y curva de aprendizaje |
| Comparación estadística | Árbol de decisión vs. Random Forest con 10×10 folds y test de Wilcoxon |
| Evaluación final | Accuracy, precision, recall, F1, AUC-ROC, matrices de confusión, curvas ROC, árbol e importancias |
| t-SNE | Visualización 2D con la etiqueta real como color |
| Conclusiones y referencias | Conclusiones, limitaciones, trabajo futuro y referencias en formato IEEE |

## Resultados principales

Conjunto de prueba (20 % de los datos, 138 solicitudes):

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Dummy (clase mayoritaria) | 0.558 | 0.000 | 0.000 | 0.000 | 0.500 |
| Regresión logística | 0.870 | 0.853 | 0.853 | 0.853 | 0.960 |
| **Árbol de decisión (optimizado)** | **0.884** | 0.857 | 0.885 | 0.871 | 0.949 |
| **Random Forest (optimizado)** | **0.884** | 0.941 | 0.787 | 0.857 | 0.959 |

- La variable `A9` es la más determinante (EDA, información mutua y raíz del árbol coinciden).
- Mejores hiperparámetros del árbol: `criterion=entropy`, `max_depth=5`, `min_samples_leaf=20`, `k=15` variables.
- Test de Wilcoxon (10×10 folds): el Random Forest es mejor de forma significativa (p ≈ 0.0002), aunque con una ventaja pequeña (≈ 1.4 puntos de accuracy).

## Cómo ejecutarlo

**En Google Colab (recomendado):** abrir la notebook con el botón de arriba y ejecutar *Entorno de ejecución → Ejecutar todo*. La primera celda instala `ucimlrepo`. No requiere GPU.

**En local:**

```bash
pip install -r requirements.txt
jupyter notebook 02_clasificador_aprobacion_credito.ipynb
```

El tiempo de ejecución es de unos pocos minutos.

## Estructura del repositorio

```
02_clasificador_aprobacion_credito/
├── 02_clasificador_aprobacion_credito.ipynb   # Proyecto completo
├── 02_clasificador_aprobacion_credito.pdf     # Versión en PDF de la notebook
├── README.md
├── requirements.txt
└── .gitignore
```

## Referencias principales

- J. R. Quinlan, *Credit Approval*, UCI Machine Learning Repository, 1987. https://doi.org/10.24432/C5FS30
- F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
- Lista completa en formato IEEE al final de la notebook.

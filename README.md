# Proyecto: Clasificación de Personajes de Los Simpsons - KNN vs SVM

## 🎯 Descripción del Proyecto

Este proyecto implementa un estudio comparativo exhaustivo entre los algoritmos **K-Nearest Neighbors (KNN)** y **Support Vector Machine (SVM)** para la clasificación automatizada de personajes de Los Simpsons (Bart vs Homer). El estudio evalúa el impacto de diferentes métodos de codificación de características visuales: RGB, HSV, LBP y combinaciones híbridas.

### 📊 Resultados Principales
- **Mejor rendimiento**: RGB-SVM (92% accuracy)
- **Método más robusto**: HSV+LBP-SVM (84% accuracy)
- **Paridad algorítmica**: LBP (KNN≈SVM, 66% ambos)
- **Análisis comparativo**: SVM superior en 3/4 configuraciones

---

## 🗂️ Estructura del Proyecto

```
si_proyecto_unidad_2_final/
├── notebooks/                          # Jupyter Notebooks principales
│   ├── histograma_color.ipynb         # Sistema baseline RGB
│   ├── histograma_hsv.ipynb           # Sistema optimizado HSV
│   ├── histograma_lbp_descriptor.ipynb # Sistema textural LBP
│   └── lbp_descriptor.ipynb           # Sistema híbrido HSV+LBP
├── data/simpsons/                      # Dataset de imágenes
│   ├── training/                       # Datos de entrenamiento
│   │   ├── bart_simpson/              # Imágenes de Bart (entrenamiento)
│   │   └── homer_simpson/             # Imágenes de Homer (entrenamiento)
│   └── test/                          # Datos de prueba independiente
│       ├── bart_simpson/              # Imágenes de Bart (test)
│       └── homer_simpson/             # Imágenes de Homer (test)
├── output/                            # Gráficos y resultados generados
├── Informe.md                         # Informe académico completo
├── environment.yml                    # Configuración de entorno Anaconda
└── README.md                          # Este archivo
```

---

## 🚀 Cómo Usar Este Proyecto

### Prerrequisitos

**Software requerido:**
- **Anaconda** o **Miniconda** (recomendado para gestión de entornos)
- Python 3.8+ (incluido en Anaconda)
- Jupyter Notebook o VS Code con extensión Python
- Git (para clonar el repositorio)

**¿Por qué Anaconda?**
- Gestión simplificada de entornos virtuales
- Instalación automática de dependencias científicas (NumPy, SciPy, etc.)
- Evita conflictos entre librerías
- Optimizaciones específicas para ciencia de datos

### Instalación y Configuración

1. **Clona el repositorio:**
```bash
git clone <URL_DEL_REPOSITORIO>
cd si_proyecto_unidad_2_final
```

2. **Crea y activa un entorno Anaconda:**
```bash
# Crear entorno con Python 3.9
conda create -n simpsons-classification python=3.9

# Activar el entorno
conda activate simpsons-classification
```

3. **Instala las dependencias:**
```bash
# Opción 1: Usando environment.yml (recomendado)
conda env create -f environment.yml
conda activate simpsons-classification

# Opción 2: Instalación manual con conda
conda create -n simpsons-classification python=3.9
conda activate simpsons-classification
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
pip install opencv-python
```

4. **Verifica la instalación:**
```bash
# Probar importaciones principales
python -c "import sklearn, cv2, numpy, pandas, matplotlib; print('✅ Todas las librerías instaladas correctamente')"
```

5. **Verifica la estructura de datos:**
```bash
# Debe mostrar las carpetas bart_simpson y homer_simpson
ls data/simpsons/training/
ls data/simpsons/test/
```

---

## 📓 Guía de Notebooks

### 🔵 1. histograma_color.ipynb - Sistema Baseline RGB
**Propósito**: Implementación base usando características cromáticas RGB

**Características principales:**
- Histogramas concatenados R+G+B (192 features)
- Comparación directa KNN vs SVM
- Visualizaciones de distribuciones cromáticas
- Matrices de confusión y métricas de rendimiento

**Cómo ejecutar:**
1. **Activar entorno**: `conda activate simpsons-classification`
2. Abrir notebook en Jupyter o VS Code
3. Ejecutar celdas secuencialmente
4. Revisar gráficos generados en `/output/`

**Tiempo estimado**: 10-15 minutos

---

### 🟡 2. histograma_hsv.ipynb - Sistema Optimizado HSV
**Propósito**: Evaluación con espacio de color HSV optimizado

**Características principales:**
- Histogramas bidimensionales Hue-Saturation (~3000 features)
- Robustez ante variaciones de iluminación
- Análisis de separabilidad cromática
- Comparación de rendimiento vs RGB

**Puntos clave:**
- Exclusión del canal Value para mayor robustez
- Visualización de distribuciones H-S bidimensionales
- Evaluación de impacto dimensional en algoritmos

**Tiempo estimado**: 15-20 minutos

---

### 🟢 3. histograma_lbp_descriptor.ipynb - Sistema Textural LBP
**Propósito**: Análisis con descriptores texturales Local Binary Patterns

**Características principales:**
- Codificación de texturas locales invariante a iluminación
- Primera configuración donde KNN≈SVM
- Análisis de limitaciones de features texturales puros
- Comparación con métodos cromáticos

**Configuración LBP:**
- Radius: 3 píxeles
- Neighbors: 24 puntos
- Método: uniform patterns

**Tiempo estimado**: 12-18 minutos

---

### 🟣 4. lbp_descriptor.ipynb - Sistema Híbrido HSV+LBP
**Propósito**: Combinación óptima de información cromática y textural

**Características principales:**
- Concatenación de features HSV + LBP (~3000+ features)
- Mejor rendimiento de KNN (74%)
- Análisis de complementariedad de características
- Evaluación de trade-offs complejidad vs rendimiento

**Resultados esperados:**
- SVM: ~84% accuracy (test independiente)
- KNN: ~74% accuracy (test independiente)
- Visualizaciones comparativas finales

**Tiempo estimado**: 20-25 minutos

---

## 🛠️ Flujo de Trabajo Recomendado

### Configuración Inicial (Una sola vez):
```bash
# 1. Activar entorno
conda activate simpsons-classification

# 2. Iniciar Jupyter desde el directorio del proyecto
cd path/to/si_proyecto_unidad_2_final
jupyter notebook

# O usar VS Code
code .
```

### Para Usuarios Nuevos:
1. **Comenzar con RGB** (`histograma_color.ipynb`)
   - Entender pipeline básico
   - Familiarizarse con estructura de datos
   - Revisar métricas de evaluación

2. **Explorar optimizaciones** (`histograma_hsv.ipynb`)
   - Comparar con baseline RGB
   - Analizar impacto de dimensionalidad
   - Observar degradación/mejora de algoritmos

3. **Investigar texturas** (`histograma_lbp_descriptor.ipynb`)
   - Entender convergencia KNN≈SVM
   - Analizar limitaciones de LBP puro
   - Comparar con métodos cromáticos

4. **Sistema completo** (`lbp_descriptor.ipynb`)
   - Evaluar combinación óptima
   - Revisar resultados finales
   - Generar visualizaciones comparativas

### Para Investigadores:
- **Entorno activo**: Siempre usar `conda activate simpsons-classification`
- Ejecutar notebooks en paralelo para comparaciones
- Modificar hiperparámetros para experimentación
- Adaptar código para otros datasets/problemas
- Extender con algoritmos adicionales

---

## 📈 Interpretación de Resultados

### Métricas Principales:
- **Accuracy**: Porcentaje de clasificaciones correctas
- **Precision/Recall**: Rendimiento balanceado por clase
- **F1-Score**: Media armónica precision-recall
- **Gap de Generalización**: Diferencia validación vs test

### Gráficos Generados:
- **Matrices de confusión**: Patrones de error por algoritmo
- **Histogramas cromáticos**: Distribuciones de características
- **Métricas comparativas**: Rendimiento KNN vs SVM
- **Evaluaciones de generalización**: Robustez de modelos

### Archivos de Salida:
```
output/
├── rgb_metrics_comparison.png          # Comparación RGB
├── hsv_metrics_comparison.png          # Comparación HSV  
├── lbp_performance_eval.png            # Evaluación LBP
├── lbp_hsv_performance_eval.png        # Evaluación híbrida
├── *_confusion_matrix_*.png            # Matrices de confusión
└── feature_distributions_*.png         # Distribuciones de características
```

---

## 🔧 Personalización y Extensiones

### Modificar Hiperparámetros:
```python
# En cualquier notebook, buscar estas secciones:
# KNN
knn_model = KNeighborsClassifier(
    n_neighbors=5,        # Modificar K
    weights='distance',   # Cambiar esquema de pesado
    metric='euclidean'    # Probar diferentes métricas
)

# SVM
svm_model = SVC(
    kernel='rbf',         # Probar 'linear', 'poly'
    C=10,                 # Ajustar regularización
    gamma='scale',        # Modificar parámetro kernel
    random_state=42
)
```

### Añadir Nuevos Algoritmos:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
```

### Experimentar con Features:
- Modificar bins de histogramas: `bins=32, 64, 128`
- Ajustar parámetros LBP: `radius=1,2,3` / `n_points=8,16,24`
- Probar combinaciones: RGB+LBP, RGB+HSV+LBP

---

## 📚 Referencias Académicas

### Documentación Técnica:
- **Informe completo**: `Informe.md` (35+ páginas)
- **Fundamentos teóricos**: Secciones 2.1-2.3 del informe
- **Metodología experimental**: Sección 3 del informe
- **Resultados detallados**: Secciones 4-7 del informe

### Librerías Utilizadas:
- **scikit-learn**: Algoritmos ML y métricas
- **OpenCV**: Procesamiento de imágenes y LBP
- **NumPy/Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualizaciones

---

## 🤝 Contribución y Colaboración

### Estructura para Contribuir:
1. Fork del repositorio
2. Crear branch para nueva feature: `git checkout -b feature/nueva-caracteristica`
3. Implementar cambios con documentación
4. Commit con mensajes descriptivos: `git commit -m "Add: nuevo algoritmo XYZ"`
5. Push y crear Pull Request

### Áreas de Mejora:
- [ ] Implementación de más algoritmos (Random Forest, Neural Networks)
- [ ] Optimización automática de hiperparámetros
- [ ] Extensión a más personajes/clases
- [ ] Interfaz web para clasificación interactiva
- [ ] Análisis de interpretabilidad de modelos

---

## 👥 Autores

**Equipo de Desarrollo:**
- Díaz Rodríguez, Carlo Franco
- Ramos Guerra, Ainhoa Jolie  
- Castrejón, Bringas Melanny Angeles

**Institución:** Universidad Nacional de Cajamarca  
**Curso:** Sistemas Inteligentes  
**Fecha:** Julio 2025

---

## 📄 Licencia

Este proyecto es desarrollado con fines académicos. Para uso comercial o redistribución, contactar a los autores.

---

## 🆘 Soporte y Troubleshooting

### Problemas Comunes:

**Error: "conda: command not found"**
- Instalar Anaconda desde: https://www.anaconda.com/download/
- Reiniciar terminal después de la instalación
- Verificar: `conda --version`

**Error: "No module named 'cv2'"**
```bash
conda activate simpsons-classification
conda install opencv
# o alternativamente:
pip install opencv-python
```

**Error: "Environment 'simpsons-classification' not found"**
```bash
# Crear el entorno nuevamente
conda create -n simpsons-classification python=3.9
conda activate simpsons-classification
# Reinstalar dependencias
```

**Error: "FileNotFoundError: data/simpsons/..."**
- Verificar estructura de carpetas
- Asegurar que las imágenes estén en las rutas correctas
- Ejecutar notebooks desde el directorio raíz del proyecto

**Error: "Memory error during model training"**
- Reducir número de bins en histogramas
- Usar subconjunto de datos para pruebas
- Reiniciar kernel de Jupyter

**Notebooks lentos:**
- Verificar que el entorno Anaconda esté activo
- Usar un entorno con más RAM
- Considerar usar Google Colab para recursos adicionales

**Problemas con kernels de Jupyter:**
```bash
# Agregar el entorno como kernel de Jupyter
conda activate simpsons-classification
pip install ipykernel
python -m ipykernel install --user --name simpsons-classification --display-name "Python (Simpsons)"
```

### Contacto:
Para dudas específicas del proyecto, revisar primero:
1. Este README
2. Comentarios en notebooks  
3. Sección de troubleshooting en `Informe.md`

---

**¿Listo para empezar? 🚀**  
Abre `notebooks/histograma_color.ipynb` y comienza tu análisis comparativo KNN vs SVM!
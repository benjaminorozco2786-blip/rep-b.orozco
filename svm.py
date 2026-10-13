import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. DATOS


# Cada fila representa: [Edad, Nivel de glucosa]

X = np.array([
    [20, 80],
    [25, 85],
    [30, 90],
    [32, 85],
    [35, 95],

    [40, 130],
    [45, 140],
    [50, 150],
    [53, 115],
    [55, 160],

    [60, 170],
    [65, 180],
    [28, 88],
    [30, 99],
    [48, 145]
])


# 0 = sano
# 1 = riesgo

# IMPORTANTE:debe existir una etiqueta por cada fila de X

y = np.array([
    0,   # [20, 80]
    0,   # [25, 85]
    0,   # [30, 90]
    0,   # [32, 85]
    0,   # [35, 95]

    1,   # [40, 130]
    1,   # [45, 140]
    1,   # [50, 150]
    1,   # [53, 115]
    1,   # [55, 160]

    1,   # [60, 170]
    1,   # [65, 180]
    0,   # [28, 88]
    0,   # [30, 99]
    1    # [48, 145]
])


# Comprobar que ambos tengan la misma cantidad
print("Cantidad de datos en X:", len(X))
print("Cantidad de etiquetas en y:", len(y))

# 2. DIVIDIR LOS DATOS

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 3. ESCALAR LOS DATOS

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. CREAR Y ENTRENAR EL MODELO SVM

modelo_svm = SVC(
    kernel="linear",
    C=1.0
)

modelo_svm.fit(X_train_scaled, y_train)


# 5. PREDICCIONES

y_pred = modelo_svm.predict(X_test_scaled)


# 6. EVALUACIÓN

precision = accuracy_score(y_test, y_pred)

print("\nRESULTADOS DEL MODELO")
print("---------------------")

print(f"Precisión del modelo: {precision * 100:.2f}%")

print(
    "Número de vectores de soporte por clase:",
    modelo_svm.n_support_
)

print("\nVectores de soporte identificados:")
print(modelo_svm.support_vectors_)

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Sano", "Riesgo"],
        zero_division=0
    )
)

# 7. GRÁFICA 1:FRONTERA DE DECISIÓN Y VECTORES DE SOPORTE

x_min = X_train_scaled[:, 0].min() - 1
x_max = X_train_scaled[:, 0].max() + 1

y_min = X_train_scaled[:, 1].min() - 1
y_max = X_train_scaled[:, 1].max() + 1


xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)


# Predecir todos los puntos de la malla
Z = modelo_svm.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 7))


# Dibujar regiones
plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25
)

# FRONTERA Y MÁRGENES

decision_values = modelo_svm.decision_function(
    np.c_[xx.ravel(), yy.ravel()]
)

decision_values = decision_values.reshape(xx.shape)


plt.contour(
    xx,
    yy,
    decision_values,
    levels=[-1, 0, 1],
    linestyles=["--", "-", "--"]
)

# PUNTOS DE ENTRENAMIENTO

nombres_clases = {
    0: "Sano",
    1: "Riesgo"
}

for clase in np.unique(y_train):

    plt.scatter(
        X_train_scaled[y_train == clase, 0],
        X_train_scaled[y_train == clase, 1],

        label=nombres_clases[clase],

        s=100
    )


# VECTORES DE SOPORTE

plt.scatter(
    modelo_svm.support_vectors_[:, 0],
    modelo_svm.support_vectors_[:, 1],

    s=250,

    facecolors="none",
    edgecolors="black",

    linewidths=2,

    label="Vectores de soporte"
)


plt.title(
    "SVM - Frontera de decisión y vectores de soporte"
)

plt.xlabel(
    "Edad (valor estandarizado)"
)

plt.ylabel(
    "Glucosa (valor estandarizado)"
)

plt.legend()

plt.grid(True)


# 8. GRÁFICA 2: DATOS DE PRUEBA Y PREDICCIONES


plt.figure(figsize=(10, 7))


for clase in np.unique(y_test):

    plt.scatter(

        X_test_scaled[y_test == clase, 0],

        X_test_scaled[y_test == clase, 1],

        label=f"Real: {nombres_clases[clase]}",

        s=120
    )


# Mostrar predicción al lado de cada punto

for i in range(len(X_test_scaled)):

    prediccion = nombres_clases[y_pred[i]]

    plt.text(

        X_test_scaled[i, 0] + 0.03,

        X_test_scaled[i, 1] + 0.03,

        f"Pred: {prediccion}",

        fontsize=10
    )


plt.title(
    "SVM - Datos de prueba y predicciones"
)

plt.xlabel(
    "Edad (valor estandarizado)"
)

plt.ylabel(
    "Glucosa (valor estandarizado)"
)

plt.legend()

plt.grid(True)

# 9. MOSTRAR GRÁFICAS

plt.show()
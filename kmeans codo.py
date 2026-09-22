from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#  DATOS DE LOS CLIENTES (proporcionada por una base de datos)
# [compras al mes, gasto mensual]

clientes = [
    [2, 300],
    [5, 220],
    [3, 280],
    [2, 150],
    [8, 200],
    [3, 400],
    [5, 1050],
    [6, 1200],
    [5, 900],
    [1, 2000],
    [3, 1800],
    [2, 1280],
    [9, 2500],
    [10, 2800],
    [8, 2300]
]
#  MÉTODO DEL CODO

inercias = []

# Vamos a probar desde K = 1 hasta K = 6
valores_k = range(1, 7)

for k in valores_k:

    modelo = KMeans(
        n_clusters=k,
        random_state=0
    )

    # Entrenar K-Means
    modelo.fit(clientes)

    # Guardar la inercia obtenida
    inercias.append(modelo.inertia_)


#  MOSTRAR GRÁFICA DEL CODO

plt.figure(figsize=(8, 5))

plt.plot(
    valores_k,
    inercias,
    marker="o"
)

plt.title("Método del Codo")

plt.xlabel("Número de grupos (K)")
plt.ylabel("Inercia")

plt.grid()

plt.show()

#  CREAR K-MEANS CON K = 3

# Después de método del codo
# seleccionamos K = 3

kmeans = KMeans(
    n_clusters=3,
    random_state=0
)

# metodo kmeans

kmeans.fit(clientes)


# Obtener el grupo de cada cliente
grupos = kmeans.labels_


# Obtener los centroides
centroides = kmeans.cluster_centers_


#  MOSTRAR RESULTADOS EN CONSOLA

print("\nCLASIFICACIÓN DE CLIENTES")
print("---------------------------------------")

for i in range(len(clientes)):

    print(
        "Cliente", i + 1,
        "| Compras:", clientes[i][0],
        "| Gasto: $", clientes[i][1],
        "| Grupo:", grupos[i]
    )

# MOSTRAR LOS CENTROIDES


print("\nCENTROIDES")
print("---------------------------------------")

for i in range(len(centroides)):

    print(
        "Grupo", i,
        "| Compras:", round(centroides[i][0], 2),
        "| Gasto: $", round(centroides[i][1], 2)
    )

# CREAR GRÁFICA DE K-MEANS


plt.figure(figsize=(8, 6))


# Dibujar clientes
for i in range(len(clientes)):

    compras = clientes[i][0]
    gasto = clientes[i][1]

    plt.scatter(
        compras,
        gasto,
        c=f"C{grupos[i]}",
        s=100
    )

# DIBUJAR CENTROIDES

for centro in centroides:

    plt.scatter(
        centro[0],
        centro[1],
        marker="X",
        s=250,
        c="black"
    )

# CONFIGURAR GRÁFICA

plt.title("Clasificación de clientes con K-Means")

plt.xlabel("Compras al mes")
plt.ylabel("Gasto mensual ($)")

plt.grid()

plt.show()
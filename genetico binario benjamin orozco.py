import random

# PARÁMETROS DEL ALGORITMO

TAMANO_POBLACION = 10
LONGITUD_CROMOSOMA = 10
NUM_GENERACIONES = 20
PROBABILIDAD_MUTACION = 0.1


# 1. CREAR UN INDIVIDUO

def crear_individuo():
    """ Crea un individuo formado por 10 bits aleatorios.
    Ejemplo: [1, 0, 1, 1, 0, 0, 1, 0, 1, 0] """

    individuo = []

    for i in range(LONGITUD_CROMOSOMA):
        bit = random.randint(0, 1)
        individuo.append(bit)

    return individuo


# 2. CREAR LA POBLACIÓN

def crear_poblacion():
    """ Crea una población de 10 individuos."""

    poblacion = []

    for i in range(TAMANO_POBLACION):
        individuo = crear_individuo()
        poblacion.append(individuo)

    return poblacion


# 3. FUNCIÓN FITNESS

def fitness(individuo):
    """ El fitness será la cantidad de unos que tenga
    el individuo. Ejemplo: [1, 0, 1, 1, 0, 1, 0, 0, 1, 1] Tiene 6 unos.
    Por lo tanto:fitness = 6"""

    return sum(individuo)


# 4. SELECCIÓN

def seleccion(poblacion):
    """Selección por torneo.Elegimos dos individuos al azar.
    El que tenga mayor fitness se convierte en padre."""

    individuo1 = random.choice(poblacion)
    individuo2 = random.choice(poblacion)

    if fitness(individuo1) > fitness(individuo2):
        return individuo1
    else:
        return individuo2


# 5. CRUZAMIENTO

def cruzamiento(padre1, padre2):
    """ Combina dos padres para crear un hijo.
     Ejemplo:
     Padre 1 = 1111100000
     Padre 2 = 0000011111
     Punto de corte = 5 Hijo = 1111111111 """

    punto = random.randint(1, LONGITUD_CROMOSOMA - 1)

    parte1 = padre1[:punto]
    parte2 = padre2[punto:]

    hijo = parte1 + parte2

    return hijo


# 6. MUTACIÓN

def mutacion(individuo):
    """Cada gen tiene una pequeña probabilidad de cambiar.
    0 puede convertirse en 1 y 1 puede convertirse en 0"""

    for i in range(LONGITUD_CROMOSOMA):

        if random.random() < PROBABILIDAD_MUTACION:

            if individuo[i] == 0:
                individuo[i] = 1
            else:
                individuo[i] = 0

    return individuo

# 7. MOSTRAR POBLACIÓN

def mostrar_poblacion(poblacion):

    for i, individuo in enumerate(poblacion):

        print(
            f"Individuo {i + 1}: "
            f"{individuo} "
            f"Fitness = {fitness(individuo)}"
        )

# 8. ALGORITMO GENÉTICO


# Creamos la población inicial

poblacion = crear_poblacion()


# Mostramos la población inicial

print("=" * 60)
print("POBLACIÓN INICIAL")
print("=" * 60)

mostrar_poblacion(poblacion)


# COMENZAMOS LAS GENERACIONES

for generacion in range(NUM_GENERACIONES):

    nueva_poblacion = []

    # Creamos tantos hijos como individuos tenga la población

    for i in range(TAMANO_POBLACION):

        # SELECCIÓN

        padre1 = seleccion(poblacion)
        padre2 = seleccion(poblacion)

        # CRUZAMIENTO


        hijo = cruzamiento(padre1, padre2)


        # MUTACIÓN
    

        hijo = mutacion(hijo)

        # Agregamos el hijo a la nueva población

        nueva_poblacion.append(hijo)

    # La nueva población reemplaza a la anterior

    poblacion = nueva_poblacion

    # BUSCAR EL MEJOR INDIVIDUO DE ESTA GENERACIÓN
  
    mejor = max(poblacion, key=fitness)

    print()
    print(
        f"Generación {generacion + 1}: "
        f"{mejor} "
        f"Fitness = {fitness(mejor)}"
    )


# 9. RESULTADO FINAL


mejor = max(poblacion, key=fitness)

print()
print("=" * 60)
print("RESULTADO FINAL")
print("=" * 60)

print("Mejor individuo:", mejor)
print("Fitness:", fitness(mejor))
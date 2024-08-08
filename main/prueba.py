import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class KNN:
    """Clasificador KNN.

    Parámetros
    ------------
    k : int
        número de vecinos cercanos
    p : int
        valor para selección de métrica (1: Manhattan, 2: Euclídea)
    """

    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p

    def distancia(self, vec_1, vec_2, p=2):
        dim = len(vec_1)
        distance=0

        for d in range(dim):
            distance += (abs(vec_1[d]-vec_2[d]))**p

        distance = (distance)**(1/p)
        return distance

    def fit(self, X, y):
        """Entrenamiento del clasificador kNN, es un algoritmo 'perezoso'
        sólo almacena los datos y sus etiquetas
        Parameters
        ----------
        X : array
            vector de características.
        y : array
            clases asociadas a los datos.
        """
        self.X = np.array(X) # hay una posibilidad de que esto haya pasarlo a array
        self.y = np.array(y)
        self.n_datos = X.shape[0] #estaria tomando el numero total de datos

    def predict(self, X_new):
        """Predice la clase de los nuevos datos.
        
        Parameters
        ----------
        X_new : array
            Nuevos vectores de características para clasificar.
        
        Returns
        -------
        y_pred : array
            Clases predichas para los nuevos datos.
        """
        y_pred = []
        for vec in X_new:
            # Calcular las distancias a todos los puntos de entrenamiento
            distancias = [self.distancia(vec, x_train, self.p) for x_train in self.X]
            
            k_distancias = np.argsort(distancias)

            k_etiqueta= self.y[k_distancias[:self.k]]

            c= Counter(k_etiqueta).most_common(1)
            y_pred.append(c[0][0])

        return y_pred


df_iris = load_iris(as_frame=True).frame
X = df_iris[ ['petal length (cm)', 'petal width (cm)'] ]
y = df_iris.target

#print('Etiquetas de clase:', np.unique(y))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1,  stratify = y) #lo que hace esta linea es dividir los datos en datos de
                                                                                                                    #entremiento y de test, en este caso es 70-30. (test_size)
#El random_state es la semilla
#el shuffle es para mezclar los datos asi te aseguras que no haya sesgo
#stratify basicamente lo que hace es que no hayan 90 datos de un tipo y solo 10 de otro por ej, los nivela.



# Normalización de los datos / estandarizacion
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# Creamos un objeto knn usando la clase implementada
knn = KNN(k=3)
# llamamos al método de entrenamiento ---> Datos de entrenamiento


knn.fit(X_train_std, y_train)



# Evaluamos el clasificador con los datos de prueba
y_pred = knn.predict(X_test_std)
# Comparamos nuestra predicción con los targets
(y_pred==y_test).sum()



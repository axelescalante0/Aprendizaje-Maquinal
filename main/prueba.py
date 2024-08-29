import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df_prestamos = pd.read_csv("./1_datos/prestamos.csv", index_col=False)
df_prestamos.head()

train_set, test_set = train_test_split(df_prestamos, test_size=0.2, random_state=42, stratify=df_prestamos["estado"])

# hacemos una copia del conjunto de entrenamiento
housing = train_set.copy()

x_train = housing.drop("estado", axis=1)
x_train_labels = housing["estado"].copy()

print('Etiquetas de clase:', np.unique(x_train_labels))

faltante = df_prestamos.isna().sum() #si hay NA los sume, basicamente un ftable de R
print(faltante,"\n")

datos_faltantes = df_prestamos[df_prestamos.isnull().any(axis=1)]
datos_faltantes.head() #Algunos datos faltantes

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") #Para remplazar los valores faltante con la mediana
imputer.fit(x_train)

print(imputer.statistics_,"\n")

#Ahora se puede usar este imputador "entrenado" para transformar el conjunto de entrenamiento reemplazando los valores faltantes con las medianas calculadas:

X_train = imputer.transform(x_train) # me devuelve un numpy array
type(X_train)

print(imputer.feature_names_in_) # Muestra los nombres de las columnas que fueron pasadas al imputador

df_x_train_num = pd.DataFrame(X_train, columns=x_train.columns, index=x_train.index)
#null_rows_idx = x_train.isnull().any(axis=1)
#df_x_train_num.loc[null_rows_idx].head()

x_train_cat = train_set.loc[:,["estado"]]
x_train_cat.head(8)

#Probamos clase ordinal de sckit-learn

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
x_train_cat_encoded = ordinal_encoder.fit_transform(x_train_cat)

x_train_cat_encoded[:8]

#Clase one_hot
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
x_train_cat_1hot = cat_encoder.fit_transform(x_train_cat)
x_train_cat_1hot

x_train_cat_1hot.toarray() #

print(cat_encoder.feature_names_in_) #devuelve el nombre de la columna

print(cat_encoder.get_feature_names_out()) #devuelve un nombre de columna con cada estado


#Creamos un Data Frame con las clases ya transformadas a Dummy

df_x_train_cat_1hot = pd.DataFrame(x_train_cat_1hot.toarray(), columns=cat_encoder.get_feature_names_out(), index=x_train_cat.index)
print(df_x_train_cat_1hot)

#Concatenacion de Dataframe de las etiquetas con el que contiene las variables numericas

df_train = pd.concat([df_x_train_num, df_x_train_cat_1hot], axis=1)
df_train

Y_test = test_set.copy()
Y_test

y_test = Y_test.drop("estado", axis=1)
y_test_labels = Y_test["estado"].copy()

#Imputamos datos faltantes

X_test = imputer.transform(y_test) # me devuelve un numpy array
df_x_test_num = pd.DataFrame(X_test, columns=y_test.columns, index=y_test.index)


#pasamos las variables "estado" a dummy y creamos un dataframe con las variables categoricas

x_test_cat = test_set.loc[:,["estado"]]
x_test_cat.head()

x_test_cat_1hot = cat_encoder.transform(x_test_cat)

df_test_cat = pd.DataFrame(x_test_cat_1hot.toarray(), columns=cat_encoder.get_feature_names_out(), index=x_test_cat.index)
df_test_cat


#Creamos el dataframe final para test
df_test = pd.concat([df_x_test_num, df_test_cat], axis=1)
df_test


Y_train_target = df_train["estado_pagado"]
Y_train_target

# Dividir el conjunto de entrenamiento original en nuevo conjunto de entrenamiento y validación
X_valid = df_train[:20000]
y_valid = Y_train_target[:20000]

X_train_new = df_train[20000:]
y_train_new = Y_train_target[20000:]

#Pasamos a Array los data frame (Queda sujeta a revison para optimizar esta parte)
X_train_new = X_train_new.to_numpy()
y_train_new = y_train_new.to_numpy()
X_valid_array = X_valid.to_numpy()


from sklearn.neighbors import KNeighborsClassifier

#probamos el clasificador con k=3
knn_clasificador = KNeighborsClassifier(n_neighbors=3)

# Entrenamos el clasificador con el conjunto de entrenamiento
knn_clasificador.fit(X_train_new, y_train_new)

# Predecir las etiquetas en el conjunto de validación
y_pred_valid = knn_clasificador.predict(X_valid_array)


# Calcular el numero de aciertos
aciertos = (y_pred_valid == y_valid).sum()

total_etiquetas = len(y_valid)

#Calculamos la tasa de aciertos
tasa_aciertos = aciertos / total_etiquetas
print(f"Tasa de aciertos del clasificador: {tasa_aciertos * 100:.2f}%")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_valid, y_pred_valid)
accuracy

presiciones = []
#con los 20k datos tarda 56s aprox
k_values = range(1, 21) #rango

for k in k_values:
    # Crea y entrena el clasificador con el valor actual de K
    knn_clasificador = KNeighborsClassifier(n_neighbors=k)
    knn_clasificador.fit(X_train_new, y_train_new)
    
    # Predecir las etiquetas en el conjunto de validacion
    y_pred_valid = knn_clasificador.predict(X_valid_array)
    
    #Calculamos el accuracy
    accuracy = accuracy_score(y_valid, y_pred_valid)
    presiciones.append(accuracy) #la guardamos

    
# Graficar la tasa de aciertos en función de K
plt.figure(figsize=(13, 6))
plt.plot(k_values, presiciones, marker="o")
plt.title("Accuracy en función de K")
plt.xlabel("Número de vecinos cercanos (K)")
plt.ylabel("Presición")
plt.xticks(k_values)
plt.grid(True)
plt.show()







# class KNN:
#     """Clasificador KNN.

#     Parámetros
#     ------------
#     k : int
#         número de vecinos cercanos
#     p : int
#         valor para selección de métrica (1: Manhattan, 2: Euclídea)
#     """

#     def __init__(self, k=3, p=2):
#         self.k = k
#         self.p = p

#     def distancia(self, vec_1, vec_2, p=2):
#         dim = len(vec_1)
#         distance=0

#         for d in range(dim):
#             distance += (abs(vec_1[d]-vec_2[d]))**p

#         distance = (distance)**(1/p)
#         return distance

#     def fit(self, X, y):
#         """Entrenamiento del clasificador kNN, es un algoritmo 'perezoso'
#         sólo almacena los datos y sus etiquetas
#         Parameters
#         ----------
#         X : array
#             vector de características.
#         y : array
#             clases asociadas a los datos.
#         """
#         self.X = np.array(X) # hay una posibilidad de que esto haya pasarlo a array
#         self.y = np.array(y)
#         self.n_datos = X.shape[0] #estaria tomando el numero total de datos

#     def predict(self, X_new):
#         """Predice la clase de los nuevos datos.
        
#         Parameters
#         ----------
#         X_new : array
#             Nuevos vectores de características para clasificar.
        
#         Returns
#         -------
#         y_pred : array
#             Clases predichas para los nuevos datos.
#         """
#         y_pred = []
#         for vec in X_new:
#             # Calcular las distancias a todos los puntos de entrenamiento
#             distancias = [self.distancia(vec, x_train, self.p) for x_train in self.X]
            
#             k_distancias = np.argsort(distancias)

#             k_etiqueta= self.y[k_distancias[:self.k]]

#             c= Counter(k_etiqueta).most_common(1)
#             y_pred.append(c[0][0])

#         return y_pred


# # df_iris = load_iris(as_frame=True).frame
# # X = df_iris[ ['petal length (cm)', 'petal width (cm)'] ]
# # y = df_iris.target

# # #print('Etiquetas de clase:', np.unique(y))


# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1,  stratify = y) #lo que hace esta linea es dividir los datos en datos de
# #                                                                                                                     #entremiento y de test, en este caso es 70-30. (test_size)
# # #El random_state es la semilla
# # #el shuffle es para mezclar los datos asi te aseguras que no haya sesgo
# # #stratify basicamente lo que hace es que no hayan 90 datos de un tipo y solo 10 de otro por ej, los nivela.



# # # Normalización de los datos / estandarizacion
# # from sklearn.preprocessing import StandardScaler
# # sc = StandardScaler()
# # sc.fit(X_train)
# # X_train_std = sc.transform(X_train)
# # X_test_std = sc.transform(X_test)


# # # Creamos un objeto knn usando la clase implementada
# # knn = KNN(k=3)
# # # llamamos al método de entrenamiento ---> Datos de entrenamiento


# # knn.fit(X_train_std, y_train)



# # # Evaluamos el clasificador con los datos de prueba
# # y_pred = knn.predict(X_test_std)
# # # Comparamos nuestra predicción con los targets
# # (y_pred==y_test).sum()

# # y_pred = knn.predict(X_test_std)

# # def accuracy(y_pred, y_test):
# #     return np.sum( np.equal(y_pred, y_test) ) / len(y_test)

# # print(accuracy(y_pred, y_test))


# #usando knn coso
# df_prestamos = pd.read_csv("./1_datos/prestamos.csv", index_col=False)
# df_prestamos.head()

# df_prestamos.info()
# print("\n")
# df_prestamos.describe()

# train_set, test_set = train_test_split(df_prestamos, test_size=0.2, random_state=42, stratify=df_prestamos["estado"])

# # hacemos una copia del conjunto de entrenamiento
# housing = train_set.copy()


# x_train = housing.drop("estado", axis=1)
# x_train_labels = housing["estado"].copy()

# print('Etiquetas de clase:', np.unique(x_train_labels))

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median") #Para remplazar los valores faltante con la mediana
# imputer.fit(x_train)

# print(imputer.statistics_,"\n")

# #Ahora se puede usar este imputador "entrenado" para transformar el conjunto de entrenamiento reemplazando los valores faltantes con las medianas calculadas:

# x_train_1 = imputer.transform(x_train) # me devuelve un numpy array
# type(x_train_1)

# print(imputer.feature_names_in_) # Muestra los nombres de las columnas que fueron pasadas al imputador

# df_x_train_num = pd.DataFrame(x_train_1, columns=x_train.columns, index=x_train.index)
# #null_rows_idx = x_train.isnull().any(axis=1)
# #df_x_train_num.loc[null_rows_idx].head()

# x_train_cat = train_set.loc[:,["estado"]]
# x_train_cat.head(8)

# #Clase one_hot
# from sklearn.preprocessing import OneHotEncoder

# cat_encoder = OneHotEncoder()
# x_train_cat_1hot = cat_encoder.fit_transform(x_train_cat)
# x_train_cat_1hot

# x_train_cat_1hot.toarray() #

# print(cat_encoder.feature_names_in_) #devuelve el nombre de la columna

# print(cat_encoder.get_feature_names_out()) #devuelve un nombre de columna con cada estado

# #Creamos un Data Frame con las clases ya transformadas a Dummy

# df_x_train_cat_1hot = pd.DataFrame(x_train_cat_1hot.toarray(), columns=cat_encoder.get_feature_names_out(), index=x_train_cat.index)
# print(df_x_train_cat_1hot)

# #Concatenacion de Dataframe de las etiquetas con el que contiene las variables numericas

# df_train = pd.concat([df_x_train_num, df_x_train_cat_1hot], axis=1)
# df_train



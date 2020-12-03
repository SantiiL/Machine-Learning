# # Ejemplo Regresion
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

casas = pd.read_csv("precios_casas.csv")

casas.head()
casas_x = casas.drop("median_house_value", axis=1)
casas_x.head()

casas_y = casas['median_house_value']
casas_y.head()

X_train,X_test,y_train,y_test = train_test_split(casas_x,casas_y,test_size=0.3)

X_train.head()

X_test.head()

from sklearn.preprocessing import MinMaxScaler

normalizador = MinMaxScaler()

normalizador.fit(X_train)

X_train.head()

X_train = pd.DataFrame(data=normalizador.transform(X_train),columns=X_train.columns,index=X_train.index)
#sobre escribe la variable X_train con los valores normalizados

X_train.head() #todos los valores estan normalizados entre 0 y 1

X_test = pd.DataFrame(data=normalizador.transform(X_test),columns=X_test.columns,index=X_test.index)

X_test.head()

casas.columns

longitude = tf.feature_column.numeric_column("longitude")
latitude = tf.feature_column.numeric_column("latitude")
housing_median_age = tf.feature_column.numeric_column("housing_median_age")
total_rooms = tf.feature_column.numeric_column("total_rooms")
total_bedrooms = tf.feature_column.numeric_column("total_bedrooms")
population = tf.feature_column.numeric_column("population")
households = tf.feature_column.numeric_column("households")
median_income = tf.feature_column.numeric_column("median_income")

columnas = [longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,
           households,median_income]
columnas

funcion_entrada = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10
                                                      ,num_epochs=1000,shuffle=True)

modelo = tf.estimator.DNNRegressor(hidden_units=[10,10,10],feature_columns=columnas)

modelo.train(input_fn=funcion_entrada, steps = 8000)

funcion_entrada_prediccion = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,
                                                                 num_epochs=1,shuffle=False)

generador_predicciones = modelo.predict(funcion_entrada_prediccion)

predicciones = list(generador_predicciones)
predicciones

predicciones_finales =[]
for prediccion in predicciones:
    predicciones_finales.append(prediccion['predictions'])
predicciones_finales

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,predicciones_finales)**0.5

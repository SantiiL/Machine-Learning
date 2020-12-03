# # Ejemplo de Clasificacion
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

pwd
ingresos = pd.read_csv('ingresos.csv')
ingresos

ingresos["income"].unique()

def cambio_valor(valor):
    if valor == '<=50K':
        return 0
    else:
        return 1

ingresos['income'] = ingresos['income'].apply(cambio_valor) # lo que hago es clasificar a las personas que ganan + o - de 50k

ingresos.head()

from sklearn.model_selection import train_test_split

datos_x = ingresos.drop('income',axis=1) #caracteristicas del conjunto de datos, menos la columna a predecir
            #el drop saca income, el axis=1 hace que se salga esa columna

datos_y = ingresos['income']
datos_y

X_train, X_test,y_train, y_test = train_test_split(datos_x,datos_y,test_size=0.3)

ingresos.columns

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",['Female , Male']) #utilizamos vocabulary porque
                                                                        #sabemos los posibles valores de la columna

occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
#Se utiliza el hash bucket cuando no sabes la cantidad de elementos que tiene la columna.

marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status",hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native-country",hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=1000)
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)

age = tf.feature_column.numeric_column("age")
fnlwgt = tf.feature_column.numeric_column("fnlwgt")
educational_num = tf.feature_column.numeric_column("educational-num")
capital_gain = tf.feature_column.numeric_column("capital-gain")
capital_loss = tf.feature_column.numeric_column("capital-loss")
hours_per_week = tf.feature_column.numeric_column("hours-per-week")

columnas_categorias = [gender,occupation,marital_status,relationship,education,native_country,workclass,age,fnlwgt,
                       educational_num,capital_gain,capital_loss,hours_per_week]

funcion_entrada = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)

modelo = tf.estimator.LinearClassifier(feature_columns=columnas_categorias)

modelo.train(input_fn = funcion_entrada,steps=8000)

funcion_prediccion = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
                            #el batch_size es igual al largo de x_test para que reccorra todo test

generador_predicciones = modelo.predict(input_fn=funcion_prediccion)

predicciones = list(generador_predicciones)

predicciones

predicciones_finales = [prediccion['class_ids'][0] for prediccion in predicciones]

predicciones_finales

from sklearn.metrics import classification_report

print (classification_report(y_test,predicciones_finales))

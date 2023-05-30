import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler # normalização de dados
from sklearn.model_selection import train_test_split # dividir dados entre treinmaneto e teste
from sklearn.metrics import r2_score #verificar o erro entre a predição e o valor real
from sklearn.neural_network import MLPRegressor
import datetime as dt

dataset = pd.read_csv("./ibov_2017_2023.csv")

dataset = dataset.drop('Vol.', axis=1)
dataset = dataset.drop('Data', axis=1)

dataset[['Último_amanha']] = dataset[['Último']].shift(-1)
dataset['Retorno'] = dataset['Último_amanha'] - dataset['Último']
dataset = dataset.drop('Var%', axis=1)

escala = StandardScaler()

for c in dataset.columns:
    dataset[c+'_Norm'] = escala.fit_transform(dataset[c].to_numpy().reshape(-1, 1))
dataset = dataset.dropna()


X_norm_train, X_norm_test, Y_train, Y_test = train_test_split(dataset[['Último_Norm']], dataset[['Último_amanha_Norm']], test_size=0.3)

rna = MLPRegressor(hidden_layer_sizes=(10, 5),
                   max_iter=2000,
                   tol=0.00000001,
                   learning_rate_init=0.1,
                   solver="sgd",
                   activation="logistic",
                   learning_rate="constant",
                   verbose=2)

rna.fit(X_norm_train, Y_train)

Y_rna_previsao = rna.predict(X_norm_test)

r2_rna = r2_score(Y_test, Y_rna_previsao)

print("R2 RNA:", r2_rna)

X_test = escala.inverse_transform(X_norm_test)

plt.plot(X_test, Y_test, alpha=0.5, label="Reais")
plt.plot(X_test, Y_rna_previsao, alpha=0.5, label="MLP")
plt.xlabel("Último antes")
plt.ylabel("Último amanhã")
plt.title("Comparação entre algoritmos previstos")
plt.legend(loc=1)
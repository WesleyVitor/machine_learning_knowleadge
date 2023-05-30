import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler # normalização de dados
from sklearn.model_selection import train_test_split # dividir dados entre treinmaneto e teste
from sklearn.metrics import r2_score #verificar o erro entre a predição e o valor real
from sklearn.neural_network import MLPRegressor

#Carregamento do dataset

dataset = pd.read_csv("./auto-mpg.csv")

def plot_grafic(dataset):
    plt.scatter(dataset[['weight']], dataset[['mpg']])
    plt.xlabel("Peso (libras)")
    plt.ylabel("Autonomia (mpg)")
    plt.title("Relação entre o peso e a autonomia dos veículos")


#Pré-processamento

X = dataset[['weight']]
Y = dataset[['mpg']]

#Transformação de libras para Kg
dataset[['weight']] = dataset[['weight']] * 0.453592

#Milhas/galao para Km/litro
dataset[['mpg']] = dataset[['mpg']] * 0.425144

#Normalização de dados para 0->1, a fim de evitar distorcoes no processamento

escala = StandardScaler()
escala.fit(X)

X_norm = escala.transform(X) #Coloca o valor entre 0 e 1

#Dividir entre treinamento e teste
X_norm_train, X_norm_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.3)

#Precessamento

rna = MLPRegressor(hidden_layer_sizes=(10, 5),
                   max_iter=2000,
                   tol=0.00000001,
                   learning_rate_init=0.1,
                   solver="sgd",
                   activation="logistic",
                   learning_rate="constant",
                   verbose=2)

rna.fit(X_norm_train, Y_train) #treina a IA

# Pós-processamento

Y_rna_previsao = rna.predict(X_norm_test) #prever valores com base no conjunto separado para testes

#Cálculo do R*2, ou seja, qual próximo ficou dos dados reais, quanto mais próximo de 1 melhor

r2_rna = r2_score(Y_test, Y_rna_previsao)

print("R2 RNA:", r2_rna)


X_test = escala.inverse_transform(X_norm_test) 

def plot_grafic_algoritm():
    plt.scatter(X_test, Y_test, alpha=0.5, label="Reais")
    plt.scatter(X_test, Y_rna_previsao, alpha=0.5, label="MLP")
    plt.xlabel("Peso (libras)")
    plt.ylabel("Autonomia (mpg)")
    plt.title("Comparação entre algoritmos previstos")
    plt.legend(loc=1)

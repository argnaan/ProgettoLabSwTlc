####################################################################################################
#
#   Progetto d'esame per il corso Laboratorio di Software per le telecomunicazioni
#   Andrea Argnani
#   2023/2024
#
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# importazione dei dati
df = pd.read_csv('housing.csv')
print(df.head(12))

# data preprocessing
for column in df.columns:
    n_missing = df[column].isnull().sum()
    print(f"{column} -> {n_missing} missing values")

# sostituzione dei valori mancanti di total_bedrooms con un valore proporzionale a total_rooms
df_p = df.dropna(subset=['total_bedrooms'])
BoR = df_p['total_bedrooms'] / df_p['total_rooms']
print(f"Media di BoR: {BoR.mean()}\nDeviazione standard di BoR: {BoR.std()}")
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_rooms'] * BoR.mean())

# ricerca di misspelling nelle variabili non numeriche:
print(df['ocean_proximity'].unique())

# ricerca di outlier
for column in df.columns:
    if column != 'ocean_proximity':
        pass
        # sns.boxplot(data=df[column])
        # plt.title(column)
        # plt.show()


# data format
print(df.dtypes)
df = pd.get_dummies(data=df, columns=['ocean_proximity'], drop_first=False)
print(df.dtypes)
print(df.head(12))


# regressione lineare semplice
df_reg = df
X = df_reg.drop(columns=['median_house_value'])
Y = df_reg['median_house_value']

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_hat = regressor.predict(X_test)

# prestazioni del modello

mse = np.mean((y_hat - Y_test)**2)
mae = np.mean(np.abs(y_hat - Y_test))
print(f"MSE: {mse}")
print(f"Root MSE: {np.sqrt(mse)}")
print(f"MAE: {mae}")
print(f"R2 score {regressor.score(X_test, Y_test)}")
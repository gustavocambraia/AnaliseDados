import displayfunction
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np


df = pd.read_csv('advertising.csv')
print(df)

# Verificar os tipos de dados e se existem dados nulos
print(df.info())

sns.heatmap(df.corr(), cmap='Wistia', annot=True)
plt.show()

# definindo inputs do modelo (eixo x) e os outputs (eixo y)
x = df.drop('Vendas', axis=1)
y = df['Vendas']

# criar variáveis para serem utilizadas para treino da IA e variáveis para testar a eficácia
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

# Treinar a IA
lin_reg = LinearRegression()
lin_reg.fit(x_treino, y_treino)

rf_reg = RandomForestRegressor()
rf_reg.fit(x_treino, y_treino)

# Teste de IA
teste_lin = lin_reg.predict(x_teste)
teste_rf = rf_reg.predict(x_teste)

# Indicadores de resultados da Regressão Linear e do Random Forest
r2_lin = metrics.r2_score(y_teste, teste_lin)
rmse_lin = np.sqrt(metrics.mean_squared_error(y_teste, teste_lin))
print(f'R² da Regressão Linear: {r2_lin}')
print(f'RMSE da Regressão Linear: {rmse_lin}')

r2_rf = metrics.r2_score(y_teste, teste_rf)
rmse_rf = np.sqrt(metrics.mean_squared_error(y_teste, teste_rf))
print(f'R² do Random Forest: {r2_rf}')
print(f'RMSE do Random forest: {rmse_rf}')
# Aqui é possível perceber que o Random Forest acerta mais e erra menos

# Análise Gráfica
df_resultado = pd.DataFrame()
df_resultado['y_teste'] = y_teste
df_resultado['y_previsao_rf'] = teste_rf
# df_resultado['y_previsao_lin'] = teste_lin
df_resultado = df_resultado.reset_index(drop=True)
fig = plt.figure(figsize=(15, 5))
sns.lineplot(data=df_resultado)
plt.show()
displayfunction.display(df_resultado)

# Verifica importância das variáveis
importancia_features = pd.DataFrame(rf_reg.feature_importances_, x_treino.columns)
plt.figure(figsize=(5, 5))
sns.barplot(x=importancia_features.index, y=importancia_features[0])
plt.show()



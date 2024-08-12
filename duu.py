
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df= pd.read_csv("sber1.csv",sep=';',usecols=['<DATE>','<CLOSE>'])

df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%d/%m/%y')
df.set_index('<DATE>', inplace = True)
df.index = pd.to_datetime(df.index)



train = df[:'2023-12']
test = df['2024-01':]
#plt.plot(train, color="black")
#plt.plot(test, color="red")

# заголовок и подписи к осям
# /plt.title('Разделение данных о перевозках на обучающую и тестовую выборки')
##plt.xlabel('Месяцы')

# добавим сетку$


# обучим модель с соответствующими параметрами, SARIMAX(3, 0, 0)x(0, 1, 0, 12)
# импортируем класс модели
from statsmodels.tsa.arima.model import ARIMA

# создадим объект этой модели
model = ARIMA(train,
                order=(0, 2,1 ))

# применим метод fit
result = model.fit()
start = len(train)


# и закончится в конце тестового
end = len(train) + len(test)-1

# применим метод predict

predictions = result.predict(start,end)
start = len(df)

# и закончится 36 месяцев спустя
end = (len(df) - 1) + 3 * 12

# теперь построим прогноз на три года вперед
forecast = result.predict(start, end)

# посмотрим на весь 1963 год
forecast[-12:]
# выведем две кривые (фактические данные и прогноз на будущее)
plt.plot(df, color='black')
plt.plot(forecast, color='blue')

# заголовок и подписи к осям
plt.title('Фактические данные и прогноз на будущее')
plt.ylabel('Стоимость акции')
plt.xlabel('Месяцы')

# добавим сетку
plt.grid()
plt.show()
ax = df.plot(figsize = (12,6), legend = None)
ax.set(title = 'Стоимость акции сбербанка с 01.01.2020 по 01.04.2024', xlabel = 'Месяцы', ylabel = 'Цена')
plt.show()

from sklearn.metrics import mean_squared_error

# рассчитаем MSE
print(mean_squared_error(test, predictions))

# и RMSE
print(np.sqrt(mean_squared_error(test, predictions)))








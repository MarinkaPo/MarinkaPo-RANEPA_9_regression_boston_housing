import streamlit as st
import pandas as pd
import shap
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Прогноз цен на недвижимость в Бостоне.

(используется регрессионная модель RandomForestRegressor)
""")

img = Image.open('Boston_house_prices.png')
st.image(img, use_column_width=True)

# -----Об аббериватурах-----
expander_bar = st.expander("Расшифровка сокращений")
expander_bar.markdown(
    """
* **CRIM**: количество преступлений на душу населения,
* **ZN**: доля земли под жилую застройку (для участков более 25 тыс.кв.футов),
* **INDUS**: доля акров неторгового бизнеса на город,
* **CHAS**: граничит ли участок с рекой Чарльз,
* **NO**: концентрация оксидов азота в воздухе,
* **RM**: количество жилых комнат,
* **AGE**: доля домов в округе старше 1940 года постройки,
* **DIS**: среднее расстрояние до бизнес-районов Бостона,
* **RAD**: индекс доступности до главных дорог города,
* **TAX**: будущий налог на имущество (на каждые 10 тыс.долларов),
* **PTRATIO**: соотношение кол-ва учеников на одного учителя в районе,
* **B**: доля чернокожего населения в районе,
* **LSTAT**: доля (в %) нищего населения в районе,
* **MEDV**: наш **таргет** - предсказанная стоимость жилья.
"""
)

# -----Загружаем Boston House Price Dataset-----

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = ['CRIM', 'ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
X = pd.DataFrame(data, columns=feature_names)
Y = pd.DataFrame(target, columns=["MEDV"])

# -----Боковая панель-----
# -----Корректируем заголовок у боковой панели-----
st.sidebar.header('Выберите параметры')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# -----ОСНОВНАЯ ПАНЕЛЬ-----

# -----Выводим заданные входные параметры-----
st.header('Заданные входные параметры')
st.write(df)
st.write('---')

# -----Задаём регрессионную модель-----
model = RandomForestRegressor()
model.fit(X, Y)
# -----Предикт-----
prediction = model.predict(df)

st.header('Предсказанная цена (тыс.долларов):')
st.write(prediction)
st.write('---')

# -----Explaining the model's predictions using SHAP values-----
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Используем SHAP values - визуальное отображение значимость каждой фичи для предсказания:')

st.subheader('...в виде summary_plot:')
# plt.title('В виде summary_plot')
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X)
st.pyplot(fig, bbox_inches='tight')
st.write('---')

st.subheader('...в виде bar_plot:')
# plt.title('В виде bar_plot')
f, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(f, bbox_inches='tight')

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def predictProba(sex,age,class_job,heavy,heat,noise,exp,aud_left_low,aud_left_up,aud_right_low,aud_right_up,glucose,holesterin,lpnp,lpvp,ia,pain):
    data = np.array([[sex,age,class_job,heavy,heat,noise,exp,aud_left_low,aud_left_up,aud_right_low,aud_right_up,glucose,holesterin,lpnp,lpvp,ia,pain]])
    return model.predict_proba(data)

def predictDisease(sex,age,class_job,heavy,heat,noise,exp,aud_left_low,aud_left_up,aud_right_low,aud_right_up,glucose,holesterin,lpnp,lpvp,ia,pain):
    data = np.array([[sex,age,class_job,heavy,heat,noise,exp,aud_left_low,aud_left_up,aud_right_low,aud_right_up,glucose,holesterin,lpnp,lpvp,ia,pain]])
    return model.predict(data)

def load_model():
    hear = pd.read_excel('hearing_impairment.xlsx')
    hear['Стаж (количество лет)'] = hear['Стаж (количество лет)'].fillna(int(hear['Стаж (количество лет)'].mean()))
    hear['Тяжесть'] = hear['Тяжесть'].fillna(int(hear['Тяжесть'].mean()))
    hear['Нагрев'] = hear['Нагрев'].fillna(int(hear['Нагрев'].mean()))
    hear['Шум'] = hear['Шум'].fillna(int(hear['Шум'].mean()))
    hear['Глюкоза, ммоль/л'] = hear['Глюкоза, ммоль/л'].fillna(int(hear['Глюкоза, ммоль/л'].mean()))
    hear['Общий холестерин, ммоль/л'] = hear['Общий холестерин, ммоль/л'].fillna(int(hear['Общий холестерин, ммоль/л'].mean()))
    hear['ЛПНП, ммоль/л'] = hear['ЛПНП, ммоль/л'].fillna(int(hear['ЛПНП, ммоль/л'].mean()))
    hear['ЛПВП, ммоль/л'] = hear['ЛПВП, ммоль/л'].fillna(int(hear['ЛПВП, ммоль/л'].mean()))
    hear['Индекс атерогенности'] = hear['Индекс атерогенности'].fillna(int(hear['Индекс атерогенности'].mean()))
    hear['Жалобы на боль в анамнезе (ШОП)'] = hear['Жалобы на боль в анамнезе (ШОП)'].fillna(int(hear['Жалобы на боль в анамнезе (ШОП)'].mean()))
    hear = hear.drop('Профессия (должность)', axis=1)
    X = hear.drop('Наруш слух', axis=1)
    y = hear['Наруш слух']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание модели случайного леса
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Обучение модели
    model.fit(X_train, y_train)
    return model

model = load_model()
#model.predict(X_test) #предсказываем на основе данных и фич параметры заболевания

st.title('Прогнозирование риска потери слуха у работников заводских и промышленных предприятий')

st.subheader("Введите Ваши данные")

options1 = ['жен', 'муж']
sex = st.selectbox('Пол', options1)

age = st.number_input('Возраст')
options = [1, 2, 3, 4]
class_job = st.selectbox("Выберите классификацию труда:", options)

exp = st.number_input('Стаж (количество лет)')

st.subheader("Укажите наличие данных факторов на Вашем месте работы")

heavy = st.checkbox('Тяжесть')
heat = st.checkbox('Нагрев')
noise= st.checkbox('Шум')


st.subheader("Укажите показатели аудиометрии")

aud_left_low = st.number_input('Аудиометрия (справа) НЧ, дБ')
aud_left_up = st.number_input('Аудиометрия (справа) ВЧ, дБ')
aud_right_low = st.number_input('Аудиометрия (слева) НЧ, дБ') 
aud_right_up = st.number_input('Аудиометрия (слева) ВЧ, дБ') 

st.subheader("Укажите параметры вашего анализа: биохимический анализ крови")

glucose = st.number_input('Глюкоза, ммоль/л')
holesterin = st.number_input('Общий холестерин, ммоль/л')
lpnp = st.number_input('ЛПНП, ммоль/л')
lpvp = st.number_input('ЛПВП, ммоль/л')
ia = st.number_input('Индекс атерогенности')

st.subheader("Укажите иные признаки")
pain = st.checkbox('Есть ли жалобы на боль в ушах?')

done = st.button('Вычислить риски')



if done:
    res_heavy = 1 if heavy else 0
    res_heat = 1 if heat else 0
    res_noise = 1 if noise else 0

    res_pain = 1 if pain else 0

    if sex == "муж":
        sex_value = 1
    else:
        sex_value = 0

    

    result = predictProba(sex_value,age,class_job,res_heavy,res_heat,res_noise,exp,aud_left_low,aud_left_up,aud_right_low,aud_right_up,glucose,holesterin,lpnp,lpvp,ia,res_pain)
    rec = predictDisease(sex_value,age,class_job,res_heavy,res_heat,res_noise,exp,aud_left_low,aud_left_up,aud_right_low,aud_right_up,glucose,holesterin,lpnp,lpvp,ia,res_pain)
    if rec is None:
        st.error("Не удалось рассчитать.")
    else:
        if rec == 1:
            rec_value = 'Есть риск нарушения слуха! Рекомендуется немедленное посещение врача'
        else:
            rec_value = 'Риск потери слуха маловероятен'
        st.text(rec_value)


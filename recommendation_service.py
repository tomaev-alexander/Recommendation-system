import os
from typing import List
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from schema import PostGet
import pickle

# Загрузка файла с ML моделью на сервер/LMS
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # Проверяем, где выполняется код в системе/сайте LMS, или локально
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    # model_path = get_model_path("/my/super/path")
    model_path = get_model_path("sklearn_model_x_post_XG.pkl")
    with open(model_path, 'rb') as model_file:  # Открываем файл в бинарном режиме
        model = pickle.load(model_file)  # Загружаем модель
    return model

# Выгрузка моей загруженной в базу данных таблицы из Postgres в сервис/LMS
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# В этом датафрейме все, что нужно для прогноза. но удаляем фичи, которые относятся к характеристикам постов, а не юзеров
def load_features() -> pd.DataFrame:
    query = "SELECT * FROM aleksandr_tomaev_features_short"  
    features_on_service = batch_load_sql(query)
    return features_on_service

# В этом датафрейме постов есть только те фичи, что в модели (числовые)
def load_post_short() -> pd.DataFrame:
    query = "SELECT * FROM post_final_2gru_short"   
    post_short = batch_load_sql(query)
    return post_short

# В этом датафрейме полные данные по постам
def load_post_final() -> pd.DataFrame:
    query = "SELECT * FROM post_final_2gru"   
    post_final = batch_load_sql(query)
    return post_final

# Для оценки модели нужны фичи, загруженные с помощью def load_features(), а также числовые (векторные) представления текста
# При этом для того, чтобы сервис выдал 'post_id','text','topic' , эти три тоже должны быть в таблице

app = FastAPI()

model = load_models()
features_on_service = load_features()
post_short = load_post_short()
post_final = load_post_final()

def get_recommendation_for_user(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    # Выделяем отдельно взятого юзера
    user_features = features_on_service[features_on_service['user_id'] == id]

    # Добавляем проверку на пустой user_features
    if user_features.empty:
        return []
    
    # Делаем cross join , чтобы данному юзеру противопоставить все  посты
    user_features.loc[:, 'key'] = 1
    post_short.loc[:, 'key'] = 1
    doubled_data = pd.merge(user_features, post_short, on='key', how='right').drop('key', axis=1)
    # Заполняем пропуски, то есть дублируем характеристики юзера напротив каждого post_id
    for column in doubled_data.columns:
        if doubled_data[column].isnull().any():  # Проверяем, есть ли пропуски в столбце
            value_to_fill = doubled_data[column].dropna().iloc[0]  # Получаем первое ненулевое значение
            doubled_data[column].fillna(value_to_fill, inplace=True)  # Заполняем пропуски 
    
    # Наводим порядок с последовательностью колонок    
    desired_order = [
        'user_id', 'post_id', 'gender', 'final_component_1', 'final_component_2', 'final_component_3',
        'age_below_17', 'age_18_25', 'exp_group_1', 'exp_group_2', 'exp_group_3', 'exp_group_4', 
        'os_iOS', 'source_organic', 'topic_covid', 'topic_entertainment', 'topic_movie', 
        'topic_politics', 'topic_sport', 'topic_tech', 'country_mean_target', 'city_mean_target'
    ]       
    # Переупорядочиваем колонки в doubled_data
    doubled_data = doubled_data[desired_order]
    # С помощью модели делаем прогноз с учетом характеристик выбранного пользователя и признаков постов
    predictions = model.predict_proba(doubled_data)[:,1]
    # Создаем столбец
    doubled_data['prediction'] = predictions
    # Добавляем столбцы 'post_id','text','topic' (по PCA характеристикам, по сути)
    max_data = pd.merge(doubled_data, post_final, on='post_id', how='left')
    # Ранжируем
    # Получаем ID топ N постов
    top_post_ids = np.argsort(predictions)[::-1][:limit]

    recommendations = []
    for post_id_index in top_post_ids:
        post_id = max_data.iloc[post_id_index]['post_id']
        post_result = max_data.iloc[post_id_index]
        recommendations.append({
            'id': int(post_id),
            'text': post_result['text'],
            'topic': post_result['topic']
        })
    return recommendations 

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int = Query(..., description="User ID"),
                      time: datetime = Query(..., description="Timestamp"),
                      limit: int = Query(5, description="Number of recommendations")):
    try:
        recommendations = get_recommendation_for_user(id, time, limit)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



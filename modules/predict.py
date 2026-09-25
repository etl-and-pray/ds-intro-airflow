import glob
import json
import logging
import os
import dill
import pandas as pd
from datetime import datetime

# путь к проекту
path = os.environ.get('PROJECT_PATH', 'D:/airflow_hw')

logging.basicConfig(level=logging.INFO)


def predict() -> None:
    # 1. Находим и загружаем самую свежую обученную модель из data/models
    models_path = f'{path}/data/models'
    models = sorted(os.listdir(models_path))
    if not models:
        raise FileNotFoundError("Файлы моделей не найдены в папке data/models")

    latest_model = models[-1]
    model_file_path = f'{models_path}/{latest_model}'

    logging.info(f'Загрузка модели из {model_file_path}')
    with open(model_file_path, 'rb') as file:
        model = dill.load(file)

    # 2. Собираем все JSON из data/test и делаем предсказания
    preds = []
    test_files = glob.glob(f'{path}/data/test/*.json')

    for file_path in test_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame([data])
            y_pred = model.predict(df)

            preds.append({
                'car_id': data.get('id'),
                'pred': y_pred[0]
            })

    # 3. Сохраняем результат в CSV
    df_preds = pd.DataFrame(preds)

    predictions_path = f'{path}/data/predictions'
    os.makedirs(predictions_path, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d%H%M")
    result_file = f'{predictions_path}/preds_{now}.csv'

    df_preds.to_csv(result_file, index=False)
    logging.info(f'Предсказания успешно сохранены в {result_file}')


if __name__ == '__main__':
    predict()
# detector-graffiti

Детектор граффити распознаёт рисунки и надписи на стенах зданий, разделяет их на хорошие и плохие, а также выдаёт детекции пользователю. Это полезно для своевременного выявления порчи городской среды.

Основная модель - Grounding DINO. В основе модели лежит комбинированная архитектура, состоящая из Swin энкодера, который преобразует изображения в визуальные признаки, текстового энкодера на основе BERT, который преобразует текстовые запросы в эмбеддинги, и слоёв трансформера, которые сопоставляют текстовые и визуальные признаки. Модель вычисляет семантическое соответствие между регионами изображения и элементами текстового запроса, формируя представления, на которых основываются предсказания.

Ссылка на hugging face репозиторий модели: https://huggingface.co/IDEA-Research/grounding-dino-base.


# Prerequisits:
- poetry:
```
pip install poetry
```

# Установка и настройка
1. Клонирование репозитория:

Через SSH:
```
git clone git@github.com:DashaMed555/temp-detector-graffiti.git
```

Через HTTPS:
```
git clone https://github.com/DashaMed555/temp-detector-graffiti.git
```

2. Создание виртуального окружения и установка необходимых библиотек:

```
poetry install
```

3. Настройка DVC:

<CLIENT_ID> и <CLIENT_SECRET> можно получить, написав d.medvdeveda@g.nsu.ru.

```
poetry run dvc remote modify gdrive --local gdrive_client_id "<CLIENT_ID>"
```

```
poetry run dvc remote modify gdrive --local gdrive_client_secret "<CLIENT_SECRET>"
```

4. Загрузка датасетов:

```
poetry run dvc pull
```


# Работа с пакетом *convert_dataset*:
1. Обработка нескольких датасетов: слияние в один датасет и балансировка классов:
```
poetry run detector_graffiti/convert_dataset/process_datasets.py
```

2. Разделение датасета на тренировочную, валидационную и тестовую выборки:
```
poetry run detector_graffiti/convert_dataset/split_dataset.py
```

3. Конвертация YOLO аннотаций в JSON формат:
```
poetry run detector_graffiti/convert_dataset/convert_yolo_to_json.py
```

4. Валидация JSON аннотаций:
```
poetry run detector_graffiti/convert_dataset/validate.py
```

# Работа с моделью
- Запуск тренировки:
```
poetry run graffiti-detector train
```

- Запуск инференса:
```
poetry run graffiti-detector inference
```

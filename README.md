# detector-graffiti

Детектор граффити распознаёт рисунки и надписи на стенах зданий, разделяет их на хорошие и плохие, а также выдаёт детекции пользователю. Это полезно для своевременного выявления порчи городской среды.

Основная модель - Grounding DINO. В основе модели лежит комбинированная архитектура, состоящая из Swin энкодера, который преобразует изображения в визуальные признаки, текстового энкодера на основе BERT, который преобразует текстовые запросы в эмбеддинги, и слоёв трансформера, которые сопоставляют текстовые и визуальные признаки. Модель вычисляет семантическое соответствие между регионами изображения и элементами текстового запроса, формируя представления, на которых основываются предсказания.

Ссылка на hugging face репозиторий модели: https://huggingface.co/IDEA-Research/grounding-dino-base.

# Prerequisits

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

2. Переход в корень проекта:

```
cd detector-graffiti
```

3. Создание виртуального окружения и установка необходимых библиотек:

```
poetry install
```

4. Настройка DVC:

<CLIENT_ID> и <CLIENT_SECRET> можно получить, написав d.medvdeveda@g.nsu.ru.

```
poetry run dvc remote modify gdrive --local gdrive_client_id "<CLIENT_ID>"
```

```
poetry run dvc remote modify gdrive --local gdrive_client_secret "<CLIENT_SECRET>"
```

5. Загрузка датасетов:

При первом выполнении команды ниже нужно будет пройти авторизацию технического аккаунта google.
Для получения логина и пароля пишите d.medvdeveda@g.nsu.ru.

```
poetry run dvc pull
```

# Работа с пакетом _convert_dataset_

1. Обработка нескольких датасетов: слияние в один датасет и балансировка классов:

```
poetry run python detector_graffiti/convert_dataset/process_datasets.py
```

2. Разделение датасета на тренировочную, валидационную и тестовую выборки:

```
poetry run python detector_graffiti/convert_dataset/split_dataset.py
```

3. Конвертация YOLO аннотаций в JSON формат:

```
poetry run python detector_graffiti/convert_dataset/convert_yolo_to_json.py
```

4. Валидация JSON аннотаций:

```
poetry run python detector_graffiti/convert_dataset/validate.py
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

Для запуска mlflow для trining:
poetry mlflow ui --host 127.0.0.1 --port 8888

1. data_loading.yaml - Настройки данных
   train_json_path (str): Путь к JSON файлу с аннотациями тренировочных данных

val_json_path (str): Путь к JSON файлу с аннотациями валидационных данных

test_json_path (str): Путь к JSON файлу с аннотациями тестовых данных

train_image_path (str): Путь к директории с тренировочными изображениями

val_image_path (str): Путь к директории с валидационными изображениями

test_image_path (str): Путь к директории с тестовыми изображениями

2. model.yaml - Настройки модели
   model_id (str): Идентификатор или путь к модели (например, "runs/fine-tuning/2026-01-21 18-13-57/ft_model")

max_length (int): Максимальная длина текстового ввода (по умолчанию 64)

3. fine_tuning.yaml - Настройки тонкой настройки
   Основные параметры:
   output_dir (str): Директория для сохранения результатов обучения

prompt (str): Текстовый промпт для детекции (например, "legal graffiti . illegal graffiti .")

threshold (float): Порог детекции (0.0-1.0)

seed (int): Random seed для воспроизводимости

Параметры обучения:
per_device_train_batch_size (int): Batch size для обучения на каждом устройстве

per_device_eval_batch_size (int): Batch size для оценки на каждом устройстве

eval_accumulation_steps (int): Количество шагов аккумуляции для оценки

gradient_accumulation_steps (int): Количество шагов аккумуляции градиентов

num_train_epochs (int): Количество эпох обучения

learning_rate (float): Скорость обучения

weight_decay (float): Вес регуляризации

adam_beta2 (float): Beta2 параметр для оптимизатора Adam

optim (str): Оптимизатор (например, "adamw_torch")

Стратегии:
eval_strategy (str): Стратегия оценки ("epoch", "steps", "no")

save_strategy (str): Стратегия сохранения моделей ("best", "epoch", "no")

logging_strategy (str): Стратегия логирования ("epoch", "steps", "no")

lr_scheduler_type (str): Тип scheduler'а ("cosine", "linear", "constant")

Флаги:
eval_on_start (bool): Выполнять оценку перед началом обучения

remove_unused_columns (bool): Удалять неиспользуемые колонки из датасета

report_to (str): Куда отправлять отчеты ("none", "wandb", "mlflow")

load_best_model_at_end (bool): Загружать лучшую модель в конце обучения

dataloader_pin_memory (bool): Использовать pinned memory для DataLoader

disable_tqdm (bool): Отключить прогресс-бар tqdm

Настройки метрик:
metric_for_best_model (str): Метрика для выбора лучшей модели (например, "f1")

greater_is_better (bool): Большее значение метрики означает лучшую модель

4. freeze_layers.yaml - Заморозка слоев модели
   freeze_layers (bool): Включить заморозку слоев

encoder (bool): Разморозить энкодер

reference_points_head (bool): Разморозить reference points head

bbox_embed (bool): Разморозить bbox embedding слои

5. inference.yaml - Настройки инференса
   output_dir (str): Директория для сохранения результатов инференса

prompt (str): Текстовый промпт для детекции

threshold (float): Порог детекции для инференса

6. logging.yaml - Настройки логирования
   tracking_uri (str): URI для подключения к MLflow серверу

experiment_name (str): Имя эксперимента в MLflow

plots_dir (str): Поддиректория для сохранения графиков

7. onnx_converter.yaml - Конвертация в ONNX формат
   output_dir (str): Директория для сохранения ONNX модели

ft_model_id (str): Путь к fine-tuned модели

image_path (str): Путь к тестовому изображению для конвертации

image_size (tuple): Размер изображения (ширина, высота)

prompt (str): Текстовый промпт для конвертации

max_length (int): Максимальная длина текстового ввода

8. config.yaml - Основной конфигурационный файл
   defaults (list): Список подключаемых конфигураций:

data_loading: normal_dataset

model: model

freeze_layers: freeze_layers

fine_tuning: fine_tuning

logging: logging

inference: inference

onnx_converter: onnx_converter

Использование конфигурации
Проект использует Hydra для управления конфигурацией. Все параметры автоматически подключаются через основной config.yaml. Для переопределения параметров через командную строку можно использовать синтаксис:

poetry run

graffiti-detector train fine_tuning.learning_rate=2e-5 fine_tuning.num_train_epochs=3

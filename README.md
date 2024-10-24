# GeoService Project

## Описание
Проект представляет собой сервис для работы с геопространственными данными и анализа NDVI (Normalized Difference Vegetation Index) на основе данных PostgreSQL и GeoJSON.

## Установка

#### Установка Python и PostgreSQL:
1. Установите Python (рекомендуется Anaconda):
   [Инструкция по установке Anaconda](https://jino.ru/spravka/articles/anaconda.html#%D1%83%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-anaconda)

2. Установите PostgreSQL:
   ```bash
   sudo apt -y install postgresql postgresql-contrib
   sudo apt update

#### Создание виртуального окружения и установка пакетов:

 1. Создайте виртуальное окружение с Python 3.9:
    ```bash
    conda create --name <env> python=3.9
    conda activate <env>


 2. Установите необходимые пакеты:
    ```bash
    conda install geopandas
    conda install aiohttp 
    conda install tqdm 
    conda install psycopg2
    pip install asyncio
    pip install argparse




#### Настройка базы данных:

 1. Войдите в терминал PostgreSQL:
    ```bash
    sudo -i -u postgres
    psql


 2. Создайте пользователя и базу данных:
    ```bash
    CREATE USER <name> WITH PASSWORD '<password>';
    ALTER USER <name> WITH SUPERUSER;
    CREATE DATABASE geoservice;
    CREATE TABLE geoservice.agrofieldndvi
    (
        "AgroFieldID" bigint NOT NULL,
        "Date" date,
        "NDVI" double precision,
        "NDVIMin" double precision,
        "NDVIMax" double precision
    );
    ALTER TABLE agrofieldndvi OWNER TO <name>;



### Настройка конфигурации

В проекте присутствует файл конфигурации configuration.json, который необходимо настроить. Пример параметров:
    ![image](https://github.com/user-attachments/assets/4038d457-7490-4045-925f-ed30afb1fe0b)

        

Параметры конфигурации:

 * train_geojson_file: Путь к GeoJSON файлу для обучения модели;
 * classify_geojson_file: Путь к GeoJSON файлу для классификации данных;
 * new_geojson_file: Путь для сохранения классифицированных данных;
 * model: Путь к файлу модели классификатора;
 * year: Год, для которого выполняется обработка данных;
 * persecond: Количество запросов в секунду к прокси для получения рядов NDVI;
 * database: Параметры подключения к базе данных PostgreSQL.

Примечания

 При работе с базой данных в разных операционных системах запросы могут отличаться:
  * Для Windows: INSERT INTO geoservice.agrofieldndvi ...
  * Для Linux: INSERT INTO agrofieldndvi ...
### Запуск

Перед запуском необходимо настроить файл конфигурации.

      python crops_classifier.py --config .path/to/config --mode [train or classify] --cleaning [y or n]
      
Конфиг можно опционально не указывать, тогда применится конфиг по умолчанию. Также по умолчанию mode =  train 

### Выходные данные

После обучения модель сохраняется, а также возвращается точность классификации. Результаты классификации записываются в новый GeoJSON файл с указанием названий культур для каждого поля.

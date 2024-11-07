import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json
import asyncio
import logging
import time
import argparse
from collections import defaultdict
from joblib import dump, load
from os import path
from tqdm import tqdm
from asyncio import TimeoutError
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from ndvi_parser import NDVIParserService

TODAY = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
LOG_PATH = f'logs/crops_{TODAY}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ])

logger = logging.getLogger(__name__)


class CropsClassifierService:
    """
    Класс AgroClassifier отвечает за загрузку и обработку данных NDVI,
    обучение модели классификации и классификацию данных на основе обученной модели.
    """

    def __init__(self, config_file: str, cleaning: bool) -> None:
        """
        Инициализация AgroClassifier и загрузка конфигурации из указанного файла.

        Args:
            config_file: Путь к файлу конфигурации JSON.
            cleaning: Флаг очистки: если true - то бд перед работой скрипта очищается и в нее загружаются новые данные NDVI, иначе - очистка не призводится.
        """
        self.config = self.__load_config__(config_file)
        self.clean_flag = cleaning
        if not self.config.get('year'):
            raise ValueError('The year is not specified in the configuration')

        self.label_dict = {'озимые': 0, 'многолетние травы': 1,
                           'ранние яровые': 2, 'поздние яровые': 3, 'пар': 4}
        self.label_inv = {0: 'озимые', 1: 'многолетние травы',
                          2: 'ранние яровые', 3: 'поздние яровые', 4: 'пар'}

        logger.info(f"Initialization AgroClassifier")

    def __load_config__(self, config_file: str) -> dict:
        """
        Загрузка конфигурации из JSON файла.

        Args:
            config_file: Путь к файлу конфигурации JSON.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            json.JSONDecodeError: Если произошла ошибка декодирования JSON.

        Returns:
            Конфигурационный словарь.
        """
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
                logger.info(f"Configuration loaded from {config_file}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _create_split_(self, features: list, labels: list, random_state: int) -> \
            tuple[np.array, np.array, np.array, np.array]:
        """
        Разделяет данные на обучающую и тестовую выборки.

        Args:
            features: Список признаков.
            labels: Список меток.
            random_state: Случайное состояние для воспроизводимости.

        Returns:
            tuple[np.array, np.array, np.array, np.array]: Кортеж из X_train, X_test, y_train, y_test.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=random_state * 17)
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    async def __download_ndvi_async__(self, geojson_file: str, year: int) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Асинхронно загружает данные NDVI для указанного geojson файла и года.

        Args:
            geojson_file (str): Путь к geojson файлу.
            year (int): Год, для которого загружаются данные.

        Raises:
            TimeoutError: Если произошел таймаут при загрузке данных.
            RuntimeError: Если произошла ошибка выполнения в asyncio.

        Returns:
            tuple[gpd.GeoDataFrame, pd.DataFrame]: Кортеж из GeoDataFrame и DataFrame с данными NDVI.
        """
        afs = NDVIParserService(self.config)
        if self.clean_flag:
            afs.clear_data()
            try:
                gdf = await afs.compute_ndvi(geojson_file)
            except TimeoutError as e:
                logger.error(f"Error loading data: {e}")
                raise
            except RuntimeError:
                pass
        else:
            gdf = gpd.read_file(geojson_file)
        gdf['ID'] = gdf['ID'].astype(int)
        gdf['Date'] = year
        df_series = afs.read_data()
        return gdf, df_series

    async def _download_ndvi_(self, geojson_file: str, year: int) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Обертка для асинхронной загрузки данных NDVI.

        Args:
            geojson_file (str): Путь к geojson файлу.
            year (int): Год, для которого загружаются данные.

        Returns:
            tuple[gpd.GeoDataFrame, pd.DataFrame]: Кортеж из GeoDataFrame и DataFrame с данными NDVI.
        """
        logger.info(f"Loading NDVI for file {geojson_file} and year {year}")
        return await self.__download_ndvi_async__(geojson_file, year)

    def _process_ndvi_data_(self, df_series: pd.DataFrame, gdf: gpd.GeoDataFrame, year: int, flag: bool) -> \
            tuple[np.array, (np.array, list)]:
        """
        Обрабатывает данные NDVI, выполняет интерполяцию и формирует признаки и метки.

        Args:
            df_series: DataFrame с данными NDVI.
            gdf: GeoDataFrame с геометрическими данными.
            year: Год для обработки данных.

        Returns:
            tuple[np.array, (np.array or list)]: Кортеж из массива признаков и массива меток.
        """
        features, labels, obj_ids = [], [], []
        veg_start_date = datetime.date(year, 4, 1).toordinal()
        veg_end_date = datetime.date(year, 9, 1).toordinal()

        for key in tqdm(df_series['ID'].unique(), desc='Preprocessing data'):
            X = df_series.loc[df_series['ID'] ==
                              key, 'Date'].to_numpy().flatten()
            Y = df_series.loc[df_series['ID'] ==
                              key, 'NDVI'].to_numpy().flatten()

            X_ordinals = np.array([i.toordinal() for i in X[::-1]])
            Y = Y[::-1]

            if Y[0] < 0:
                X_ordinals[0] = veg_start_date
                Y[0] = 0
            for i in range(1, len(Y) - 1):
                if Y[i - 1] > 0 and Y[i] < 0 and Y[i + 1] > 0:
                    Y[i] = np.mean((Y[i - 1], Y[i + 1]))

            newX = np.arange(veg_end_date - veg_start_date)
            X_ordinals -= veg_start_date
            sorted_indices = np.argsort(X_ordinals)
            X_ordinals = X_ordinals[sorted_indices]
            Y = Y[sorted_indices]
            if X_ordinals[0] > 20:
                X_ordinals = np.insert(X_ordinals, 0, 0)
                Y = np.insert(Y, 0, 0)
            s = InterpolatedUnivariateSpline(X_ordinals, Y, k=1)
            newY = s(newX)

            features.append(newY)
            if flag:
                label = gdf.loc[(gdf['ID'] == key), 'CropClass'].values
                label_str = ''.join(label).translate(
                    {ord(i): None for i in '\'[]'})
                labels.append(self.label_dict[label_str])
            else:
                obj_ids.append(key)

        if flag:
            return np.array(features, dtype='float'), np.array(labels)
        else:
            return np.array(features, dtype='float'), obj_ids

    def model_training(self) -> tuple[RandomForestClassifier, float]:
        """
        Обучает модель классификации на основе данных NDVI.

        Raises:
            Exception: Если произошла ошибка во время обучения модели.

        Returns:
            tuple[RandomForestClassifier, float]: Кортеж из обученной модели RandomForestClassifier и её точности.
        """
        if not path.exists(self.config['train_geojson_file']):
            raise FileNotFoundError(
                f"Training file with path {self.config['train_geojson_file']} was not found")
        geojson_file = self.config['train_geojson_file']
        year = self.config['year']
        logger.info(
            f"Starting model training for file {geojson_file} and year {year}")

        try:
            gdf, df_series = asyncio.run(self._download_ndvi_(
                geojson_file, year))
            asyncio.run(asyncio.sleep(2))
            features, labels = self._process_ndvi_data_(
                df_series, gdf, year, flag=True)
            logger.info("Feature extraction and data labeling completed")

            counter = 10
            models = []
            accuracys = np.zeros((counter, 1), dtype='float32')

            for i in range(counter):
                X_train, X_test, y_train, y_test = self._create_split_(
                    features, labels, i)
                rfc = RandomForestClassifier(
                    max_depth=13, n_estimators=143, random_state=i)
                rfc_model = rfc.fit(X_train, y_train)
                preds = rfc.predict(X_test)
                accuracy = accuracy_score(y_test, preds)
                accuracys[i] = accuracy
                models.append(rfc_model)
                logger.info(
                    f"Model {i+1} trained with accuracy {accuracy:.4f}")

            best_index = np.argmax(accuracys)
            model = models[best_index]
            logger.info(
                f"The best model is trained with accuracy {accuracys[best_index][0]:.4f}")
            self.plot_NDVI(features, labels)
            model_path = f"models/model_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_{int(accuracys[best_index][0]*100)}.joblib"
            dump(model, model_path, compress=9)
            dump(model, 'models/last.joblib', compress=9)
            return model, accuracys[best_index][0]

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def predict_data(self) -> None:
        """
        Классифицирует данные NDVI на основе обученной модели и сохраняет результаты в новый geojson файл.

        Raises:
            Exception: Если произошла ошибка во время классификации данных.
        """
        if not path.exists(self.config['classify_geojson_file']):
            raise FileNotFoundError(
                f"Classify file with path {self.config['classify_geojson_file']} was not found")
        if not self.config['predicted_geojson_file']:
            raise FileNotFoundError('Path to predicted file was not specified')
        if not path.exists(self.config['model']):
            raise FileNotFoundError(
                f"Model file {self.config['model']} not found")

        geojson_file = self.config['classify_geojson_file']
        year = self.config['year']
        model_path = self.config['model']
        logger.info(f"Data classification for file {geojson_file}")

        try:
            model = load(model_path)
            logger.info(f"Model loaded from {model_path}")

            gdf, df_series = asyncio.run(
                self._download_ndvi_(geojson_file, year))
            asyncio.run(asyncio.sleep(2))
            new_features, obj_ids = self._process_ndvi_data_(
                df_series, gdf, year, flag=False)
            preds = model.predict(new_features)
            preds_decoded = [self.label_inv[pred] for pred in preds]
            gdf['CropClass'] = None

            for obj_id, pred in tqdm(zip(obj_ids, preds_decoded), desc='Writing classes', total=len(obj_ids)):
                gdf.loc[(gdf['ID'] == obj_id), 'CropClass'] = pred

            gdf.drop(columns=['Date'], inplace=True)
            gdf.to_file(self.config['predicted_geojson_file'], driver='GeoJSON')
            logger.info(f"Classification completed for file {geojson_file}")

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            raise

    def _plot_NDVI_(self, features: np.array, labels: np.array) -> None:
        """Создает график средних рядов NDVI для каждой культуры

        Args:
            features (np.array): Массив признаков
            labels (np.array): Массив меток
        """
        culture_ndvi = defaultdict(list)
        for feature, label in zip(features, np.array([self.label_inv[label] for label in labels])):
            culture_ndvi[label].append(feature)

        average_ndvi = {}
        for culture, ndvi_series in culture_ndvi.items():
            average_ndvi[culture] = np.mean(ndvi_series, axis=0)

        plt.figure(figsize=(15, 8))
        for culture, avg_ndvi in average_ndvi.items():
            plt.plot(avg_ndvi, label=culture)

        plt.title(
            f"Средние ряды NDVI для {path.basename(self.config['train_geojson_file'])}")
        plt.xlabel('Дни')
        plt.ylabel('NDVI')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(
            f"images/NDVI_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.png")


def main(config_path: str, mode: str, cleaning: bool):
    start_time = time.time()
    ac = CropsClassifierService(config_path, cleaning)

    if mode == 'train':
        try:
            ac.model_training()
        except Exception as e:
            logger.error(f"An error occurred: {e}")
    if mode == 'predict':
        try:
            ac.predict_data()
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    end_time = time.time()
    logger.info(f"Running time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgroClassifierService')
    parser.add_argument('--config', type=str, required=False, default='./configs/default.json',
                        help='Path to the configuration file (default.json is used by default)')
    parser.add_argument('--mode', type=str, required=False, choices=['predict', 'train'],
                        help='Running a method (predict - classification, train - training)')
    parser.add_argument('--cleaning', type=str, required=False, choices=['y', 'n'], default='y',
                        help='Clears the database before operation and loads new NDVI records, otherwise leaves old data (y - yes, n - no)')

    args = parser.parse_args()
    main(args.config, args.mode, True if args.cleaning == 'y' else False)

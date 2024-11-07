import asyncio
import aiohttp
import json
import pandas as pd
import geopandas as gpd
import logging
import datetime
from tqdm.asyncio import tqdm
from json.decoder import JSONDecodeError
from psycopg2 import pool, OperationalError
from shapely.geometry import shape
from shapely.validation import make_valid

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


class AgroFieldService:
    """
    Сервис для работы с данными NDVI.
    Выполняет загрузку данных, их обработку, сохранение в базу данных и чтение из базы данных.
    """

    INSERT_NDVI_QUERY = '''
    INSERT INTO agrofieldndvi ("AgroFieldID", "Date", "NDVI", "NDVIMin", "NDVIMax")
    VALUES (%s, %s, %s, %s, %s);
    '''

    DELETE_NDVI_QUERY = '''
    DELETE FROM agrofieldndvi;
    '''

    READ_NDVI_QUERY = '''
    SELECT "AgroFieldID", "NDVI", "Date" FROM agrofieldndvi WHERE EXTRACT(YEAR FROM "Date") = %s;
    '''

    NDVI_FIS_URL = 'https://functions.yandexcloud.net/d4e1flb0tjah7plqhdjf?LAYER=NDVISRC&STYLE=INDEX&CRS=EPSG%3A3857&TIME={1}-04-01%2F{1}-09-01&GEOMETRY={0}&RESOLUTION=50'

    def __init__(self, config: dict) -> None:
        """
        Инициализация сервиса.

        :param config: Словарь с конфигурацией.
        """
        self.db_config = config['database']
        self.persecond = config['persecond'] if config['persecond'] else 20
        self.year = config['year']
        self.connection_pool = pool.ThreadedConnectionPool(
            5, 100, **self.db_config)
        self.__check_db_connection__()

    def __check_db_connection__(self) -> None:
        """
        Проверка соединения с базой данных.

        :raises OperationalError: Если соединение с базой данных не удалось установить.
        """
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            record = cursor.fetchone()
            if record:
                logger.info(
                    f"Database connection: {record} established successfully")
        except OperationalError as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def _process_geometry_(self, geom) -> str:
        """
        Обрабатывает геометрию для формирования корректного WKT.

        :param geom: Геометрия объекта.
        :return (str): Обработанная геометрия в формате WKT.
        """
        valid_geom = make_valid(geom)
        simplified_geom = valid_geom.simplify(
            tolerance=100, preserve_topology=True)
        buffered_geom = simplified_geom.buffer(0.01)
        buffered_wkt = buffered_geom.wkt
        if len(buffered_wkt) > 1000:
            envelope_geom = valid_geom.envelope
            return envelope_geom.wkt
        else:
            return buffered_wkt

    async def _fetch_(self, session: aiohttp.ClientSession, task: dict) -> None:
        """
        Асинхронно выполняет HTTP-запрос для получения данных NDVI и сохраняет их в базу данных.

        :param session: Сессия aiohttp для выполнения запросов.
        :param task: Словарь с информацией о задаче, содержащий 'url' и 'ID'.
        """
        url = task['url']
        retries = 0
        max_retries = 5
        retry_delay = 0.5

        while retries < max_retries:
            try:
                async with session.get(url) as response:
                    if response.status == 429:
                        wait_time = retry_delay * (2 ** retries)
                        logger.warning(
                            f"Error 429: Waiting {wait_time} seconds before trying {url} again...")
                        await asyncio.sleep(wait_time)
                        retries += 1
                        continue
                    elif response.status != 200:
                        logger.error(
                            f"Failed to get {url}. Status code: {response.raise_for_status()}")
                        task['result'] = None
                        task['status'] = 'failed'
                        return

                    task['result'] = await response.text()
                    task['status'] = 'done'
                    sentinel_data = json.loads(task['result'])
                    agro_field_id = task['ID']
                    await self._save_ndvi_data_(agro_field_id, sentinel_data)
                    return

            except JSONDecodeError as e:
                logger.error(f"JSONDecoder error for {url}: {e}")
                task['result'] = None
                task['status'] = 'failed'
                return

            except aiohttp.ClientError as e:
                logger.error(f"{url} request error: {e}")
                task['result'] = None
                task['status'] = 'failed'
                return
        logger.error(
            f"The maximum number of retries for {url} has been reached. Pass...")
        task['result'] = None
        task['status'] = 'failed'
        return

    async def _fetch_all_(self, session: aiohttp.ClientSession, urls: list[dict]) -> list[dict]:
        """
        Асинхронно выполняет все задачи в очереди по загрузке данных NDVI.

        :param session: Сессия aiohttp для выполнения запросов.
        :param urls: Список словарей с 'ID' и 'url' для загрузки данных.
        :return: Обновленный список задач с результатами.
        """
        url_tasks = [{'ID': i['_ID'], 'url': i['url'],
                      'result': None, 'status': 'new'} for i in urls]
        n = 0

        total_tasks = len(url_tasks)

        with tqdm(total=total_tasks, desc="Fetching NDVI data", unit="task") as progress:
            while True:
                running_tasks = len(
                    [i for i in url_tasks if i['status'] == 'fetch'])
                is_tasks_to_wait = len(
                    [i for i in url_tasks if i['status'] != 'done' and i['status'] != 'failed'])

                if n < len(url_tasks) and running_tasks < self.persecond:
                    url_tasks[n]['status'] = 'fetch'
                    asyncio.create_task(self._fetch_(session, url_tasks[n]))
                    n += 1

                if running_tasks >= self.persecond:
                    await asyncio.sleep(1)

                completed_or_failed_tasks = len(
                    [i for i in url_tasks if i['status'] == 'done' or i['status'] == 'failed'])
                progress.n = completed_or_failed_tasks
                progress.refresh()

                if is_tasks_to_wait != 0:
                    await asyncio.sleep(0.1)
                else:
                    break

        return url_tasks

    async def compute_ndvi(self, geojson_file: str) -> gpd.GeoDataFrame:
        """
        Загружает и обрабатывает данные NDVI для указанного geojson файла и года.

        :param geojson_file: Путь к geojson файлу с геометрией полей.
        :return: GeoDataFrame с геометрией и добавленными данными NDVI.
        """
        logger.info(f"Read file {geojson_file}")
        gdf = gpd.read_file(geojson_file)
        urls = []

        for _, row in gdf.iterrows():
            geom = shape(row['geometry'])
            processed_wkt = self._process_geometry_(geom)
            url = self.NDVI_FIS_URL.format(processed_wkt, self.year)
            urls.append({'_ID': row['ID'], 'url': url})

        logger.info(
            f"NDVI parsing begins for {self.year} with fields {len(urls)}.")

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            await self._fetch_all_(session, urls)
        return gdf

    async def _save_ndvi_data_(self, agro_field_id: int, sentinel_data: dict) -> None:
        """
        Сохраняет полученные данные NDVI в базу данных.

        :param agro_field_id: Идентификатор агро поля.
        :param sentinel_data: Данные NDVI, полученные из ответа API.
        """
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            for date in sentinel_data['C0']:
                cursor.execute(self.INSERT_NDVI_QUERY,
                               (agro_field_id,
                                date['date'],
                                date['basicStats']['mean'],
                                date['basicStats']['min'],
                                date['basicStats']['max']))
            conn.commit()
        except Exception as e:
            logger.error(
                f"Error saving NDVI series for AgroFieldID {agro_field_id}: {e}")
            raise
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def clear_data(self) -> None:
        """
        Удаляет все записи NDVI из базы данных.
        """
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            cursor.execute(self.DELETE_NDVI_QUERY)
            conn.commit()
            logger.info(f"Database entries were successfully deleted")
        except Exception as e:
            logger.error(f"Database cleaning error: {e}")
            raise
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def read_data(self) -> pd.DataFrame:
        """
        Читает данные NDVI из базы данных для заданного года.

        :return: DataFrame с колонками 'ID', 'NDVI', 'Date'.
        """
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            cursor.execute(self.READ_NDVI_QUERY, (self.year,))
            df = pd.DataFrame(cursor.fetchall(), columns=[
                              'ID', 'NDVI', 'Date'])
            df = df.dropna()
            logger.info(f"Records from the database were successfully read")
        except Exception as e:
            logger.error(f"Reading error: {e}")
            raise
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)
        return df

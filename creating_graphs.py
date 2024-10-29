import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
import pandas as pd
from psycopg2 import pool
from scipy.interpolate import InterpolatedUnivariateSpline
from crops_classifier import AgroClassifierService


def plot_NDVI_id(id: int, config: str) -> None:
    """Создает график ряда NDVI для конкретного поля

    Args:
        id (int): Идентификатор поля
    """
    ac = AgroClassifierService(config, False)
    veg_start_date = datetime.date(ac.config['year'], 4, 1).toordinal()
    veg_end_date = datetime.date(ac.config['year'], 9, 1).toordinal()

    try:
        connection_pool = pool.ThreadedConnectionPool(
            5, 10, **ac.config['database'])
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT "AgroFieldID", "NDVI", "Date" FROM agrofieldndvi WHERE EXTRACT(YEAR FROM "Date") = %s''', (ac.config['year'],))
        df_series = pd.DataFrame(cursor.fetchall(), columns=[
            'ID', 'NDVI', 'Date'])
        df_series = df_series.dropna()
    except Exception as error:
        raise error
    finally:
        cursor.close()
        connection_pool.putconn(conn)

    for key in df_series['ID'].unique():
        if key == id:
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
            print('NDVI field range:\n', newY)
            break

    plt.figure(figsize=(15, 8))
    plt.plot(newY)
    plt.title(f'Ряд NDVI для id = {id}')
    plt.xlabel('Дни')
    plt.ylabel('NDVI')
    plt.grid(True)
    path = f"images/NDVI_for_{id}_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(path)
    print(f'The graph is saved in {path}')


def main(config: str, id: int):
    try:
        plot_NDVI_id(id, config)
    except Exception as error:
        print(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AgroClassifierService')
    parser.add_argument('--config', type=str, required=False, default='./configs/default.json',
                        help='Path to the configuration file (default.json is used by default)')
    parser.add_argument('--id', type=int, required=True, help='Field ID')
    args = parser.parse_args()
    main(args.config, args.id)

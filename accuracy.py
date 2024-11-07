import json
import argparse
import geopandas as gpd
import os


def accuracy_processing(config_file: str) -> float:
    with open(config_file, 'r') as file:
        config = json.load(file)

    print('Read geojson\'s files.')
    gdf_train = gpd.read_file(config['train_geojson_file'])
    gdf_pred = gpd.read_file(config['predicted_geojson_file'])
    gdf_train['ID'] = gdf_train['ID'].astype(int)
    gdf_pred['ID'] = gdf_pred['ID'].astype(int)
    accuracy = 0

    for row_train, row_pred in zip(gdf_train.iterrows(), gdf_pred.iterrows()):
        if row_pred[1]['ID'] == row_train[1]['ID']:
            if row_pred[1]['CropClass'] == row_train[1]['CropClass']:
                accuracy += 1

    accuracy /= len(gdf_train)
    return accuracy


def main(config: str):
    output_filename = f'accuracy_{os.path.splitext(os.path.basename(config))[0]}.txt'
    output_path = os.path.join('.', 'data', output_filename)
    accuracy = accuracy_processing(config)
    with open(output_path, 'w') as f:
        f.write(f'Accuracy: {accuracy}')
    print(f'The prediction was completed and the result is saved at {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AccuracyProcessing')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)

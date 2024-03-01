import yaml

from tqdm import tqdm

import numpy as np


if __name__ == "__main__":
    # train_file_path = '/home/liam/datasets/CARPK/train_tiles_target_nba_0.08_input_size_256_mode_test/tiles_dict.yaml'
    # val_file_path = '/home/liam/datasets/CARPK/val_tiles_target_nba_0.08_input_size_256_mode_test/tiles_dict.yaml'
    # test_file_path = '/home/liam/datasets/CARPK/test_tiles_target_nba_0.08_input_size_256_mode_test/tiles_dict.yaml'
    train_file_path = '/home/liam/datasets/CARPK/train_tiles_target_nba_0.02_input_size_256_mode_test/tiles_dict.yaml'
    val_file_path = '/home/liam/datasets/CARPK/val_tiles_target_nba_0.02_input_size_256_mode_test/tiles_dict.yaml'
    test_file_path = '/home/liam/datasets/CARPK/test_tiles_target_nba_0.02_input_size_256_mode_test/tiles_dict.yaml'
    train_num_tiles = 0
    train_max_num_tiles = 0
    val_num_tiles = 0
    val_max_num_tiles = 0
    test_num_tiles = 0
    test_max_num_tiles = 0
    with open(train_file_path) as train_file:
        train_tile_dict = yaml.full_load(train_file)
        for num, image in tqdm(train_tile_dict.items()):
            train_num_tiles += len(image['tiles'])
            if len(image['tiles']) > train_max_num_tiles:
                train_max_num_tiles = len(image['tiles'])
    with open(val_file_path) as val_file:
        val_tile_dict = yaml.full_load(val_file)
        for num, image in tqdm(val_tile_dict.items()):
            val_num_tiles += len(image['tiles'])
            if len(image['tiles']) > val_max_num_tiles:
                val_max_num_tiles = len(image['tiles'])
    with open(test_file_path) as test_file:
        test_tile_dict = yaml.full_load(test_file)
        for num, image in tqdm(test_tile_dict.items()):
            test_num_tiles += len(image['tiles'])
            if len(image['tiles']) > test_max_num_tiles:
                test_max_num_tiles = len(image['tiles'])


    print(f'Test max num tiles: {test_max_num_tiles}')
    print(f'Test average num tiles: {test_num_tiles/len(test_tile_dict)}')

    print(f'Val max num tiles: {val_max_num_tiles}')
    print(f'Val average num tiles: {val_num_tiles/len(val_tile_dict)}')

    print(f'Train max num tiles: {train_max_num_tiles}')
    print(f'Train average num tiles: {train_num_tiles/len(train_tile_dict)}')



    print(f"Number of tiles: {train_num_tiles+val_num_tiles+test_num_tiles}")
    print(f'overall average num tiles: {(train_num_tiles+val_num_tiles+test_num_tiles)/(len(train_tile_dict)+len(val_tile_dict)+len(test_tile_dict))}')
    # print(f"Average number of tiles: {num_tiles/len(train_tile_dict)}")
    # print(f"Max number of tiles: {max_num_tiles}")
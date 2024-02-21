import torch
import cv2
import random
import yaml
import os

import numpy as np
import torch.nn as nn
import torchvision.ops as ops

from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class Tile:
    x_min: int
    x_max: int
    y_min: int
    y_max: int


class Tiler:

    def __init__(self, config):

        self.tile_dirs_exist = False
        self.read_config(config)
        self.model_input_size = self.config['network_input_size']
        self.target_nba = self.config['target_nba']
        self.dataset_yaml = self.config['dataset_yaml']
        self.mode = self.config['mode']
        self.keep_empty_tiles = self.config['keep_empty_tiles']
        self.filer_iou = self.config['filter_iou_threshold']
        self.filter_intersection_ratio = self.config['filter_intersection_ratio_threshold']
        self.read_dataset_yaml()
        self.dataset_dir = self.dataset_config['path']
        self.create_new_dirs()
        # if not self.tile_dirs_exist:
        #     self.load_dataset()
        #     self.get_tiled_splits()

    def create_new_dirs(self):
        self.new_train_dir = self.dataset_dir + '/train_tiles_target_nba_' + str(self.target_nba) +\
            '_input_size_' + str(self.model_input_size) + '_mode_' + self.mode
        self.new_val_dir = self.dataset_dir + '/val_tiles_target_nba_' + str(self.target_nba) +\
            '_input_size_' + str(self.model_input_size) + '_mode_' + self.mode
        self.new_test_dir = self.dataset_dir + '/test_tiles_target_nba_' + str(self.target_nba) +\
            '_input_size_' + str(self.model_input_size) + '_mode_' + self.mode
        
        # Check if dirs exists
        if os.path.exists(self.new_train_dir):
            self.tile_dirs_exist = True
        else:    
            os.makedirs(self.new_train_dir + '/images')
            os.makedirs(self.new_train_dir + '/labels')
        if not os.path.exists(self.new_val_dir):
            os.makedirs(self.new_val_dir + '/images')
            os.makedirs(self.new_val_dir + '/labels')
        if os.path.exists(self.new_test_dir):
            self.tile_dirs_exist = True
        else:
            os.makedirs(self.new_test_dir + '/images')
            os.makedirs(self.new_test_dir + '/labels')

    def get_split_dataset(self):
        if not self.tile_dirs_exist:
            self.load_dataset()
            self.get_tiled_splits()
        else:
            print("Tile directories already exist. If you want to re-tile the dataset, delete the existing tile directories.")

    def load_images(self, image_path):
        path = self.dataset_dir + '/' + image_path
        images = []
        for filename in tqdm(os.listdir(path)):
            img = cv2.imread(os.path.join(path, filename))
            images.append({'filename': filename, 'image': img})
        return images
    
    def load_labels(self, path):
        labels_path = self.dataset_dir + '/' + path.split('/')[0] + '/labels'
        labels = []
        for filename in os.listdir(labels_path):
            with open(os.path.join(labels_path, filename), 'r') as f:
                instances = f.read().splitlines()
                instances = [instance.split(' ') for instance in instances]
                instances_float = []
                for instance in instances:
                    instances_float.append([float(coord) for coord in instance])

                labels.append({'filename': filename, 'instances': np.array(instances_float)})
        return labels

    def load_dataset(self):
        self.og_train_images = self.load_images(self.dataset_config['train'])
        self.og_train_images = sorted(self.og_train_images, key=lambda k: k['filename'])
        self.og_train_labels = self.load_labels(self.dataset_config['train'])
        self.og_train_labels = sorted(self.og_train_labels, key=lambda k: k['filename'])

        self.og_val_images = self.load_images(self.dataset_config['val'])
        self.og_val_images = sorted(self.og_val_images, key=lambda k: k['filename'])
        self.og_val_labels = self.load_labels(self.dataset_config['val'])
        self.og_val_labels = sorted(self.og_val_labels, key=lambda k: k['filename'])

        self.og_test_images = self.load_images(self.dataset_config['test'])
        self.og_test_images = sorted(self.og_test_images, key=lambda k: k['filename'])
        self.og_test_labels = self.load_labels(self.dataset_config['test'])
        self.og_test_labels = sorted(self.og_test_labels, key=lambda k: k['filename'])

    def read_dataset_yaml(self):
        with open(self.dataset_yaml, 'r') as stream:
            try:
                self.dataset_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def read_config(self, config):
        # Read the configuration file which is a yaml file
        with open(config, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def get_avg_bbox_area(self, bboxes):
        areas = []
        for bbox in bboxes:
            areas.append(bbox[3]* bbox[4])

        return sum(areas) / len(areas)
    
    def get_tile_wh(self, full_image_shape, avg_nba):
        scale_factor = self.target_nba / avg_nba
        img_area = full_image_shape[0] * full_image_shape[1]
        tile_wh = np.ceil(np.sqrt(img_area / scale_factor)).astype(int)
        return max(tile_wh, self.model_input_size)
    
    def get_list_of_tiles(self, og_image, avg_bbox_area, debug_plot=False):

        if self.mode == 'test':
            optimal_tile_overlap = 1.5*np.sqrt(avg_bbox_area*og_image.shape[0]*og_image.shape[1])
        else:
            optimal_tile_overlap = 0
        tile_wh = self.get_tile_wh(og_image.shape, avg_bbox_area)
        # w, h, top_left, _, _ = self.get_shape_of_image_without_padding(og_image)
        w, h = og_image.shape[1], og_image.shape[0]

        num_tiles_w = np.ceil(w / tile_wh).astype(int)
        if num_tiles_w == 1:
            centers_x = np.array([w / 2])
        else:
            optimal_overlap_w = False
            while not optimal_overlap_w:
                centers_x = np.linspace(tile_wh / 2, w - tile_wh / 2, num=num_tiles_w)
                x_spacing = centers_x[1] - centers_x[0]
                overlap = tile_wh - x_spacing
                if overlap < optimal_tile_overlap:
                    num_tiles_w += 1
                else:
                    optimal_overlap_w = True

        num_tiles_h = np.ceil(h / tile_wh).astype(int)
        if num_tiles_h == 1:
            centers_y = np.array([h / 2])
        else:
            optimal_overlap_h = False
            while not optimal_overlap_h:
                centers_y = np.linspace(tile_wh / 2, h - tile_wh / 2, num=num_tiles_h)
                y_spacing = centers_y[1] - centers_y[0]
                overlap = tile_wh - y_spacing
                if overlap < optimal_tile_overlap:
                    num_tiles_h += 1
                else:
                    optimal_overlap_h = True

        tiles = []

        for x in centers_x:
            for y in centers_y:
                x_min = int(x - tile_wh / 2)
                x_max = int(x_min + tile_wh)
                y_min = int(y - tile_wh / 2)
                y_max = int(y_min + tile_wh)
                tile = Tile(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                tiles.append(tile)

        if debug_plot:
            fake_image = og_image.copy()
            for tile in tiles:
                # create a tuple color for the rectangle where the values for the 
                # 3 channels are chosen randomly
                color = [random.randint(0, 255), random.randint(0, 255),
                            random.randint(0, 255)]
                cv2.rectangle(fake_image, (tile.x_min, tile.y_min),
                                (tile.x_max, tile.y_max), color, 2)
                cv2.imshow('fake_image', fake_image)
                cv2.waitKey(0)
            cv2.destroyAllWindows()

        return tiles
    
    def extract_tile_from_image(self, image, tile):

        image_tile = image[tile.y_min:tile.y_max, tile.x_min:tile.x_max, :]
        resized_tile = cv2.resize(image_tile, (self.model_input_size, self.model_input_size))

        return resized_tile
    
    # def get_tile_label(self, tile, image_labels):
    #     # Get the bboxes that are within the tile
    #     bboxes = image_labels['instances']
    #     tile_bboxes = []
    #     for bbox in bboxes:
    #         bbox = bbox[1:]
    #         x1 = bbox[0] - bbox[2] / 2
    #         x2 = bbox[0] + bbox[2] / 2
    #         y1 = bbox[1] - bbox[3] / 2
    #         y2 = bbox[1] + bbox[3] / 2
    #         if bbox[0] > tile.x_min and bbox[2] < tile.x_max and bbox[1] > tile.y_min and bbox[3] < tile.y_max:
    #             tile_bboxes.append(bbox)
    #     return tile_bboxes

    def check_bbox(self, x_min, y_min, w, h):
        """
        Checks whether the bbox dimensions are valid in the sense that the width and height have to be larger than 0.
        :param x_min: minimal x value of bounding box
        :param y_min: minimal y value of bounding box
        :param w: width of bounding box
        :param h: height of bounding box
        :return: bbox_is_valid: is False when bbox dimensions are invalid
        """
        x_max = x_min + w
        y_max = y_min + h
        bbox_is_valid = True

        if x_max <= x_min:
            bbox_is_valid = False
            return bbox_is_valid
        if y_max <= y_min:
            bbox_is_valid = False
            return bbox_is_valid

        return bbox_is_valid


    def check_that_down_sampled_bbox_is_valid(self, x_min, y_min, w, h):
        if x_min + w > self.model_input_size:
            print("hello")
            w -= 1
        if y_min + h > self.model_input_size:
            print("hello 2")
            h -= 1

        return x_min, y_min, w, h


    def create_new_instance_inside_tile(self, old_instance, tile, img_shape):
        """
        Determines which part of the old instance bbox is inside the tile and creates a new instance with a bbox that fits
        into the tile and has coordinates relative to the tile edges.
        :param old_instance: object instance dict in original image frame
        :param tile: tile object that overlaps with old_instance bbox
        :return: a new instance dict
        """

        bbox = old_instance[1:]
        x1 = (bbox[0] - bbox[2] / 2) * img_shape[1]
        y1 = (bbox[1] - bbox[3] / 2) * img_shape[0]
        w = bbox[2] * img_shape[1]
        h = bbox[3] * img_shape[0]

        width_scale_down = (tile.x_max - tile.x_min) / self.model_input_size
        height_scale_down = (tile.y_max - tile.y_min) / self.model_input_size

        if width_scale_down < 0 or height_scale_down < 0:
            print("Invalid tile selection! Tiles must be larger than network input size.")

        if x1 > tile.x_min:
            x_new = x1 - tile.x_min
            if x_new + w > tile.x_max - tile.x_min:
                new_width = tile.x_max - x_new - tile.x_min - 1
            else:
                new_width = w

        else:
            x_new = 0
            new_width = x1 + w - tile.x_min

        if y1 > tile.y_min:
            y_new = y1 - tile.y_min
            if y_new + h > tile.y_max - tile.y_min:
                new_height = tile.y_max - y_new - tile.y_min - 1
            else:
                new_height = h

        else:
            y_new = 0
            new_height = y1 + h - tile.y_min

        x_new_scaled = int(np.round(x_new / width_scale_down))
        y_new_scaled = int(np.round(y_new / height_scale_down))
        w_new_scaled = int(np.round(new_width / width_scale_down))
        h_new_scaled = int(np.round(new_height / height_scale_down))

        x_new_scaled, y_new_scaled, \
        w_new_scaled, h_new_scaled = self.check_that_down_sampled_bbox_is_valid(x_new_scaled,
                                                                                y_new_scaled,
                                                                                w_new_scaled,
                                                                                h_new_scaled)

        if self.check_bbox(x_new_scaled, y_new_scaled, w_new_scaled, h_new_scaled):
            new_instance = {
                            "label": 1,
                            "specific_label": old_instance[0],
                            "x": x_new_scaled,
                            "y": y_new_scaled,
                            "w": w_new_scaled,
                            "h": h_new_scaled
                        }
            # Convert back to relative coordinates
            x = (x_new_scaled + w_new_scaled / 2) / self.model_input_size
            y = (y_new_scaled + h_new_scaled / 2) / self.model_input_size
            w = w_new_scaled / self.model_input_size
            h = h_new_scaled / self.model_input_size
            new_instance = np.array([old_instance[0], x, y, w, h])
            return new_instance

        else:
            return None


    def check_whether_bbox_is_in_current_tile(self, instance, tile, img_shape):
        """
        Checks whether the current bbox overlaps with the tile and if so returns a new instance dict that is adapted to
        this tile in terms of size and coordinate frame.
        :param instance: dict containing info about class and bbox dimensions
        :param tile: instance of Tile class
        :return: None or new instance dict
        """
        bbox = instance[1:]
        x1 = (bbox[0] - bbox[2] / 2) * img_shape[1]
        x2 = (bbox[0] + bbox[2] / 2) * img_shape[1]
        y1 = (bbox[1] - bbox[3] / 2) * img_shape[0]
        y2 = (bbox[1] + bbox[3] / 2) * img_shape[0]
        if x1 >= tile.x_max or x2 <= tile.x_min or y1 >= tile.y_max or y2 <= tile.y_min:
            return None
        else:
            instance_inside_tile = self.create_new_instance_inside_tile(instance, tile, img_shape)
            return instance_inside_tile

    def write_tiled_images_to_disk(self, tiled_images, tiled_labels, new_dir):
        idx = 0
        og_filenames = []
        for img, labels in zip(tiled_images, tiled_labels):
            if img['og_filename'] not in og_filenames:
                idx = 0
                og_filenames.append(img['og_filename'])
            yolo_annotations = ""

            for instance in labels['boundingBoxes']:
                cls = instance[0]
                x = instance[1]
                y = instance[2]
                w = instance[3]
                h = instance[4]
                yolo_annotations += (f'{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

            annotation_name = labels['image_name'].split('.')[0] + '_' + str(idx) + '.txt'
            annot_file_path = new_dir + '/labels/' + annotation_name
            with open(annot_file_path, 'w') as f:
                f.writelines(yolo_annotations)
            image_name = img['og_filename'].split('.')[0] + '_' + str(idx).zfill(4) + '.png'
            cv2.imwrite(new_dir + '/images/' + image_name, img['tiled_image'])
            idx += 1

    def get_tiled_splits(self):

        # tiled_train_images, tiled_train_labels, \
        #     train_tiles_dict = self.split_images_into_tiles(self.og_train_images, self.og_train_labels)
        # self.write_tiled_images_to_disk(tiled_train_images, tiled_train_labels, self.new_train_dir)

        # tiled_val_images, tiled_val_labels, \
        #     val_tiles_dict = self.split_images_into_tiles(self.og_val_images, self.og_val_labels)
        # self.write_tiled_images_to_disk(tiled_val_images, tiled_val_labels, self.new_val_dir)
        
        tiled_test_images, tiled_test_labels, \
            test_tiles_dict = self.split_images_into_tiles(self.og_test_images, self.og_test_labels)
        self.write_tiled_images_to_disk(tiled_test_images, tiled_test_labels, self.new_test_dir)

        # write the tiles dict to disk
        with open(self.new_test_dir + '/tiles_dict.yaml', 'w') as f:
            yaml.dump(test_tiles_dict, f)

        
        
    def plot_tiled_images(self, image, tiled_images, tiled_labels, tiles):


        fake_image = image.copy()
        w, h = image.shape[1], image.shape[0]
        for img, lbls, in zip( tiled_images, tiled_labels):
            color = [random.randint(0, 255), random.randint(0, 255),
                     random.randint(0, 255)]

            tile_img = img.copy()
            for bbox in lbls['boundingBoxes']:
                x1 = int((bbox[1] - bbox[3] / 2) * self.model_input_size)
                y1 = int((bbox[2] - bbox[4] / 2) * self.model_input_size)
                x2 = int((bbox[1] + bbox[3] / 2) * self.model_input_size)
                y2 = int((bbox[2] + bbox[4] / 2) * self.model_input_size)
                cv2.rectangle(tile_img, (x1, y1), (x2, y2), color, 2)

            cv2.imshow('tile_image', tile_img)

                
            cv2.imshow('og_image', fake_image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_tiles_dict_to_yaml_compatible(self, tiles_dict):

        tiles_dict_converted = {}
        for key, value in tiles_dict.items():
            tiles_dict_converted[key] = {'image_name': value['image_name'], 'tiles': []}
            for tile in value['tiles']:
                tiles_dict_converted[key]['tiles'].append({'x_min': tile.x_min,
                                                 'x_max': tile.x_max,
                                                 'y_min': tile.y_min,
                                                 'y_max': tile.y_max})
        return tiles_dict_converted


    def split_images_into_tiles(self, images, labels, debug_plot=False):

        tiled_images = []
        tiled_labels = []
        tiles_dict = {}
        num_empty_tiles = 0
        num_empty_tiles_that_are_used = 0
        idx = 0
        for image_dict, image_labels in zip(images, labels):
            image = image_dict['image']
            img_shape = image.shape
            avg_bbox_area = self.get_avg_bbox_area(image_labels['instances'])
            tiles_dict[idx] = {'image_name': image_labels['filename'],
                                'tiles': self.get_list_of_tiles(image, avg_bbox_area)}
            
            current_tiled_images = []
            current_tiled_labels = []

            # for tile in tiles_dict[idx]:
            #     tiled_images.append(self.extract_tile_from_image(image, tile))
            
            for tile in tiles_dict[idx]['tiles']:
                # tile = tiles_dict[index]
                bboxes_list = []
                number_of_bboxes_in_this_tile = 0
                for instance in image_labels['instances']:
                    new_instance = self.check_whether_bbox_is_in_current_tile(instance, tile, img_shape)
                    if new_instance is not None:
                        bboxes_list.append(new_instance)
                        number_of_bboxes_in_this_tile += 1
                if number_of_bboxes_in_this_tile == 0:
                    num_empty_tiles += 1

                empty_tile_frequency = 1
                if self.mode == 'train':
                    empty_tile_frequency = 16

                if number_of_bboxes_in_this_tile > 0 or (self.keep_empty_tiles and
                                                        num_empty_tiles % empty_tile_frequency == 0):
                    if number_of_bboxes_in_this_tile == 0:
                        num_empty_tiles_that_are_used += 1
                    tiled_image_dict = {
                        'image_name': image_labels['filename'],
                        'tile': {'x_min': tile.x_min,
                                'x_max': tile.x_max,
                                'y_min': tile.y_min,
                                'y_max': tile.y_max
                                },
                        'boundingBoxes': bboxes_list
                    }
                    resized_image = self.extract_tile_from_image(image, tile)
                    
                    current_tiled_images.append({'og_filename': image_dict['filename'], 'tiled_image': resized_image})
                    current_tiled_labels.append(tiled_image_dict)

            if debug_plot:
                self.plot_tiled_images(image, current_tiled_images, current_tiled_labels, tiles_dict[idx]['tiles'])

            tiled_images.extend(current_tiled_images)
            tiled_labels.extend(current_tiled_labels)

            idx += 1
        tiles_dict = self.convert_tiles_dict_to_yaml_compatible(tiles_dict)
        return tiled_images, tiled_labels, tiles_dict

    def get_intersection(self, bboxes1, bboxes2, equal_boxes=True):
        intersection = torch.zeros(bboxes1.shape[0], bboxes2.shape[0]).to('cpu')

        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                if equal_boxes and i == j:
                    intersection[i, j] = 0
                    continue
                x1 = max(bbox1[0], bbox2[0])
                y1 = max(bbox1[1], bbox2[1])
                x2 = min(bbox1[2], bbox2[2])
                y2 = min(bbox1[3], bbox2[3])
                intersection[i, j] = max(0, x2 - x1) * max(0, y2 - y1)
     
        return intersection
    
    def get_intersection_ratio(self, bboxes, intersection):
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        ratio = intersection / area
        return ratio
    
    def get_all_bboxes_and_confs(self, stitched_pres):
        all_bboxes = torch.empty(0, 4).to('cpu')
        all_confs = []
        for tile_idx in stitched_pres:
            tile = stitched_pres[tile_idx]
            bboxes = tile['predictions']
            for instance in bboxes:
                all_bboxes = torch.cat((all_bboxes, torch.tensor([instance['bbox']]).to('cpu')))
                conf = instance['conf'].to('cpu')
                all_confs.append(conf)

        all_confs = torch.tensor(np.asarray(all_confs)).to('cpu')

        return all_bboxes, all_confs
    
    def get_intersection_ratio_mask(self, bboxes):
        intersection = self.get_intersection(bboxes, bboxes)
        intersection_ratio = self.get_intersection_ratio(bboxes, intersection)
        max_ratio = torch.amax(intersection_ratio, dim=0)
        mask = torch.where(max_ratio < self.filter_intersection_ratio, True, False)

        return mask

    def non_max_suppression(self, stitched_preds):
        """ For all bboxes check the iou with all other boxes and if the iou is higher than the threshold, remove the
        bbox that is smaller. """

        all_bboxes, all_confs = self.get_all_bboxes_and_confs(stitched_preds)
        
        iou_mask = ops.nms(all_bboxes, all_confs, iou_threshold=self.filer_iou)
        remaining_bboxes = all_bboxes[iou_mask]
        remaining_confs = all_confs[iou_mask]

        ratio_mask = self.get_intersection_ratio_mask(remaining_bboxes)
        remaining_bboxes = remaining_bboxes[ratio_mask]
        remaining_confs = remaining_confs[ratio_mask]
            
        return remaining_bboxes, remaining_confs
    
    def merge_bboxes(self, stitched_preds):

        all_bboxes, all_confs = self.get_all_bboxes_and_confs(stitched_preds)

        iou = ops.box_iou(all_bboxes, all_bboxes)
        intersection = self.get_intersection(all_bboxes, all_bboxes)
        intersection_ratio = self.get_intersection_ratio(all_bboxes, intersection)

        # Group bboxes that have a high iou or high intersection ratio
        all_correspondence_ids = []
        # for i in range(iou.shape[0]):
        #     if i not in [cors for cors in all_correspondence_ids]:
        #         # correspondence_ids = [i]
        #         for j in range(iou.shape[1]):
        #             if iou[i, j] > self.filer_iou or intersection_ratio[i, j] > self.filter_intersection_ratio:
        #                 if j not in [cors for cors in all_correspondence_ids]:
        #                     correspondence_ids = [i, j]
        #                 else:
        #                     # Find the group that j is in and add i to that group
        #                     for group in all_correspondence_ids:
        #                         if j in group:
        #                             group.append(i)
        #                             break

        #         all_correspondence_ids.append(correspondence_ids)

        for i in range(iou.shape[0]):
                correspondence_ids = [i]
                for j in range(iou.shape[1]):
                    if j == i:
                        continue
                    if iou[i, j] > self.filer_iou or intersection_ratio[i, j] > self.filter_intersection_ratio:
                        correspondence_ids.append(j)
                all_correspondence_ids.append(correspondence_ids)

        merged_sets = self.merge_sets(all_correspondence_ids)

        # Convert sets back to lists and sort them
        unique_correspondences = [sorted(list(s)) for s in merged_sets]

        # Merge the bboxes that are in the same group
        merged_bboxes = []
        merged_confs = []
        for correspondence_ids in unique_correspondences:
            bboxes = all_bboxes[correspondence_ids]
            confs = all_confs[correspondence_ids]
            x1 = torch.amin(bboxes[:, 0])
            y1 = torch.amin(bboxes[:, 1])
            x2 = torch.amax(bboxes[:, 2])
            y2 = torch.amax(bboxes[:, 3])
            merged_bboxes.append([x1, y1, x2, y2])
            merged_confs.append(torch.amax(confs))


        return merged_bboxes, merged_confs

    def merge_sets(self, list_of_sets):
        merged = []
        for s in list_of_sets:
            added = False
            for m in merged:
                if any(x in m for x in s):
                    m.update(s)
                    added = True
                    break
            if not added:
                merged.append(set(s))
        return merged
    
    def stitch_tiled_predictions(self, tiled_predictions, tiles_dict, image_name):
        for idx in tiles_dict:
            if tiles_dict[idx]['image_name'].split('.')[0] == image_name.split('.')[0]:
                tile_info = tiles_dict[idx]['tiles']
                tile_wh = tile_info[0]['x_max'] - tile_info[0]['x_min']
                break
        
        stitched_predictions = {}
        for tile_idx, (tile, pred) in enumerate(zip(tile_info, tiled_predictions)):
            stitched_predictions[tile_idx] = {'tile': {'x_min': tile['x_min'],
                                                     'x_max': tile['x_max'],
                                                     'y_min': tile['y_min'],
                                                     'y_max': tile['y_max']},
                                            'predictions': []}
            for box, conf in zip(pred.boxes.xyxy, pred.boxes.conf):
                x1, y1, x2, y2 = box
                x1 = x1 * tile_wh/self.model_input_size + tile['x_min']
                x2 = x2 * tile_wh/self.model_input_size + tile['x_min']
                y1 = y1 * tile_wh/self.model_input_size + tile['y_min']
                y2 = y2 * tile_wh/self.model_input_size + tile['y_min']
                stitched_predictions[tile_idx]['predictions'].append({'bbox': [x1, y1, x2, y2],
                                                                      'conf': conf})
                
        # filtered_bboxes, filtered_confs = self.non_max_suppression(stitched_predictions)
        filtered_bboxes, filtered_confs = self.merge_bboxes(stitched_predictions)

        return stitched_predictions, filtered_bboxes, filtered_confs


if __name__ == "__main__":
    tiler = Tiler('/home/liam/ultralytics/tiling_config.yaml')
    tiler.get_split_dataset()
    a=1
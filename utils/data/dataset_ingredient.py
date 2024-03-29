import os.path as osp
import os
from typing import NamedTuple, Optional

import logging
from glob import glob
from collections import defaultdict
from sacred import Ingredient
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torchvision import transforms
import numpy as np
import pickle
import math

from .image_dataset import ImageDataset
from .utils import RandomReplacedIdentitySampler, TripletSampler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def config():
    name = 'sop'
    data_path = 'data/Stanford_Online_Products'
    train_folder = '/content/drive/MyDrive/small/train'
    test_folder = '/content/drive/MyDrive/tokyo_night/test'

    batch_size = 32
    sample_per_id = 2
    assert (batch_size % sample_per_id == 0)
    test_batch_size = 100
    sampler = 'triplet' #"random"

    num_workers = 8 
    pin_memory = True

    crop_size = 224
    recalls = [1, 5, 10, 20]

    num_identities = batch_size // sample_per_id 
    num_iterations = 59551 // batch_size

    train_cache_nn_inds  = '/content/AML_Rerank_MobileNet/rrt_sop_caches/rrt_r50_sop_nn_inds_train.pkl'
    test_cache_nn_inds   = None


@data_ingredient.named_config
def sop_global():
    name = 'sop_global'
    batch_size = 800
    test_batch_size = 800
    sampler = 'random_id'


@data_ingredient.named_config
def sop_rerank():
    name = 'sop_rerank'
    batch_size = 300
    test_batch_size = 600
    sampler = 'triplet'
    # Recall 1, 5, 10, 20
    recalls = [1, 10, 100]

    #train_cache_nn_inds  = 'rrt_sop_caches/rrt_r50_sop_nn_inds_train.pkl'
    #test_cache_nn_inds   = 'rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl'


@data_ingredient.capture
def get_transforms(crop_size):
    train_transform, test_transform = [], []
    train_transform.extend([
        transforms.RandomResizedCrop(size=crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])
    test_transform.append(transforms.Resize((256, 256)))
    test_transform.append(transforms.CenterCrop(size=224))
    test_transform.append(transforms.ToTensor())
    return transforms.Compose(train_transform), transforms.Compose(test_transform)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


@data_ingredient.capture
def get_sets(name, data_path, train_folder, test_folder, num_workers, M=10, alpha=30, N=5, L=2,
                 current_group=0, min_images_per_class=10, queries_folder_name = "queries",
                 positive_dist_threshold=25):

    # Open training folder
    logging.debug(f"Searching training images in {train_folder}")
        
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Folder {train_folder} does not exist")
        
    images_paths = sorted(glob(f"{train_folder}/**/*.jpg", recursive=True))
    logging.debug(f"Found {len(images_paths)} images")
        
    logging.debug("For each image, get its UTM east, UTM north and heading from its path")
    images_metadatas = [p.split("@") for p in images_paths]
    # field 1 is UTM east, field 2 is UTM north, field 9 is heading
    utmeast_utmnorth_heading = [(m[1], m[2], m[9]) for m in images_metadatas]
    utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(float)
    
    logging.debug("For each image, get class and group to which it belongs")
    class_id = [get__class_id(*m, M, alpha)
                            for m in utmeast_utmnorth_heading]
    
    logging.debug("Group together images belonging to the same class")
    images_per_class = defaultdict(list)
    for image_path, class_id in zip(images_paths, class_id):
        images_per_class[class_id].append(image_path)
    
    # Images_per_class is a dict where the key is class_id, and the value
    # is a list with the paths of images within that class.
    images_per_class = [(v, k) for k, v in images_per_class.items() if len(v) >= min_images_per_class]
    
    samples = [(img, sublist[1]) for sublist in images_per_class for img in sublist[0]]

    print(f"samples len: {len(samples)}")

    train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(utmeast_utmnorth_heading)
    #positives_per_query = knn.radius_neighbors(utmeast_utmnorth_heading,
    #                                                radius=positive_dist_threshold,
    #                                                return_distance=False)
    distances, indices = knn.kneighbors(utmeast_utmnorth_heading, n_neighbors=20)
    
    with open("/content/AML_Rerank_MobileNet/rrt_sop_caches/rrt_r50_sop_nn_inds_train.pkl", "wb+") as f:
        pickle.dump(indices, f)

    train_set = ImageDataset(samples=samples, transform=train_transform)

    # Open test/val folder
    database_folder = os.path.join(test_folder, "database")
    queries_folder = os.path.join(test_folder, queries_folder_name)

    base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    database_paths = sorted(glob(os.path.join(database_folder, "**", "*.jpg"), recursive=True))
    queries_paths = sorted(glob(os.path.join(queries_folder, "**", "*.jpg"),  recursive=True))
        
    # The format must be path/to/file/@utm_easting@utm_northing@...@heading@....@.jpg
    database_utms = np.array([(path.split("@")[1], path.split("@")[2], 0) for path in database_paths]).astype(float)
    queries_utms = np.array([(path.split("@")[1], path.split("@")[2], 0) for path in queries_paths]).astype(float)

    # img, class_id
    class_id_database = [get__class_id(*m, M, alpha)
                            for m in database_utms]
    class_id_queries = [get__class_id(*m, M, alpha)
                            for m in queries_utms]
    
    print(len(set(class_id_queries)))
    
    images_per_class_database = defaultdict(list)
    for image_path, class_id in zip(database_paths, class_id_database):
        images_per_class_database[class_id].append(image_path)

    images_per_class_database = [(v, k) for k, v in images_per_class_database.items()]

    images_per_class_queries = defaultdict(list)
    for image_path, class_id in zip(queries_paths, class_id_queries):
        images_per_class_queries[class_id].append(image_path)
    
    images_per_class_queries = [(v, k) for k, v in images_per_class_queries.items()]

    samples_database = [(img, sublist[1]) for sublist in images_per_class_database for img in sublist[0]]
    samples_queries = [(img, sublist[1]) for sublist in images_per_class_queries for img in sublist[0]]

    print(samples_queries[0][0])

    print(f"samples_database len: {len(samples_database)}") #8023
    print(f"samples_queries len: {len(samples_queries)}") #8002

    database_utms = [(x[0], x[1]) for x in database_utms]
    queries_utms = [(x[0], x[1]) for x in queries_utms]

    knn = NearestNeighbors(n_jobs=-1, algorithm="brute", metric="euclidean")
    knn.fit(database_utms)
    distances, positives_per_query = knn.radius_neighbors(queries_utms,
                                                    radius=positive_dist_threshold,
                                                    return_distance=True, sort_results=True)
    
    print(f"len first query: {len(distances[0])}")
    print(f"distances: {distances[0][0:8]}")
    print(f"positives: {positives_per_query[0][0:8]}")
    print(f"image_path of closest: {samples_database[positives_per_query[0][0]][0]}")
    
    with open("/content/AML_Rerank_MobileNet/rrt_sop_caches/rrt_r50_sop_nn_inds_positives.pkl", "wb+") as f:
        pickle.dump(positives_per_query, f)
        
    distances, indices = knn.kneighbors(queries_utms, n_neighbors=len(samples_database))

    print(f"distances: {distances[0][0:8]}")
    print(f"test: {indices[0][0:8]}")
    
    with open("/content/AML_Rerank_MobileNet/rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl", "wb+") as f:
        pickle.dump(indices, f)
    
    # queries_v1 folder
    query_set = ImageDataset(samples=samples_queries, transform=base_transform)
    # database folder
    gallery_set =  ImageDataset(samples=samples_database, transform=base_transform)

    # query = queriesv1, gallery = database
    return train_set, (query_set, gallery_set)


class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    gallery: Optional[DataLoader] = None


@data_ingredient.capture
def get_loaders(batch_size, test_batch_size, 
        num_workers, pin_memory, 
        sampler, recalls,
        num_iterations=None, 
        num_identities=None,
        train_cache_nn_inds=None,
        test_cache_nn_inds=None):

    train_set, (query_set, gallery_set) = get_sets()

    print(f"Batch size for training set: {batch_size}")

    if sampler == 'random':
        train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=True)
        print(f"Using random sampler")
    elif sampler == 'triplet':
        if train_cache_nn_inds and osp.exists(train_cache_nn_inds):
            train_sampler = TripletSampler(train_set.targets, batch_size, train_cache_nn_inds)
            print(f"Using triplet sampler")
        else:
            # For evaluation only
            train_sampler = None
            print(f"Using the for evaluation only sampler")
    elif sampler == 'random_id':
        train_sampler = RandomReplacedIdentitySampler(train_set.targets, batch_size, 
            num_identities=num_identities, num_iterations=num_iterations)
        print(f"Using random_id sampler")
    else:
        raise ValueError('Invalid choice of sampler ({}).'.format(sampler))
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    query_loader = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    gallery_loader = None
    if gallery_set is not None:
        gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return MetricLoaders(train=train_loader, query=query_loader, gallery=gallery_loader, num_classes=max(train_set.targets) + 1), recalls


def get__class_id(utm_east, utm_north, heading, M, alpha):
    """Return class_id and group_id for a given point.
        The class_id is a triplet (tuple) of UTM_east, UTM_north and
        heading (e.g. (396520, 4983800,120)).
        The group_id represents the group to which the class belongs
        (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
    """
    rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
    rounded_utm_north = int(utm_north // M * M)
    rounded_heading = int(heading // alpha * alpha)
    
    class_id = (rounded_utm_east, rounded_utm_north, rounded_heading)

    return class_id

def calculate_neighbors(t, candidates, n_neighbors):
    # (1,2,3)
    distances = []

    for i, n in enumerate(candidates):
        distances.append(distance(t[0], n[0], t[1], n[1], t[2], n[2]))

    dist = sorted(distances)
    indices = np.argsort(dist)

    return indices[:n_neighbors]

def distance(x1, x2, y1, y2, z1, z2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) 
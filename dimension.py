import pandas as pd
import numpy as np
from typing import Set

def euclidean(a, b):
    diff = a-b
    return np.sqrt(np.dot((diff).T, diff))

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def hamming(a,b):
    return np.count_nonzero(a!=b)

def manhattan(a,b):
    return np.sum(np.abs(a-b))

def chebyshev(a,b):
    return np.max(np.abs(a-b))

def minkowski(a,b,p):
    return np.sum(np.abs(a - b)**p)**(1/p)

def jaccard_2(a,b):    
    return 1 - (np.sum(a & b)/np.sum(a | b))

def dice_index(a,b):
    return (2.0 * np.size(np.intersect1d(a, b))) / (np.size(a) + np.size(b))

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    return (
        12742 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2.0)**2 + 
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0)**2))
        )

lat1, lon1 = -23.550520, -46.633308
lat2, lon2 = -22.906847, -43.172896
a = np.arange(0, 2000, 2)
b = np.arange(0, 1000, 1)
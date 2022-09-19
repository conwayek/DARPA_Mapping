"""
Written by:
Dr. Eamon K. Conway
Geospatial Development Center (GDC)
Kostas Research Institute for Homeland Securty
Northeastern University

Contact:
e.conway@northeastern.edu

Date:
9/19/2022

DARPA Critical Mineral Challenge 2022

Purpose:
extract text from tiles

Args:
tiled imagery [n,m,3,N], N images
top left position on image
bottom right position in image

Out:
keywords,
centers,
bboxes
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras_ocr
import pandas as pd


def main(tile,tl,br):
            pipeline = keras_ocr.pipeline.Pipeline(max_size=2000,scale=2)

            keywords=[]
            bboxes=[]
            centers=[]
            toponym_info = []
            detect_kwargs = {}
            detect_kwargs['detection_threshold']=0.7
            detect_kwargs['text_threshold']=0.4
            detect_kwargs['size_threshold']=20


            for i in range(tile.shape[-1]):
                    prediction_groups = pipeline.recognize([tile[:,:,:,i]],detection_kwargs=detect_kwargs)[0];
                    for prediction in prediction_groups:
                        keywords.append(prediction[0])
                        bbox=prediction[1]
                        xs=[int(item[0]) for item in bbox]
                        ys=[int(item[1]) for item in bbox]
                        xmin = int(min(xs)+ tl[1,i])
                        xmax = int(max(xs)+ tl[1,i])
                        ymin = int(min(ys)+ tl[0,i])
                        ymax = int(max(ys)+ tl[0,i])

                        bboxes.append(((xmin, ymin), (xmax, ymax)))
                        centers.append((xmin+int((xmax-xmin)/2), ymin+int((ymax-ymin)/2)))

            pipeline=None
            prediction_groups = None
            return keywords,bboxes,centers

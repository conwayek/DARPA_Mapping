import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras_ocr
import pandas as pd
import sys
import rasterio
import glob

def tileall(img):
    nres=750

    nalf = int(nres*0.5)

    ncol = int(np.ceil((img.shape[1]) / nres))+1
    nrow =int( np.ceil((img.shape[0]) / nres))+1

    ntile = nrow*ncol

    tl = np.zeros((2,ntile),dtype=int)
    br = np.zeros((2,ntile),dtype=int)

    # create a square atr each corner of the bounds
    # there are four to create: [y,x]
    tl = np.zeros(((2,ntile)),dtype=int)
    br = np.zeros(((2,ntile)),dtype=int)
    tile = np.zeros((nres,nres,3,ntile))
    count=-1


    for j in range(ncol-1):
        startx = max(0,int(j*nres) )
        stopx = min(int( (j+1)*nres ) , img.shape[1])
        #print(startx,stopx)
        for i in range(nrow-1):
            count+=1
            starty = max(0,int(i*nres))
            stopy = min(int((i+1)*nres),img.shape[0])
            tl[1,count] = startx
            br[1,count] = stopx
            tl[0,count] = starty
            br[0,count] = stopy
            
            tile[:(stopy-starty),:(stopx-startx),:,count] = img[starty:stopy,startx:stopx,:]

    tile = np.array(tile,dtype=np.uint8)
    
    return tl,br,tile


def main(image_dir,out_dir):
    
    images = glob.glob(image_dir+'*.tif')
    nimg = len(images)
    for image_name in images:
        outname = os.path.join(out_dir,os.path.basename(image_name).split('.tif')[0]+'.csv')
        
        if(os.path.exists(outname)==False):
        
            with rasterio.open(image_name,'r') as f:
                img = f.read()
            img=img.transpose((1,2,0))

            # get tiles
            tl,br,tile = tileall(img)

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


            npts = len(keywords)

            keys = np.array(keywords)

            bbox_left_top_x = np.zeros(npts)
            bbox_left_top_y = np.zeros(npts)
            bbox_right_bot_x = np.zeros(npts)
            bbox_right_bot_y = np.zeros(npts)

            center_x = np.zeros(npts)
            center_y = np.zeros(npts)

            for i in range(npts):
                #print(bboxes[i])
                bbox_left_top_x[i] = bboxes[i][0][0]
                bbox_left_top_y[i] = bboxes[i][0][1]
                bbox_right_bot_x[i]= bboxes[i][1][0]
                bbox_right_bot_y[i]= bboxes[i][1][1]

                center_x[i] = centers[i][0]
                center_y[i] = centers[i][1]

            arr = np.array([keys,bbox_left_top_x,bbox_left_top_y,bbox_right_bot_x,bbox_right_bot_y,center_x,center_y],dtype=object)
            #print(arr.shape)
            df = pd.DataFrame(arr.T,index=None,columns=['Key','BboxTLx','BboxTLy','BboxBRx','BboxBRy','Cenx','Ceny'])
            df.to_csv(outname) 


if __name__=="__main__":
    #out_dir = sys.argv[2]
    #image_dir = sys.argv[1]
    #image_path = sys.argv[1]
    out_dir = '/scratch/e.conway/DARPA_MAPS/TextExtraction/ImageText/'
    image_dir = '/scratch/e.conway/DARPA_MAPS/Training/'
    
    main(image_dir,out_dir)

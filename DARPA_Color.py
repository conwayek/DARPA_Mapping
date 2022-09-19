import json
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras_ocr
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn import linear_model
from scipy.optimize import curve_fit
import math
import ColorDetect
import sys
from sys import exit
import time
import rasterio
import FinalNumbers2 as Fnum
import Tiling
import KeywordsEdit
import MergeKeys
import PairMatching
import WriteFile
import KerasPipeline

def lin_line(x, A, B): 
    return A*x + B


def main(image_dir,image_path,out_dir,clue_dir):
    
    if('Training' in image_dir):
        training=True
    else:
        training=False
    
    tz=time.time()
    
    us_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Washington DC', 'District of Columbia', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    


    # get state boundaries in lon/lat
    data = json.load(open(os.path.join(image_dir,'GeoJSONgz_2010_us_040_00_20m.json')))
    df = pd.DataFrame(data["features"])

    df['Location'] = df['properties'].apply(lambda x: x['NAME'])
    df['Type'] = df['geometry'].apply(lambda x: x['type'])
    df['Coordinates'] = df['geometry'].apply(lambda x: x['coordinates'])


    #img = cv2.imread(os.path.join(image_dir,image_path))
    with rasterio.open(os.path.join(image_dir,image_path)) as f:
        img = f.read()
        print('Image Shape = ',img.shape)
    img = img.transpose((1,2,0))
    
    
    space=0
    failure,mask,bounds = ColorDetect.main(image_dir,out_dir,image_path,width=space)
    #bounds = np.genfromtxt('/scratch/e.conway/DARPA_MAPS/Results/GEO_0095_Mask.txt',delimiter=',')
    print(bounds)
    #failure = True
    redo = False
    if(failure==True):
        bounds = np.array([np.nan,np.nan,np.nan,np.nan])
        redo=True
    done=False        
    try:
        #test_routine=True
        #while test_routine == True:
        while done == False:
            if(redo==True):
                print('-----Reattempting Model-----')
                print('Reset bounds')
                bounds = np.array([np.nan,np.nan,np.nan,np.nan])
                
            #----------------------------------#
            # load the mask of the map
            if(failure==False and redo==False):
                tl,br,tile = Tiling.main(bounds,img)
            else:
                tl,br,tile = Tiling.tileall(img)

            keywords,bboxes,centers = KerasPipeline.main(tile,tl,br)  
            keywords,bboxes,centers = MergeKeys.main(keywords,bboxes,centers)
            #----------------------------------#

            """
            with rasterio.open(os.path.join(image_dir,image_path)) as f:
                img = f.read()
            img = img.transpose((1,2,0))
            try:
                fig=plt.figure(figsize=(5,5))
                for bbox in bboxes:
                    cv2.rectangle(img, pt1=bbox[0], pt2=bbox[1], color=(255,0,0), thickness=0)
                plt.imshow(img[:,:,:],aspect='auto')
                plt.savefig(os.path.join(out_dir,image_path).split('.tig')[0]+'_Boxes.png',dpi=1000)
                plt.close()
            except:
                pass
            """
            #----------------------------------#
            #Get clue
            clue_file = os.path.join(clue_dir,image_path.split('.tif')[0]+'_clue.csv')

            if(training==False):
                # cannot find one for training, but we can make one for now
                try:
                    df = pd.read_csv(clue_file)
                    clue_x = df['NAD83_x'].values
                    clue_y = df['NAD83_y'].values
                except Exception as e: 
                    print(f"Exception {e} raised for file ",clue_file)
                    exit()
            else:
                try:
                    clues = np.genfromtxt(os.path.join(clue_dir,clue_file),delimiter=',')
                    clue_x = clues[0]
                    clue_y = clues[1]
                except Exception as e: 
                    print(f"Exception {e} raised for file ",clue_file)
                    exit()

            #----------------------------------#
            print(keywords)
            tot_numbers,tot_num_centers,tot_num_boxes = KeywordsEdit.main(keywords,centers,bboxes,clue_x,clue_y)

            print(tot_numbers)


            cluex_array=np.zeros(len(tot_numbers)) ; cluex_array[:] = clue_x
            cluey_array= np.zeros(len(tot_numbers)) ; cluey_array[:] = clue_y
            final_numbers,final_num_centers,final_num_boxes = Fnum.main(tot_numbers,tot_num_centers,tot_num_boxes,cluex_array,cluey_array)
            
            #----------------------------------#

            final_numbers=np.array(final_numbers)
            final_num_centers=np.array(final_num_centers)
            final_num_boxes=np.array(final_num_boxes)

            neg = -final_numbers
            final_numbers=np.concatenate((final_numbers,neg))
            final_num_centers=np.concatenate((final_num_centers,final_num_centers))
            final_num_boxes=np.concatenate((final_num_boxes,final_num_boxes))    
            #----------------------------------#
            lon = []
            lat=[]
            clon = []
            clat = []
            for i in range(len(final_numbers)):
                if(math.isclose(final_numbers[i],clue_x,abs_tol=2)):     
                    lon.append(final_numbers[i])
                    clon.append(final_num_centers[i])

            for i in range(len(final_numbers)):
                if(math.isclose(final_numbers[i],clue_y,abs_tol=2)):     
                    lat.append(final_numbers[i])
                    clat.append(final_num_centers[i])

            lon=np.array(lon,dtype=np.float64)
            lat=np.array(lat,dtype=np.float64)
            clon=np.array(clon,dtype=np.float64)
            clat=np.array(clat,dtype=np.float64)
            print('lon = ',lon)
            print('clon = ',clon)
            #print('---')
            print('lat = ',lat)
            print('clat = ',clat)

            
            

            #----------------------------------#
            print(bounds)
            lat3d,lon3d = PairMatching.main(lat,clat,lon,clon,img.shape,clue_x,clue_y,bounds)
              
            print('lat 3',lat3d)

            print('lon 3',lon3d)
            
            #----------------------------------# 
            if(redo==False):
                redo = WriteFile.main(lat3d,lon3d,training,img.shape,out_dir,image_path,image_dir,clue_x,clue_y,redo,\
lon,lat,clon,clat)
                if(np.isfinite(np.sum(bounds))==False):
                    done=True
                if(redo==False):
                    done=True
            elif(redo==True):
                redo = WriteFile.main(lat3d,lon3d,training,img.shape,out_dir,image_path,image_dir,clue_x,clue_y,redo,\
lon,lat,clon,clat)
                done=True
                
            
            #"""
            test_routine = False
            #"""
    except Exception as e:
            print(f"Exception {e}: File = ",image_path)
            #Get clue
            clue_file = os.path.join(clue_dir,image_path.split('.tif')[0]+'_clue.csv')


            # cannot find one for training, but we can make one for now
            try:
                if(training==False):
                    df = pd.read_csv(clue_file)
                    clue_x = df['NAD83_x'].values
                    clue_y = df['NAD83_y'].values
                else:
                    clues = np.genfromtxt(os.path.join(clue_dir,clue_file),delimiter=',')
                    clue_x = clues[0]
                    clue_y = clues[1]
            except Exception as e: 
                print(f"Exception {e} raised for file ",clue_file)
                return
            
            fitx_global = True
            fity_global = True
            
            print('Fit: xglobal,yglobal = ',fitx_global,', ',fity_global)

            real_res = os.path.join(image_dir,image_path.split('.tif')[0]+'.csv')
            df = pd.read_csv(real_res)
            row_test = df['row'].values
            col_test = df['col'].values
            if(training==True):
                row_lat = df['NAD83_y'].values
                col_lon = df['NAD83_x'].values
            npts = len(row_test)

            calc_lon = np.zeros(npts)
            calc_lat = np.zeros(npts)
            for i in range(npts):        
                calc_lon[i] = clue_x
                calc_lat[i] = clue_y
                if(training==True):
                    meas_lon = col_lon[i]
                    meas_lat = row_lat[i]
                    diff_lat = meas_lat - calc_lat[i]
            if(training==True):
                np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([row_test,col_test,row_lat,col_lon,calc_lat,calc_lon]).T,\
                          fmt = '%.7f,%.7f,%.7f,%.7f,%.7f,%.7f',delimiter=',')
            else:
                np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([row_test,col_test,calc_lat,calc_lon]).T,\
                      fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
            done=True
            #"""
    print('Time = ',time.time()-tz)
    return

if __name__=="__main__":
    out_dir = sys.argv[1]
    image_dir = sys.argv[2]
    image_path = sys.argv[3]
    clue_dir = sys.argv[4]
    
    print(out_dir,image_dir,image_path,clue_dir)
    
    main(image_dir,image_path,out_dir,clue_dir)
    

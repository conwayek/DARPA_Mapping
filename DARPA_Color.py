import json
import geopy.distance
import numpy as np
from sklearn.cluster import DBSCAN
import os
import pickle
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras_ocr
import pandas as pd
from geopy.geocoders import Nominatim
import requests
from shapely.geometry import Polygon, Point, MultiPolygon
#from lmfit import minimize, Parameters, Parameter, printfuncs, fit_report
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
    #bounds = np.genfromtxt('/scratch/e.conway/DARPA_MAPS/Results/GEO_0008_Mask.txt',delimiter=',')
    #print(bounds)
    #failure = False
    if(failure==True):
        bounds = np.array([np.nan,np.nan,np.nan,np.nan])

        
    try:
            #test_routine=True
            #while test_routine == True:

            #----------------------------------#
            # load the mask of the map
            if(failure==False):
                tl,br,tile = Tiling.main(bounds,img)
            else:
                tl,br,tile = Tiling.tileall(img)

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

            #----------------------------------#
            keywords,bboxes,centers = MergeKeys.main(keywords,bboxes,centers)
            #----------------------------------#

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

            #----------------------------------#
            #Get clue
            clue_file = os.path.join(clue_dir,image_path.split('.tif')[0]+'_clue.csv')

            if(training==False):
                # cannot find one for training, but we can make one for now
                try:
                    df = pd.read_csv(clue_file)
                    clue_x = df['NAD83_x'].values
                    clue_y = df['NAD83_y'].values
                    #clues = np.genfromtxt(os.path.join(clue_dir,clue_file),delimiter=',')
                    #clue_x = clues[0]
                    #clue_y = clues[1]
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
                if(math.isclose(final_numbers[i],clue_x,abs_tol=3)):     
                    lon.append(final_numbers[i])
                    clon.append(final_num_centers[i])

            for i in range(len(final_numbers)):
                if(math.isclose(final_numbers[i],clue_y,abs_tol=3)):     
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

            #"""

            fitx_own = False
            fity_own = False
            
            fitx_help = False
            fity_help = False
            
            fitx_global = False
            fity_global = False

            if(lon3d.shape[1] > 1):
                if(lon3d[2,0] != lon3d[2,1]):
                    fitx_own=True
                    poptx,pcovx = curve_fit(lin_line,lon3d[1,:],lon3d[2,:])
                    X = np.linspace(0,img.shape[1]-1,img.shape[1])
                    Y = lin_line(X,*poptx)
                    lon_max = Y[-1]
                    lon_min = Y[0]
                    print('lon max/min own = ',lon_max,lon_min)
                    if(lon_max - lon_min > 3):
                        fitx_own=False


            if(lat3d.shape[1] > 1):
                if(lat3d[2,0] != lat3d[2,1]):
                    fity_own=True
                    popty,pcovy = curve_fit(lin_line,lat3d[0,:],lat3d[2,:])
                    X = np.linspace(0,img.shape[0]-1,img.shape[0])
                    Y = lin_line(X,*popty)
                    
                    lat_min = Y[-1]
                    lat_max = Y[0]
                    print('lat max/min own = ',lat_max,lat_min)
                    if(lat_max - lat_min > 3):
                        fity_own=False

            # failed to fitx but fitted y, so copy y
            if(fitx_own==False and fity_own==True and lon3d.shape[1]==2):
                dist = np.zeros(2)
                dist[0] = abs(abs(clue_x) - abs(lon3d[2,0]))
                dist[1] = abs(abs(clue_x) - abs(lon3d[2,1]))
                poptx=popty.copy()
                poptx[0] = -poptx[0]
                if(dist[0]<dist[1]):
                    c = lon3d[2,0] - poptx[0]*lon3d[1,0]
                    poptx[1] = c
                    fitx_help=True
                else:
                    c = lon3d[2,1] - poptx[0]*lon3d[1,1]
                    poptx[1] = c
                    fitx_help=True
                X = np.linspace(0,img.shape[1]-1,img.shape[1])
                Y = lin_line(X,*poptx)
                    
                lon_max = Y[-1]
                lon_min = Y[0]
                print('lon max/min help = ',lon_max,lon_min)
                if(lon_max - lon_min > 3):
                    fitx_help=False
                    fitx_global=True
                    
            elif(fitx_own==False and fity_own==True and lon3d.shape[1]==1):
                if(lon3d[2,0]!=0):
                    poptx=popty.copy()
                    poptx[0] = -poptx[0]
                    c = lon3d[2,0] - poptx[0]*lon3d[1,0]
                    poptx[1] = c
                    fitx_help=True
                    X = np.linspace(0,img.shape[1]-1,img.shape[1])
                    Y = lin_line(X,*poptx)

                    lon_max = Y[-1]
                    lon_min = Y[0]
                    print('lon max/min help = ',lon_max,lon_min)
                    if(lon_max - lon_min > 3):
                        fitx_help=False
                        fitx_global=True
            
                

            # failed to fity but fitted x, so copy x
            if(fity_own==False and fitx_own==True and lat3d.shape[1]==2):
                # we have 2 numbers, need to pick closest to clue
                dist = np.zeros(2)
                dist[0] = abs(abs(clue_y) - abs(lat3d[2,0]))
                dist[1] = abs(abs(clue_y) - abs(lat3d[2,1]))
                if(dist[0]<dist[1]):
                    popty=poptx.copy()
                    popty[0] = -popty[0]
                    c = lat3d[2,0] - popty[0]*lat3d[0,0]
                    popty[1] = c
                    fity_help=True
                else:
                    popty=poptx.copy()
                    popty[0] = -popty[0]
                    c = lat3d[2,1] - popty[0]*lat3d[0,1]
                    popty[1] = c
                    fity_help=True
                X = np.linspace(0,img.shape[0]-1,img.shape[0])
                Y = lin_line(X,*popty)
                    
                lat_min = Y[-1]
                lat_max = Y[0]
                print('lat max/min help = ',lat_max,lat_min)
                if(lat_max - lat_min > 3):
                        fity_help=False
                        fity_global=True
            elif(fity_own==False and fitx_own==True and lat3d.shape[1]==1):
                if(lat3d[2,0]!=0):
                    popty=poptx.copy()
                    popty[0] = -popty[0]
                    c = lat3d[2,0] - popty[0]*lat3d[0,0]
                    popty[1] = c
                    fity_help=True
                    X = np.linspace(0,img.shape[0]-1,img.shape[0])
                    Y = lin_line(X,*popty)         
                    lat_min = Y[-1]
                    lat_max = Y[0]
                    print('lat max/min help = ',lat_max,lat_min)
                    if(lat_max - lat_min > 3):
                        fity_help=False
                        fity_global=True
                        
            if(fitx_own == False and fitx_help==False):
                fitx_global = True
            if(fity_own == False and fity_help==False):
                fity_global = True

            print('Fit: xown,yown = ',fitx_own,', ',fity_own)
            print('Fit: xhelp,yhelp = ',fitx_help,', ',fity_help)
            print('Fit: xglobal,yglobal = ',fitx_global,', ',fity_global)
            
            real_res = os.path.join(image_dir,image_path.split('.tif')[0]+'.csv')
            df = pd.read_csv(real_res)
            row_test = df['row'].values
            col_test = df['col'].values
            if(training==True):
                row_lat = df['NAD83_y'].values
                col_lon = df['NAD83_x'].values
            npts = len(row_test)

            if(fitx_own == True or fitx_help==True):
                if(fity_own==True or fity_help==True):
                    # fit x and y correctly
                    calc_lon = np.zeros(npts)
                    calc_lat = np.zeros(npts)
                    for i in range(npts):
                        calc_lon[i] = lin_line(col_test[i],*poptx)
                        calc_lat[i] = lin_line(row_test[i],*popty)
                        if(training==True):
                            meas_lon = col_lon[i]
                            meas_lat = row_lat[i]
                            diff_lat = meas_lat - calc_lat[i]
                            diff_lon = meas_lon - calc_lon[i]

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 2):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 2):
                            calc_lon[i] = clue_x

                    if(training==True):
                        np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([row_test,col_test,row_lat,col_lon,calc_lat,calc_lon]).T,\
                              fmt = '%.7f,%.7f,%.7f,%.7f,%.7f,%.7f',delimiter=',')
                    else:
                        np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([row_test,col_test,calc_lat,calc_lon]).T,\
                              fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')

                else:
                    #only fitted x correctly, not y
                    calc_lon = np.zeros(npts)
                    calc_lat = np.zeros(npts)
                    for i in range(npts):
                        #
                        calc_lon[i] = lin_line(col_test[i],*poptx)
                        calc_lat[i] = clue_y
                        if(training==True):
                            meas_lon = col_lon[i]
                            meas_lat = row_lat[i]
                            diff_lat = meas_lat - calc_lat[i]
                            

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 2):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 2):
                            calc_lon[i] = clue_x
                    if(training==True):
                        np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([row_test,col_test,row_lat,col_lon,calc_lat,calc_lon]).T,\
                              fmt = '%.7f,%.7f,%.7f,%.7f,%.7f,%.7f',delimiter=',')
                    else:
                        np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([row_test,col_test,calc_lat,calc_lon]).T,\
                          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')

            elif(fitx_own == False and fitx_help==False):
                #failed to fit x, but did y
                if(fity_own==True or fity_help==True):
                    calc_lon = np.zeros(npts)
                    calc_lat = np.zeros(npts)
                    for i in range(npts):
                        calc_lon[i] = clue_x
                        calc_lat[i] = lin_line(row_test[i],*popty)
                        if(training==True):
                            meas_lon = col_lon[i]
                            meas_lat = row_lat[i]
                            diff_lat = meas_lat - calc_lat[i]

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 2):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 2):
                            calc_lon[i] = clue_x
                    if(training==True):
                        np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([row_test,col_test,row_lat,col_lon,calc_lat,calc_lon]).T,\
                              fmt = '%.7f,%.7f,%.7f,%.7f,%.7f,%.7f',delimiter=',')
                    else:
                        np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([row_test,col_test,calc_lat,calc_lon]).T,\
                          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')

                else:
                    # did not fit x or y
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
    

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
from lmfit import minimize, Parameters, Parameter, printfuncs, fit_report
from sklearn.linear_model import RANSACRegressor
from sklearn import linear_model
from scipy.optimize import fmin
from scipy.optimize import fmin_slsqp ,least_squares
from scipy.optimize import curve_fit
import math
import ColorDetect
import sys
from sys import exit
import time
import rasterio
import FinalNumbers as Fnum

def lin_line(x, A, B): 
    return A*x + B


def main(image_dir,image_path,out_dir,clue_dir):
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
    #bounds = np.genfromtxt('/scratch/e.conway/DARPA_MAPS/ValidationResults/GEO_0103_Mask.txt',delimiter=',')
    #print(bounds)
    #failure = False
    
    if(failure==True):
        
        #Get clue
        clue_file = os.path.join(clue_dir,image_path.split('.tif')[0]+'_clue.csv')



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
            return
        
        real_res = os.path.join(image_dir,image_path.split('.tif')[0]+'.csv')
        df = pd.read_csv(real_res)
        row_test = df['row'].values
        col_test = df['col'].values
        #row_lat = df['NAD83_y'].values
        #col_lon = df['NAD83_x'].values
        npts = len(row_test)
        
        calc_lon = np.zeros(npts)
        calc_lat = np.zeros(npts)
        for i in range(npts):
            #meas_lon = col_lon[i]
            calc_lon[i] = clue_x
            #meas_lat = row_lat[i]
            calc_lat[i] = clue_y
            #diff_lat = meas_lat - calc_lat[i]

        #np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_lon,row_lat,calc_lon,calc_lat]).T,\
        #              fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
        np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_test,row_test,calc_lon,calc_lat]).T,\
                      fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
        
    else:
        
        #try:
        test_routine=True
        while test_routine == True:

            #----------------------------------#
            # load the mask of the map
            #print('jere1')

            nres=350

            nalf = int(nres*0.5)

            ncol = int(np.ceil((bounds[3]-bounds[2]) / nres))+1
            nrow =int( np.ceil((bounds[1]-bounds[0]) / nres))+1

            ntile = nrow*2 + ncol*2

            tl = np.zeros((2,ntile),dtype=int)
            br = np.zeros((2,ntile),dtype=int)

            # create a square atr each corner of the bounds
            # there are four to create: [y,x]
            tl = np.zeros(((2,ntile)),dtype=int)
            br = np.zeros(((2,ntile)),dtype=int)
            tile = np.zeros((nres,nres,3,ntile))
            count=-1
            #----------- Across the top     
            starty = max(0,int(bounds[0]-nalf))
            stopy = min(int(bounds[0]+nalf),img.shape[0])

            for j in range(ncol):
                count+=1
                startx = max(0,int( bounds[2] + j*nres - nalf))
                stopx = min(int(bounds[2] + (j)*nres + nalf),img.shape[1])
                #print(count,startx,stopx)
                tl[1,count] = startx
                br[1,count] = stopx
                tl[0,count] = starty
                br[0,count] = stopy
                

                tile[:(stopy-starty),:(stopx-startx),:,count] = img[starty:stopy,startx:stopx,:]

            #----------- Across the bottom
            starty = max(0,int(bounds[1]-nalf))
            stopy = min(int(bounds[1]+nalf),img.shape[0])

            for j in range(ncol):
                count+=1
                startx = max(0,int( bounds[2] + j*nres - nalf))
                stopx = min(int(bounds[2] + (j)*nres + nalf),img.shape[1])
                #print(count,startx,stopx)
                tl[1,count] = startx
                br[1,count] = stopx
                tl[0,count] = starty
                br[0,count] = stopy

                tile[:(stopy-starty),:(stopx-startx),:,count] = img[starty:stopy,startx:stopx,:]

            #----------- Down the left side
            startx = max(0,int(bounds[2]-nalf))
            stopx = min(int(bounds[2]+nalf),img.shape[1])
            #print(startx,stopx)
            for j in range(nrow):
                count+=1
                starty = max(0,int( bounds[0] + j*nres - nalf))
                stopy = min(int(bounds[0] + (j)*nres + nalf),img.shape[0])
                #print(count,starty,stopy)
                tl[0,count] = starty
                br[0,count] = stopy
                tl[1,count] = startx
                br[1,count] = stopx

                tile[:(stopy-starty),:(stopx-startx),:,count] = img[starty:stopy,startx:stopx,:]

            #----------- Down the right side
            startx = max(0,int(bounds[3]-nalf))
            stopx = min(int(bounds[3]+nalf),img.shape[1])
            
            #print('2',startx,stopx)


            for j in range(nrow):
                count+=1
                starty = max(0,int( bounds[0] + j*nres - nalf))
                stopy = min(int(bounds[0] + (j)*nres + nalf),img.shape[0])
                print(count,starty,stopy)
                tl[0,count] = starty
                br[0,count] = stopy
                tl[1,count] = startx
                br[1,count] = stopx

                tile[:(stopy-starty),:(stopx-startx),:,count] = img[starty:stopy,startx:stopx,:]
            tile = np.array(tile,dtype=np.uint8)
            

            #----------------------------------#
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
            #print(keywords)
            #exit()
            #----------------------------------#
            new_keywords = []
            new_bboxes = []
            new_centers=[]
            append = []

            # merge keywords if close together
            x = []
            for i in tqdm(range(len(keywords))):
                if(keywords[i]!=''):
                    #print(keywords[i])
                    top_left_1 = [bboxes[i][0][0],bboxes[i][0][1]]
                    top_right_1 = [bboxes[i][1][0],bboxes[i][0][1]]
                    bot_right_1 = [bboxes[i][1][0],bboxes[i][1][1]]
                    bot_left_1 = [bboxes[i][0][0],bboxes[i][1][1]]
                    contin = True
                    for j in range(len(keywords)):
                        if(keywords[j]!=''):
                            top_left_2 = [bboxes[j][0][0],bboxes[j][0][1]]
                            top_right_2 = [bboxes[j][1][0],bboxes[j][0][1]]
                            bot_right_2 = [bboxes[j][1][0],bboxes[j][1][1]]
                            bot_left_2 = [bboxes[j][0][0],bboxes[j][1][1]]
                            # is box 2 close to box 1 on left side of 1
                            if(math.isclose(bot_right_2[0],bot_left_1[0],abs_tol=20) and \
                              math.isclose(bot_right_2[1],bot_left_1[1],abs_tol=20) and i!=j):
                                app = True
                                for en in x:
                                    if([j,i] in x):
                                        app=False
                                #print('1 ',x,[j,i],app,keywords[j],keywords[i])
                                if(app==True):
                                    contin=False
                                    new_word = keywords[j]+keywords[i]
                                    #print('nw = ',new_word)
                                    new_keywords.append(new_word)
                                    new_bboxes.append(bboxes[i])
                                    new_centers.append(centers[i])
                                    x.append([j,i])
                            # is box 2 close to box 1 on right side of 1
                            if(math.isclose(bot_right_1[0],bot_left_2[0],abs_tol=20) and \
                              math.isclose(bot_right_1[1],bot_left_2[1],abs_tol=20) and i!=j):
                                app = True
                                for en in x:
                                    if([i,j] in x):
                                        app=False
                                #print('2 ',x,[i,j],app,keywords[i],keywords[j])
                                if(app==True):
                                    contin = False
                                    new_word = keywords[i]+keywords[j]
                                    new_keywords.append(new_word)
                                    new_bboxes.append(bboxes[i])
                                    new_centers.append(centers[i])
                                    x.append([i,j])
                    app=True
                    if(contin == True):
                            for en in x:
                                for yn in en:
                                    if(i == yn):
                                        app=False
                            if(app==True):
                                new_keywords.append(keywords[i])
                                new_bboxes.append(bboxes[i])
                                new_centers.append(centers[i])

            keywords = new_keywords
            bboxes = new_bboxes
            centers = new_centers
            #print(keywords)

            #----------------------------------#
            #"""
            #img = cv2.imread(os.path.join(image_dir,image_path))
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
            #"""
            #----------------------------------#
            #Get clue
            clue_file = os.path.join(clue_dir,image_path.split('.tif')[0]+'_clue.csv')


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

            #----------------------------------#

            for i in range(len(keywords)):
                for j in range(len(keywords[i])):
                    if(keywords[i][j]=='o'):
                        if(j==0):
                            new = '0'+keywords[i][j+1:]
                            keywords[i] = new
                        elif(j<(len(keywords[i])-1)):
                            new = keywords[i][0:j]+'0'+keywords[i][j+1:]
                            keywords[i] = new
                        elif(j==(len(keywords[i])-1)):
                            new = keywords[i][0:j]+'0'
                            keywords[i] = new
                    if(keywords[i][j]=='z'):
                        if(j==0):
                            new = '7'+keywords[i][j+1:]
                            keywords[i] = new
                        elif(j<(len(keywords[i])-1)):
                            new = keywords[i][0:j]+'7'+keywords[i][j+1:]
                            keywords[i] = new
                        elif(j==(len(keywords[i])-1)):
                            new = keywords[i][0:j]+'7'
                            keywords[i] = new
                            
            if('5' in str(clue_x) or '5' in str(clue_y)):
                add_5 = True
            else:
                add_5 = False

            if('3' in str(clue_x) or '3' in str(clue_y)):
                add_3 = True
            else:
                add_3=False

            if('8' in str(clue_x) or '8' in str(clue_y)):
                add_8 = True
            else:
                add_8=False
            print(add_3,add_5,add_8)



            for j in range(len(keywords)):
                word = keywords[j]
                for i in range(len(word)):
                    x=word[i]
                    key=word[i]
                    if(key=='l' and i==0):
                        word = '4'+word[1:]
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
                    elif(key=='l' and i>0 and i<len(word)-1):
                        word = word[:i]+'4'+word[i+1:]
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
                    elif(key=='l' and i>0 and i==len(word)-1):
                        word = word[:i]+'4'
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
            
                    if(key=='8' and i==0):
                        word = '3'+word[1:]
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
                    elif(key=='8' and i>0 and i<len(word)-1):
                        word = word[:i]+'3'+word[i+1:]
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
                    elif(key=='8' and i>0 and i==len(word)-1):
                        word = word[:i-1]+'3'
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
            
                    if(key=='s' and i==0):
                        if(add_5==True):
                            word = '5'+word[1:]
                            keywords.append(word)
                            centers.append(centers[j])
                            bboxes.append(bboxes[j])
                        if(add_3==True):
                                y = '3'+word[1:]
                                keywords.append(y)
                                centers.append(centers[j])
                                bboxes.append(bboxes[j])

                    elif(key=='s' and i>0 and i<len(word)-1):
                        if(add_5==True):
                            word = word[:i]+'5'+word[i+1:]
                            keywords.append(word)
                            centers.append(centers[j])
                            bboxes.append(bboxes[j])

                        if(add_3==True):
                                y = word[:i]+'3'+word[i+1:]
                                keywords.append(y)
                                centers.append(centers[j])
                                bboxes.append(bboxes[j])
            
                    elif(key=='s' and i>0 and i==len(word)-1):
                        if(add_5==True):
                            word = word[:i-1]+'5'
                            word = word[:i]+'5'+word[i+1:]
                            keywords.append(word)
                            centers.append(centers[j])
                            bboxes.append(bboxes[j])
                        if(add_3==True):
                                y = word[:i-1]+'3'
                                keywords.append(y)
                                centers.append(centers[j])
                                bboxes.append(bboxes[j])
            

            tot_numbers = []
            tot_num_centers = []
            tot_num_boxes = []
            for i in range(len(keywords)):
                word = keywords[i]
                #print(word)
                #if('r' in word):
                #    print(word,centers[i])
                if(word.isdigit() and len(word)>1 and len(word)<=12):
                    tot_numbers.append(word)
                    tot_num_centers.append(centers[i])
                    tot_num_boxes.append(bboxes[i])
                elif(len(word)==3):
                    if(word[0:-1].isdigit()):
                        tot_numbers.append(word[0:-1])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[1:].isdigit()):
                        tot_numbers.append(word[1:])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                elif(len(word)==4):
                    if(word[0:3].isdigit()):
                        tot_numbers.append(word[0:3])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[1:].isdigit()):
                        tot_numbers.append(word[1:])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[0:2].isdigit()):
                        if(word[-2:] == 'oo'):
                            word = word[0:2]+'00'
                            tot_numbers.append(word)
                            tot_num_centers.append(centers[i])
                            tot_num_boxes.append(bboxes[i])
                        elif(word[-2] == 'o'):
                            word = word[0:2]+'0'
                            tot_numbers.append(word)
                            tot_num_centers.append(centers[i])
                            tot_num_boxes.append(bboxes[i])
                        else:
                            tot_numbers.append(word[0:2])
                            tot_num_centers.append(centers[i])
                            tot_num_boxes.append(bboxes[i])
                elif(len(word)==8):
                    for j in range(len(word)):
                        x = word[j]
                        if(x=='o' and j<(len(word)-1)):
                            word=word[:j]+'0'+word[j+1:]
                        elif(x=='o' and j==(len(word)-1)):
                            word=word[:j]+'0'
                    if(word.isdigit()):
                        tot_numbers.append(word)
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                elif(len(word)==7):
                    not_number=True
                    for x in word:
                        if(x.isdigit() == True):
                            not_number = False
                    if(not_number == False):
                        for j in range(len(word)):
                            x = word[j]
                            if(x.isdigit()==False and j<(len(word)-1)):
                                word=word[:j]+'0'+word[j+1:]
                            elif(x.isdigit()==False and j==(len(word)-1)):
                                word=word[:j]+'0'
                        if(word.isdigit()):
                            tot_numbers.append(word)
                            tot_num_centers.append(centers[i])
                            tot_num_boxes.append(bboxes[i])
                elif(len(word)==5):
                    if(word[0:-1].isdigit()):
                        tot_numbers.append(word[0:-1])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[1:].isdigit()):
                        tot_numbers.append(word[1:])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[0:2].isdigit()):
                        tot_numbers.append(word[0:2])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                elif(len(word)==6):
                    if(word[0:-1].isdigit()):
                        tot_numbers.append(word[0:-1])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[1:].isdigit()):
                        tot_numbers.append(word[1:])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[0:4].isdigit()):
                        tot_numbers.append(word[0:4])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[0:3].isdigit()):
                        tot_numbers.append(word[0:3])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])



            print(tot_numbers)
            #print(tot_num_centers)
            #for key in tot_numbers:
            #    if('104' in key):
            #        print(key)
            #----------------------------------#

             
            final_numbers,final_num_centers,final_num_boxes = Fnum.main(tot_numbers,tot_num_centers,tot_num_boxes)
            #"""
            #print(final_numbers)
            #print(final_num_centers)
            #print(final_num_boxes)

            #exit()
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
            #calculate distance to our four corners
            top_left = [bounds[0],bounds[2]]
            bot_left = [bounds[1],bounds[2]]
            bot_right = [bounds[1],bounds[3]]
            top_right = [bounds[0],bounds[3]]

            print(top_left)
            print(bot_left)
            print(bot_right)
            print(top_right)

            #----------------------------------#
            #create duplicate set of lats
            dup_lats = []
            counter=-1
            for i in range(len(lat)):
                if(lat[i] not in dup_lats):
                    counter+=1
                    dup_lats.append(lat[i])

            dup_lat_final = []
            dup_lat_final_cen = []
            for i in range(len(dup_lats)):
                x = []
                y = []
                for j in range(len(lat)):
                    if(dup_lats[i]==lat[j]):
                        x.append(lat[j])
                        y.append(clat[j])
                dup_lat_final.append(x)
                dup_lat_final_cen.append(y)
            print(dup_lat_final)
            #----------------------------------#   
            #create duplicate set of lons
            dup_lons = []
            counter=-1
            for i in range(len(lon)):
                if(lon[i] not in dup_lons):
                    counter+=1
                    dup_lons.append(lon[i])

            dup_lon_final = []
            dup_lon_final_cen = []
            for i in range(len(dup_lons)):
                x = []
                y = []
                for j in range(len(lon)):
                    if(dup_lons[i]==lon[j]):
                        x.append(lon[j])
                        y.append(clon[j])
                dup_lon_final.append(x)
                dup_lon_final_cen.append(y)
            #print(dup_lon_final_cen)
            print(dup_lon_final)
            # we have len(dup_lat_final_cen) sets of lats to sort through....
            # the smallest (most negative) is first
            #----------------------------------#   
            dist_y_top = []
            dist_y_bot = []

            dist_y_ul = []
            dist_y_ur = []
            dist_y_lr = []
            dist_y_ll = []

            glob_pointer = []
            for i in range(len(dup_lat_final)):
                y1=[]
                y2=[]
                yul=[]
                yur=[]
                ylr=[]
                yll=[]
                pointer = []
                for j in range(len(dup_lat_final[i])):
                    pos_y = dup_lat_final_cen[i][j][1]
                    #print(dup_lat_final[i][j],dup_lat_final_cen[i][j][1],top_left[1],abs(pos_y - top_left[0]))
                    n1 = np.linalg.norm((dup_lat_final_cen[i][j] - top_left[::-1]))
                    n2 = np.linalg.norm((dup_lat_final_cen[i][j] - top_right[::-1]))
                    n3 = np.linalg.norm((dup_lat_final_cen[i][j] - bot_right[::-1]))
                    n4 = np.linalg.norm((dup_lat_final_cen[i][j] - bot_left[::-1]))
                    print(dup_lat_final[i][j],dup_lat_final_cen[i][j],bot_right[::-1],abs(pos_y - bot_left[0]),n1,n2,n3,n4)
                    #print(pos_y,abs(pos_y - top_left[1]))
                    y1.append(abs(pos_y - top_left[0]))
                    y2.append(abs(pos_y - bot_left[0]))
                    yul.append(abs(n1))
                    yur.append(abs(n2))
                    ylr.append(abs(n3))
                    yll.append(abs(n4))
                    pointer.append(j)
                dist_y_top.append(y1)
                dist_y_bot.append(y2)
                glob_pointer.append(pointer)
                dist_y_ul.append(yul)
                dist_y_ur.append(yur)
                dist_y_lr.append(ylr)
                dist_y_ll.append(yll)

           # we have len(dup_lon_final_cen) sets of lons to sort through....
            # the smallest (most negative) is first
            dist_x_left = []
            dist_x_right = []
            dist_x_ul = []
            dist_x_ur = []
            dist_x_lr = []
            dist_x_ll = []
            glob_pointer_lon = []
            for i in range(len(dup_lon_final)):
                x1=[]
                x2=[]
                xul=[]
                xur=[]
                xlr=[]
                xll=[]
                pointer_x = []
                for j in range(len(dup_lon_final[i])):
                    pos_x = dup_lon_final_cen[i][j][0]
                    #print(dup_lat_final_cen[i][j][1],top_left[1],abs(pos_y - top_left[1]))
                    #print(pos_y,abs(pos_y - top_left[1]))
                    n1 = np.linalg.norm((dup_lon_final_cen[i][j] - top_left[::-1]))
                    n2 = np.linalg.norm((dup_lon_final_cen[i][j] - top_right[::-1]))
                    n3 = np.linalg.norm((dup_lon_final_cen[i][j] - bot_right[::-1]))
                    n4 = np.linalg.norm((dup_lon_final_cen[i][j] - bot_left[::-1]))
                    print(dup_lon_final[i][j],dup_lon_final_cen[i][j],top_right[::-1],abs(pos_x - top_right[1]),n1,n2,n3,n4)
                    x1.append(abs(pos_x - top_left[1]))
                    x2.append(abs(pos_x - top_right[1]))
                    xul.append(abs(n1))
                    xur.append(abs(n2))
                    xlr.append(abs(n3))
                    xll.append(abs(n4))
                    pointer_x.append(j)
                dist_x_left.append(x1)
                dist_x_right.append(x2)
                glob_pointer_lon.append(pointer_x)
                dist_x_ul.append(xul)
                dist_x_ur.append(xur)
                dist_x_lr.append(xlr)
                dist_x_ll.append(xll) 


            # sorting distances from top/bottom bounds via their distance
            # only keep couple of results
            stored_dist_top = []
            stored_index_top = []
            stored_dist_bot = []
            stored_index_bot = []

            stored_dist_y_ul = []
            stored_dist_y_ur = []
            stored_dist_y_lr = []
            stored_dist_y_ll = []
            stored_indx_y_ul = []
            stored_indx_y_ur = []
            stored_indx_y_lr = []
            stored_indx_y_ll = []

            for i in range(len(dist_y_bot)):
                y2 = np.array(dist_y_bot[i],dtype=int)
                y1 = np.array(dist_y_top[i],dtype=int)
                g = np.array(glob_pointer[i],dtype=int)

                y3=np.array(dist_y_ul[i],dtype=np.float64)
                y4=np.array(dist_y_ur[i],dtype=np.float64)
                y5=np.array(dist_y_lr[i],dtype=np.float64)
                y6=np.array(dist_y_ll[i],dtype=np.float64)

                idy2 = np.argsort(y2)
                idy1 = np.argsort(y1)

                idy3 = np.argsort(y3)
                idy4 = np.argsort(y4) 
                idy5 = np.argsort(y5)
                idy6 = np.argsort(y6) 

                y1=y1[idy1]
                y2=y2[idy2]
                y3=y3[idy3]
                y4=y4[idy4]
                y5=y5[idy5]
                y6=y6[idy6]

                g1=g[idy1]
                g2=g[idy2]
                g3=g[idy3]
                g4=g[idy4]
                g5=g[idy5]
                g6=g[idy6]

                stored_dist_top.append(y1[0:2])
                stored_index_top.append(g1[0:2])
                stored_dist_bot.append(y2[0:2])
                stored_index_bot.append(g2[0:2])

                stored_dist_y_ul.append(y3[0:2])
                stored_dist_y_ur.append(y4[0:2])
                stored_dist_y_lr.append(y5[0:2])
                stored_dist_y_ll.append(y6[0:2])

                stored_indx_y_ul.append(g3[0:2])
                stored_indx_y_ur.append(g4[0:2])
                stored_indx_y_lr.append(g5[0:2])
                stored_indx_y_ll.append(g6[0:2])    


            # sorting distances from top/bottom bounds via their distance
            # only keep couple of results
            stored_dist_left = []
            stored_index_left = []
            stored_dist_right = []
            stored_index_right = []

            stored_dist_x_ul = []
            stored_dist_x_ur = []
            stored_dist_x_lr = []
            stored_dist_x_ll = []
            stored_indx_x_ul = []
            stored_indx_x_ur = []
            stored_indx_x_lr = []
            stored_indx_x_ll = []

            for i in range(len(dist_x_left)):
                x2 = np.array(dist_x_right[i],dtype=int)
                x1 = np.array(dist_x_left[i],dtype=int)
                x3=np.array(dist_x_ul[i],dtype=np.float64)
                x4=np.array(dist_x_ur[i],dtype=np.float64)
                x5=np.array(dist_x_lr[i],dtype=np.float64)
                x6=np.array(dist_x_ll[i],dtype=np.float64)

                g = np.array(glob_pointer_lon[i],dtype=int)

                idx2 = np.argsort(x2)
                idx1 = np.argsort(x1)
                idx3 = np.argsort(x3)
                idx4 = np.argsort(x4) 
                idx5 = np.argsort(x5)
                idx6 = np.argsort(x6) 

                x1=x1[idx1]
                x2=x2[idx2]
                x3=x3[idx3]
                x4=x4[idx4]
                x5=x5[idx5]
                x6=x6[idx6]  

                gx1=g[idx1]
                gx2=g[idx2]

                gx3=g[idx3]
                gx4=g[idx4]
                gx5=g[idx5]
                gx6=g[idx6]

                stored_dist_left.append(x1[0:2])
                stored_index_left.append(gx1[0:2])
                stored_dist_right.append(x2[0:2])
                stored_index_right.append(gx2[0:2])

                stored_dist_x_ul.append(x3[0:2])
                stored_dist_x_ur.append(x4[0:2])
                stored_dist_x_lr.append(x5[0:2])
                stored_dist_x_ll.append(x6[0:2])

                stored_indx_x_ul.append(gx3[0:2])
                stored_indx_x_ur.append(gx4[0:2])
                stored_indx_x_lr.append(gx5[0:2])
                stored_indx_x_ll.append(gx6[0:2])


            # trivial solution, only two lats detected, although in different locations
            # should calculate dist in pixels / delta lat and get approximate lat range in between bounds
            # should be less than about 1.5 degrees as maps are not any larger...
            # there are only 1-2 points in each
            print('distybot ',len(dist_y_bot))
            """
            if(len(dist_y_bot)==2):
                for i in range(len(stored_dist_bot[0])):
                    bot_point = dup_lat_final_cen[0][stored_index_bot[0][i]]
                    bot_lat = dup_lat_final[0][stored_index_bot[0][i]]
                    for j in range(len(stored_dist_top[1])):
                        top_point = dup_lat_final_cen[1][stored_index_top[1][j]]
                        top_lat = dup_lat_final[1][stored_index_top[1][j]]
                        delta_lat = top_lat - bot_lat
                        delta_pix = bot_point[1] - top_point[1]
                        #print(top_lat,bot_lat,delta_lat,top_point,bot_point,delta_pix)

                top_point_cen = dup_lat_final_cen[1][stored_index_top[1][0]]
                top_point_lat = dup_lat_final[1][stored_index_top[1][0]]
                bot_point_cen = dup_lat_final_cen[0][stored_index_bot[0][0]]
                bot_point_lat = dup_lat_final[0][stored_index_bot[0][0]]
                delta_pix = bot_point[1] - top_point_cen[1] 
                delta_lat = top_point_lat - bot_point_lat 
                meter_per_pix = 1e5*delta_lat/delta_pix
                lat3d = np.zeros((3,2))
                lat3d[0,0] = top_point[1] ; lat3d[0,1] = bot_point[1]
                lat3d[1,0] = top_point[0] ; lat3d[1,1] = bot_point[0]
                lat3d[2,0] = top_point_lat; lat3d[2,1] = bot_point_lat 
            """
            done=False
            min_dist_top = 1e6
            min_dist_bot = 1e6
            min_dist_sum = 1e6

            min_dist_ul = 1e6
            min_dist_ur = 1e6
            min_dist_lr = 1e6
            min_dist_ll = 1e6
            
            min_c_dist = 1e6
            totel=1e6
            min_c_dist_arr = []

            print(len(dist_y_bot))

            if(len(dist_y_bot)>=2 and done==False):
                for k in range(len(stored_dist_bot)):
                    for i in range(len(stored_dist_bot[k])):
                        bot_point = dup_lat_final_cen[k][stored_index_bot[k][i]]
                        bot_lat = dup_lat_final[k][stored_index_bot[k][i]]
                        dist_bot = stored_dist_bot[k][i]
                        #dist_ul_bot = stored_dist_y_ul[k][i]
                        #dist_ur_bot = stored_dist_y_ur[k][i]
                        dist_lr_bot = stored_dist_y_lr[k][i]
                        dist_ll_bot = stored_dist_y_ll[k][i]
                        for p in range(len(stored_dist_top)):
                            for j in range(len(stored_dist_top[p])):
                                #print(k,i,p,j)
                                top_point = dup_lat_final_cen[p][stored_index_top[p][j]]
                                top_lat = dup_lat_final[p][stored_index_top[p][j]]
                                dist_top = stored_dist_top[p][j]
                                dist_ul_top = stored_dist_y_ul[p][j]
                                dist_ur_top = stored_dist_y_ur[p][j]
                                #dist_lr_top = stored_dist_y_lr[p][j]
                                #dist_ll_top = stored_dist_y_ll[p][j]
                                min_c = min(dist_ul_top,dist_ur_top,dist_lr_bot,dist_ll_bot)
                                if(min_c<min_c_dist):
                                    min_c_dist = min_c
                                    min_c_dist_arr = [k,i,p,j]
                                    
                                
                                delta_lat = top_lat - bot_lat
                                delta_pix = bot_point[1] - top_point[1]
                                #print(top_lat,bot_lat,delta_lat,top_point,bot_point,delta_pix,dist_top,dist_bot)
                                meter_per_pix = 1e5*delta_lat/delta_pix
                                if(delta_pix > 0 and delta_lat > 0 and done==False):
                                    x = np.array([top_point[1],bot_point[1]],dtype=np.float64)
                                    y = np.array([top_lat,bot_lat],dtype=np.float64)
                                    #print(x,y)
                                    #print(top_left[1])
                                    #print(bot_left[1])
                                    #print('x = ',x)
                                    #print('y = ',y)
                                    #print(x.dtype,y.dtype)
                                    popt,pcov = curve_fit(lin_line,x,y)
                                    max_lat = lin_line(top_left[1],*popt)
                                    min_lat = lin_line(bot_left[1],*popt)
                                    #print(max_lat,min_lat)
                                    if(max_lat - min_lat < 2):
                                        #done=True
                                        top_point_cen = dup_lat_final_cen[p][stored_index_top[p][j]]
                                        top_point_lat = dup_lat_final[p][stored_index_top[p][j]]
                                        bot_point_cen = dup_lat_final_cen[k][stored_index_bot[k][i]]
                                        bot_point_lat = dup_lat_final[k][stored_index_bot[k][i]]
                                        delta_pix = bot_point[1] - top_point_cen[1] 
                                        total = dist_top + dist_bot
                                        bot_dist_corner = min(dist_ul_top,dist_ur_top)
                                        top_dist_corner = min(dist_lr_bot,dist_ll_bot)
                                        top_max = max(dist_top,top_dist_corner)
                                        bot_max = max(dist_bot,bot_dist_corner)
                                        total = top_max+bot_max
                                        print(total,top_point_lat,bot_point_lat,top_point_cen,bot_point_cen,top_max,bot_max)
                                        if(total<min_dist_sum):
                                            min_dist_sum = total
                                            arr = [k,i,p,j]
                                        if(dist_top<min_dist_top):
                                            min_dist_top = dist_top
                                        if(dist_bot<min_dist_bot):
                                            min_dist_bot = dist_bot
                                        delta_lat = top_point_lat - bot_point_lat 
                                        meter_per_pix = 1e5*delta_lat/delta_pix
                                        #print(top_point,bot_point)

                """
                print(min_dist_sum)        
                print(min_dist_top,arr) 
                print(min_dist_bot,arr) 
                print(dup_lat_final_cen[arr[2]][stored_index_top[arr[2]][arr[3]]])
                print(dup_lat_final_cen[arr[0]][stored_index_bot[arr[0]][arr[1]]])
                print(dup_lat_final[arr[2]][stored_index_top[arr[2]][arr[3]]])
                print(dup_lat_final[arr[0]][stored_index_bot[arr[0]][arr[1]]])
                """
                if(total<1000):
                    lat3d = np.zeros((3,2))
                    lat3d[0,0] = dup_lat_final_cen[arr[2]][stored_index_top[arr[2]][arr[3]]][1] 
                    lat3d[1,0] =dup_lat_final_cen[arr[2]][stored_index_top[arr[2]][arr[3]]][0] 
                    lat3d[2,0] = dup_lat_final[arr[2]][stored_index_top[arr[2]][arr[3]]]

                    lat3d[0,1] = dup_lat_final_cen[arr[0]][stored_index_bot[arr[0]][arr[1]]][1]
                    lat3d[1,1] = dup_lat_final_cen[arr[0]][stored_index_bot[arr[0]][arr[1]]][0]
                    lat3d[2,1] = dup_lat_final[arr[0]][stored_index_bot[arr[0]][arr[1]]]
                elif(total>1000 and len(lat)>0 ):
                    # no good pair
                    # need to sort the lats by didstance to pick best match to a corner
                    lat3d = np.zeros((3,1))
                    lat3d[0,0] = dup_lat_final_cen[min_c_dist_arr[2]][stored_index_top[min_c_dist_arr[2]][min_c_dist_arr[3]]][1] 
                    lat3d[1,0] =dup_lat_final_cen[min_c_dist_arr[2]][stored_index_top[min_c_dist_arr[2]][min_c_dist_arr[3]]][0] 
                    lat3d[2,0] = dup_lat_final[min_c_dist_arr[2]][stored_index_top[min_c_dist_arr[2]][min_c_dist_arr[3]]]
                elif(len(dist_y_bot)<2 and len(lat)==0):
                    lat3d = np.zeros((3,1))

            print('lat 3',lat3d)
            #"""    

            # trivial solution, only two lats detected, although in different locations
            # should calculate dist in pixels / delta lat and get approximate lat range in between bounds
            # should be less than about 1.5 degrees as maps are not any larger...
            # there are only 1-2 points in each
            #print(lon,clon)
            print('distxright ',len(dist_x_right))
            """
            if(len(dist_x_right)==2):
                for i in range(len(stored_dist_right[0])):
                    right_point = dup_lon_final_cen[0][stored_index_right[0][i]]
                    right_lon = dup_lon_final[0][stored_index_right[0][i]]
                    #print(bottom_point,bot_lat)
                    for j in range(len(stored_dist_left[1])):
                        left_point = dup_lon_final_cen[1][stored_index_left[1][j]]
                        left_lon = dup_lon_final[1][stored_index_left[1][j]]
                        delta_lon = right_lon - left_lon
                        delta_pix = right_point[1] - left_point[1]
                        #print(top_lat,bot_lat,delta_lat,top_point,bot_point,delta_pix)

                left_point_cen = dup_lon_final_cen[1][stored_index_left[1][0]]
                left_point_lon = dup_lon_final[1][stored_index_left[1][0]]
                right_point_cen = dup_lon_final_cen[0][stored_index_right[0][0]]
                right_point_lon = dup_lon_final[0][stored_index_right[0][0]]
                delta_pix = right_point[0] - left_point_cen[0] 
                delta_lon = right_point_lon - left_point_lon 
                meter_per_pix = 1e5*delta_lon/delta_pix
                #print(delta_pix,delta_lon,meter_per_pix)
                #print(right_point,right_point_lon)
                #print(left_point,left_point_lon)
                lon3d = np.zeros((3,2))
                lon3d[0,0] = right_point[1] ; lon3d[0,1] = left_point[1]
                lon3d[1,0] = right_point[0] ; lon3d[1,1] = left_point[0]
                lon3d[2,0] = right_point_lon; lon3d[2,1] = left_point_lon  

            """
            done=False
            min_dist_left = 1e6
            min_dist_right= 1e6
            min_dist_sum = 1e6

            min_dist_ul = 1e6
            min_dist_ur = 1e6
            min_dist_lr = 1e6
            min_dist_ll = 1e6
            
            min_c_dist = 1e6
            totel=1e6
            min_c_dist_arr = []


            if(len(dist_x_right)>=2 and done==False):
                for k in range(len(stored_dist_right)):
                    for i in range(len(stored_dist_right[k])):
                        right_point = dup_lon_final_cen[k][stored_index_right[k][i]]
                        right_lon = dup_lon_final[k][stored_index_right[k][i]]
                        dist_right = stored_dist_right[k][i]
                        #dist_ul_right = stored_dist_x_ul[k][i]
                        dist_ur_right = stored_dist_x_ur[k][i]
                        dist_lr_right = stored_dist_x_lr[k][i]
                        #print(right_point,right_lon,dist_ur_right)
                        #dist_ll_right = stored_dist_x_ll[k][i]
                        for p in range(len(stored_dist_left)):
                            for j in range(len(stored_dist_left[p])):
                                #print(k,i,p,j)
                                left_point = dup_lon_final_cen[p][stored_index_left[p][j]]
                                left_lon = dup_lon_final[p][stored_index_left[p][j]]
                                dist_left = stored_dist_left[p][j]
                                dist_ul_left = stored_dist_x_ul[p][j]
                                #dist_ur_left = stored_dist_x_ur[p][j]
                                #dist_lr_left = stored_dist_x_lr[p][j]
                                dist_ll_left = stored_dist_x_ll[p][j]
                                min_c = min(dist_ul_left,dist_ur_right,dist_lr_right,dist_ll_left)
                                if(min_c<min_c_dist):
                                    min_c_dist = min_c
                                    min_c_dist_arr = [k,i,p,j]
                                delta_lon = right_lon - left_lon
                                delta_pix = right_point[0] - left_point[0]
                                #print(right_lon,left_lon,delta_lon,right_point,left_point,delta_pix)
                                meter_per_pix = 1e5*delta_lon/delta_pix
                                if(delta_pix > 0 and delta_lon > 0 and done==False):
                                    x = np.array([left_point[0],right_point[0]],dtype=int)
                                    y = np.array([left_lon,right_lon],dtype=np.float64)
                                    #print(x,y)
                                    #print(top_left[1])
                                    #print(bot_left[1])
                                    popt,pcov = curve_fit(lin_line,x,y)
                                    max_lon = lin_line(top_left[0],*popt)
                                    min_lon = lin_line(top_right[0],*popt)
                                    #print(max_lat,min_lat)
                                    if(max_lon - min_lon < 4):
                                        #done=True
                                        left_point_cen = dup_lon_final_cen[p][stored_index_left[p][j]]
                                        left_point_lon = dup_lon_final[p][stored_index_left[p][j]]
                                        right_point_cen = dup_lon_final_cen[k][stored_index_right[k][i]]
                                        right_point_lon = dup_lon_final[k][stored_index_right[k][i]]
                                        delta_pix = right_point[1] - left_point_cen[1] 

                                        total = dist_right + dist_left
                                        left_dist_corner = min(dist_ul_left,dist_ll_left)
                                        right_dist_corner = min(dist_ur_right,dist_lr_right)
                                        left_max = max(dist_left,left_dist_corner)
                                        right_max = max(dist_right,right_dist_corner)
                                        total = left_max + right_max
                                        print(total,left_point_lon,right_point_lon,left_point_cen,right_point_cen,left_max,right_max)
                                        if(total<min_dist_sum):
                                            min_dist_sum = total
                                            arr = [k,i,p,j]
                                        if(dist_left<min_dist_left):
                                            min_dist_left = dist_left
                                        if(dist_right<min_dist_right):
                                            min_dist_right = dist_right
                                        delta_lon = right_point_lon - left_point_lon 
                                        meter_per_pix = 1e5*delta_lon/delta_pix
                                        #print(meter_per_pix,max_lat,min_lat)
                """
                print(min_dist_sum)        
                print(min_dist_right,arr) 
                print(min_dist_left,arr) 
                print(dup_lon_final_cen[arr[2]][stored_index_left[arr[2]][arr[3]]])
                print(dup_lon_final_cen[arr[0]][stored_index_right[arr[0]][arr[1]]])
                print(dup_lon_final[arr[2]][stored_index_left[arr[2]][arr[3]]])
                print(dup_lon_final[arr[0]][stored_index_right[arr[0]][arr[1]]])
                """
                if(total<1000):
                    lon3d = np.zeros((3,2))
                    lon3d[0,0] = dup_lon_final_cen[arr[2]][stored_index_left[arr[2]][arr[3]]][1] 
                    lon3d[1,0] =dup_lon_final_cen[arr[2]][stored_index_left[arr[2]][arr[3]]][0] 
                    lon3d[2,0] = dup_lon_final[arr[2]][stored_index_left[arr[2]][arr[3]]]

                    lon3d[0,1] = dup_lon_final_cen[arr[0]][stored_index_right[arr[0]][arr[1]]][1]
                    lon3d[1,1] = dup_lon_final_cen[arr[0]][stored_index_right[arr[0]][arr[1]]][0]
                    lon3d[2,1] = dup_lon_final[arr[0]][stored_index_right[arr[0]][arr[1]]]  

                elif(toal>1000 and len(lon)>0):
                    lon3d = np.zeros((3,1))
                    lon3d[0,0] = dup_lon_final_cen[min_c_dist_arr[2]][stored_index_left[min_c_dist_arr[2]][min_c_dist_arr[3]]][1] 
                    lon3d[1,0] =dup_lon_final_cen[min_c_dist_arr[2]][stored_index_left[min_c_dist_arr[2]][min_c_dist_arr[3]]][0] 
                    lon3d[2,0] = dup_lon_final[min_c_dist_arr[2]][stored_index_left[min_c_dist_arr[2]][min_c_dist_arr[3]]]
                elif(len(dist_x_right)<2 and len(lon)==0):
                    lon3d=np.zeros((3,1))

            print('lon 3',lon3d)

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
                    if(lon_max - lon_min > 2):
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
                    if(lat_max - lat_min > 2):
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
                if(lon_max - lon_min > 2):
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
                    if(lon_max - lon_min > 2):
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

            print('Fit: xown,yown = ',fitx_own,', ',fity_own)
            print('Fit: xhelp,yhelp = ',fitx_help,', ',fity_help)
            print('Fit: xglobal,yglobal = ',fitx_global,', ',fity_global)
            
            real_res = os.path.join(image_dir,image_path.split('.tif')[0]+'.csv')
            df = pd.read_csv(real_res)
            row_test = df['row'].values
            col_test = df['col'].values
            #row_lat = df['NAD83_y'].values
            #col_lon = df['NAD83_x'].values
            npts = len(row_test)

            if(fitx_own == True or fitx_help==True):
                if(fity_own==True or fity_help==True):
                    # fit x and y correctly
                    calc_lon = np.zeros(npts)
                    calc_lat = np.zeros(npts)
                    for i in range(npts):
                        #meas_lon = col_lon[i]
                        calc_lon[i] = lin_line(col_test[i],*poptx)
                        #meas_lat = row_lat[i]
                        calc_lat[i] = lin_line(row_test[i],*popty)
                        #diff_lat = meas_lat - calc_lat[i]
                        #diff_lon = meas_lon - calc_lon[i]

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 2):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 2):
                            calc_lon[i] = clue_x


                    #np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_lon,row_lat,calc_lon,calc_lat]).T,\
                    #          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
                    np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_test,row_test,calc_lon,calc_lat]).T,\
                              fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')

                else:
                    #only fitted x correctly, not y
                    calc_lon = np.zeros(npts)
                    calc_lat = np.zeros(npts)
                    for i in range(npts):
                        #meas_lon = col_lon[i]
                        calc_lon[i] = lin_line(col_test[i],*poptx)
                        #meas_lat = row_lat[i]
                        calc_lat[i] = clue_y
                        #diff_lat = meas_lat - calc_lat[i]

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 2):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 2):
                            calc_lon[i] = clue_x

                    #np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_lon,row_lat,calc_lon,calc_lat]).T,\
                    #          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
                    np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_test,row_test,calc_lon,calc_lat]).T,\
                          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')

            elif(fitx_own == False and fitx_help==False):
                #failed to fit x, but did y
                if(fity_own==True or fity_help==True):
                    calc_lon = np.zeros(npts)
                    calc_lat = np.zeros(npts)
                    for i in range(npts):
                        #meas_lon = col_lon[i]
                        calc_lon[i] = clue_x
                        #meas_lat = row_lat[i]
                        calc_lat[i] = lin_line(row_test[i],*popty)
                        #diff_lat = meas_lat - calc_lat[i]

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 2):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 2):
                            calc_lon[i] = clue_x

                    #np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_lon,row_lat,calc_lon,calc_lat]).T,\
                    #          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
                    np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_test,row_test,calc_lon,calc_lat]).T,\
                          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')

                else:
                    # did not fit x or y
                    calc_lon = np.zeros(npts)
                    calc_lat = np.zeros(npts)
                    for i in range(npts):
                        #meas_lon = col_lon[i]
                        calc_lon[i] = clue_x
                        #meas_lat = row_lat[i]
                        calc_lat[i] = clue_y
                        #diff_lat = meas_lat - calc_lat[i]

                    #np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_lon,row_lat,calc_lon,calc_lat]).T,\
                    #          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
                    np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_test,row_test,calc_lon,calc_lat]).T,\
                          fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
            test_routine = False
            """
            #except Exception as e:
            print(f"Exception {e}: File = ",image_path)
                #Get clue
            clue_file = os.path.join(clue_dir,image_path.split('.tif')[0]+'_clue.csv')


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
                return

            real_res = os.path.join(image_dir,image_path.split('.tif')[0]+'.csv')
            df = pd.read_csv(real_res)
            row_test = df['row'].values
            col_test = df['col'].values
            #row_lat = df['NAD83_y'].values
            #col_lon = df['NAD83_x'].values
            npts = len(row_test)

            calc_lon = np.zeros(npts)
            calc_lat = np.zeros(npts)
            for i in range(npts):
                #meas_lon = col_lon[i]
                calc_lon[i] = clue_x
                #meas_lat = row_lat[i]
                calc_lat[i] = clue_y
                #diff_lat = meas_lat - calc_lat[i]

            #np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_lon,row_lat,calc_lon,calc_lat]).T,\
            #              fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
            np.savetxt(os.path.join(out_dir,image_path.split('.tif')[0]+'.csv'),np.array([col_test,row_test,calc_lon,calc_lat]).T,\
                      fmt = '%.7f,%.7f,%.7f,%.7f',delimiter=',')
            """
    print('Time = ',time.time()-tz)
    return

if __name__=="__main__":
    out_dir = sys.argv[1]
    image_dir = sys.argv[2]
    image_path = sys.argv[3]
    clue_dir = sys.argv[4]
    
    print(out_dir,image_dir,image_path,clue_dir)
    
    main(image_dir,image_path,out_dir,clue_dir)
    

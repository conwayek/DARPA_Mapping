"""
Written by:
Dr. Eamon K. Conway
Geospatial Development Center (GDC)
Kostas Research Institute for Homeland Security
Northeastern University

Contact:
e.conway@northeastern.edu

Date:
9/19/2022

DARPA Critical Mineral Challenge 2022

Purpose:
Writes the final file of results. 
It performs the linear fits and ensures lon/lat fits are sensible

Args:
best pair of lon
best pair of lat
whether we are training or not
image shape
output directory
image name
path to image directory
clue lon
clue lat
re-do: whether this is a first or second attempt
all lon points
all lat points
all lon pixel coordinates
all lat pixel coordinates

out:
a csv file with georeferenced results in it.

"""
import math
import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

def lin_line(x, A, B): 
    return A*x + B


def main(lat3d,lon3d,training,img_shape,out_dir,image_path,image_dir,clue_x,clue_y,redo,lons,lats,clons,clats):
            if(redo==True):
                print('Second attempt......')
            else:
                print('First attempt......')
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
                    X = np.linspace(0,img_shape[1]-1,img_shape[1])
                    Y = lin_line(X,*poptx)
                    lon_max = Y[-1]
                    lon_min = Y[0]
                    deltax = 1e5 * (lon_max - lon_min) /img_shape[1] 
                    print('delta x',deltax)
                    print('lon max/min own = ',lon_max,lon_min)
                    """
                    print(lon3d[1,:])
                    print(lon3d[1,0],lon3d[1,1])
                    print(lon3d[2,:])
                    X = np.linspace(0,int((lon3d[1,1]-lon3d[1,0]))-1,int((lon3d[1,1]-lon3d[1,0])))
                    Y = lin_line(X,*poptx)
                    lon_max = Y[-1]
                    lon_min = Y[0]
 

                    deltax = 1e5 * (lon_max - lon_min) /img_shape[1] 
                    print('delta x',deltax)
                    print('Delta Lon = ',lon_max-lon_min)
                    """
                    if(lon_max - lon_min > 3.5):
                        fitx_own=False


            if(lat3d.shape[1] > 1):
                if(lat3d[2,0] != lat3d[2,1]):
                    fity_own=True
                    popty,pcovy = curve_fit(lin_line,lat3d[0,:],lat3d[2,:])
                    X = np.linspace(0,img_shape[0]-1,img_shape[0])
                    Y = lin_line(X,*popty)
                    
                    lat_min = Y[-1]
                    lat_max = Y[0]
                    print('lat max/min own = ',lat_max,lat_min)
                    deltay = 1e5 * (lat_max - lat_min) /img_shape[0] 
                    print('delta y',deltay)
                    print('Delta Lat = ',lat_max-lat_min)
                    if(lat_max - lat_min > 3.5):
                        fity_own=False

            
            if(fity_own == True and fitx_own == True):
                print('Checking Fits against each other') 
                # meter per pixel in each direction
                deltay1 = 1e5 * (lat_max - lat_min) /img_shape[0] 
                deltax1 = 1e5 * (lon_max - lon_min) /img_shape[1] 
                print('meter per pixel y,x = ',deltay1,deltax1)
                print(abs(deltay1/deltax1) ,abs(lat_max - lat_min)/abs(lon_max - lon_min)  )
                if(abs(deltax1/deltay1)>1.2 or abs(deltax1/deltay1)<0.8 or abs(deltax1)<1.5 or abs(deltay1)<1.5 \
                  or abs(deltax1)>25 or abs(deltay1)>25):
                    if( math.isclose((abs(lat_max - lat_min)/abs(lon_max - lon_min)  ),abs(deltay1/deltax1)  ,\
                                     abs_tol=0.25*min(abs(deltay1/deltax1),(abs(lat_max - lat_min)/abs(lon_max - lon_min)  )) )==False \
                       or abs(abs(lat_max - lat_min)/abs(lon_max - lon_min)  ) < 0.3 or abs(abs(lat_max - lat_min)/abs(lon_max - lon_min)  ) > 3):
                        print('Large dispersion.....checking x points against y fit')
                        # try fit x points using y fit
                        dist = np.zeros(2)
                        dist[0] = abs(abs(clue_x) - abs(lon3d[2,0]))
                        dist[1] = abs(abs(clue_x) - abs(lon3d[2,1]))
                        poptx1=popty.copy()
                        poptx1[0] = -poptx1[0]
                        # pick best x point i.e. the one closest to our key point
                        if(dist[0]<dist[1]):
                            print('Best Lon = ',lon3d[2,0]) 
                            c = lon3d[2,0] - poptx1[0]*lon3d[1,0]
                            poptx1[1] = c
                        else:
                            print('Best Lon = ',lon3d[2,1]) 
                            c = lon3d[2,1] - poptx1[0]*lon3d[1,1]
                            poptx1[1] = c
                        X = np.linspace(0,img_shape[1]-1,img_shape[1])
                        Y = lin_line(X,*poptx1)

                        lon_max = Y[-1]
                        lon_min = Y[0]
                        deltax2 = 1e5 * (lon_max - lon_min) /img_shape[1] 

                        if(abs(deltay1 / deltax2) < 1.2 and abs(deltay1/deltax2)>0.8 and (lon_max-lon_min)<3 \
                          and abs(deltax2)>1.5 and abs(deltax2)<25):
                            fitx2=True
                        else: 
                            fitx2=False
                        print('Lon point check with Y fit: ',fitx2)
                        print('Delta y1, x2 = ',deltay1,deltax2) 


                        print('Large dispersion.....checking y points against X fit')
                        # we have 2 numbers, need to pick closest to clue y
                        dist = np.zeros(2)
                        dist[0] = abs(abs(clue_y) - abs(lat3d[2,0]))
                        dist[1] = abs(abs(clue_y) - abs(lat3d[2,1]))
                        if(dist[0]<dist[1]):
                            print('Best Lat = ',lat3d[2,0]) 
                            popty1=poptx.copy()
                            popty1[0] = -popty1[0]
                            c = lat3d[2,0] - popty1[0]*lat3d[0,0]
                            popty1[1] = c
                        else:
                            print('Best Lat = ',lat3d[2,1]) 
                            popty1=poptx.copy()
                            popty1[0] = -popty1[0]
                            c = lat3d[2,1] - popty1[0]*lat3d[0,1]
                            popty1[1] = c
                        X = np.linspace(0,img_shape[0]-1,img_shape[0])
                        Y = lin_line(X,*popty1)

                        lat_min = Y[-1]
                        lat_max = Y[0]
                        deltay2 = 1e5 * (lat_max - lat_min) /img_shape[0] 
                        if(abs(deltax1 / deltay2) < 1.2 and abs(deltax1 / deltay2) > 0.8 and (lat_max-lat_min)<3 \
                          and abs(deltay2)>1.5 and abs(deltay2)<25):
                            fity2=True
                        else: 
                            fity2=False
                        print('Lat point check with X fit: ',fity2)
                        print('Delta y2, x1 = ',deltay2,deltax1) 

                        if(fitx2 == True and fity2==True): 
                            # use x1 and y2 fits
                            if(abs(deltax1 - deltay2) < abs(deltay1 - deltax2)):
                                print('Using X fit with y points')
                                fity_own = False
                            # use y1 and x2 fits
                            else:
                                print('Using Y fit with x points')
                                fitx_own = False
                        elif(fity2 == False and fitx2==True):
                            print('Using Y fit with x points')
                            fitx_own = False 
                        elif(fitx2 == False and fity2==True):
                            print('Using X fit with y points')
                            fity_own = False 

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
                X = np.linspace(0,img_shape[1]-1,img_shape[1])
                Y = lin_line(X,*poptx)
                    
                lon_max = Y[-1]
                lon_min = Y[0]
                print('lon max/min help = ',lon_max,lon_min)
                if(lon_max - lon_min > 3.5):
                    fitx_help=False
                    fitx_global=True
                    
            elif(fitx_own==False and fity_own==True and lon3d.shape[1]==1):
                if(lon3d[2,0]!=0):
                    
                    # Can we ensure that lon3d[2,0] is close to lat3d[1,0] or lat3d[1,1]
                    # if very close to right side, then ensure keypoint is close or to the left
                    # close meaning fractional degrees
                    # if close to the left, ensure clue is to the right
                    # is lon/lat not close, search for best match to dup_lons....
                    lat_right = max(lat3d[1,0],lat3d[1,1])
                    lat_left = min(lat3d[1,0],lat3d[1,1])
                    print(lat_left,lat_right)
                    print(lons)
                    print(clons)
                    nlon_dup = len(lons)
                    dif_lon_left = len(lons)
                    dif_lon_right = len(lons)
                    
                    dif_lon_left = clons[:,0] - lat_left
                    dif_lon_right = clons[:,0] - lat_right
                    
                    done=False
                    
                    #left side
                    idx = np.argsort(abs(dif_lon_left))
                    dif_lon_left = dif_lon_left[idx]
                    lons = lons[idx]
                    clons = clons[idx,:]
                    print(idx)
                    print(lons)
                    print(dif_lon_left)
                    # only take if dif_lon_left<100 and clue_x to right
                    dif_lon_left = dif_lon_left[abs(dif_lon_left)<100]
                    if(len(dif_lon_left)>=1):
                        idy = np.where(lons<=clue_x)[0]
                        if(len(idy)>=1 and done==False):
                            lon3d = np.zeros((3,1))
                            lon3d[0,0] = clons[idy[0],1]
                            lon3d[1,0] = clons[idy[0],0]
                            lon3d[2,0] = lons[idy[0]]
                            print('New Lon3d Left ', lon3d)
                            done=True

                    #right side
                    idx = np.argsort(abs(dif_lon_right))
                    dif_lon_right = dif_lon_right[idx]
                    lons = lons[idx]
                    clons = clons[idx,:]
                    # only take if dif_lon_right<100 and clue_x to left
                    dif_lon_right = dif_lon_right[abs(dif_lon_right)<100]
                    if(len(dif_lon_right)>=1 and done==False):
                        idy = np.where(lons>=clue_x)[0]
                        if(len(idy)>=1 and done==False):
                            lon3d = np.zeros((3,1))
                            lon3d[0,0] = clons[idy[0],1]
                            lon3d[1,0] = clons[idy[0],0]
                            lon3d[2,0] = lons[idy[0]]
                            print('New Lon3d Right ', lon3d)
                            done=True
                    
                    poptx=popty.copy()
                    poptx[0] = -poptx[0]
                    c = lon3d[2,0] - poptx[0]*lon3d[1,0]
                    poptx[1] = c
                    fitx_help=True
                    X = np.linspace(0,img_shape[1]-1,img_shape[1])
                    Y = lin_line(X,*poptx)

                    lon_max = Y[-1]
                    lon_min = Y[0]
                    print('lon max/min help = ',lon_max,lon_min)
                    if(lon_max - lon_min > 3.5):
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
                X = np.linspace(0,img_shape[0]-1,img_shape[0])
                Y = lin_line(X,*popty)
                    
                lat_min = Y[-1]
                lat_max = Y[0]
                print('lat max/min help = ',lat_max,lat_min)
                if(lat_max - lat_min > 3.5):
                        fity_help=False
                        fity_global=True
            elif(fity_own==False and fitx_own==True and lat3d.shape[1]==1):
                if(lat3d[2,0]!=0):
                    
                    # Can we ensure that lon3d[2,0] is close to lat3d[1,0] or lat3d[1,1]
                    # if very close to right side, then ensure keypoint is close or to the left
                    # close meaning fractional degrees
                    # if close to the left, ensure clue is to the right
                    # is lon/lat not close, search for best match to dup_lons....
                    lon_up = max(lon3d[0,0],lon3d[0,1])
                    lon_down = min(lon3d[0,0],lon3d[0,1])
                    print(lon_up,lon_down)
                    print(lats)
                    print(clats)
                    nlat_dup = len(lats)
                    dif_lat_up = len(lats)
                    dif_lat_down = len(lats)
                    
                    dif_lat_up = clats[:,1] - lon_up
                    dif_lat_down = clats[:,1] - lon_down
                    
                    done=False
                    
                    #upper side
                    idy = np.argsort(abs(dif_lat_up))
                    dif_lat_up = dif_lat_up[idy]
                    lats = lats[idy]
                    clats = clats[idy,:]
                    print(idy)
                    print(lats)
                    print(dif_lat_up)
                    # only take if dif_lat_up<100 and clue_y below
                    dif_lat_up = dif_lat_up[abs(dif_lat_up)<100]
                    if(len(dif_lat_up)>=1):
                        idy = np.where(lats>=clue_y)[0]
                        if(len(idy)>=1 and done==False):
                            lat3d = np.zeros((3,1))
                            lat3d[0,0] = clats[idy[0],1]
                            lat3d[1,0] = clats[idy[0],0]
                            lat3d[2,0] = lats[idy[0]]
                            print('New Lat3d Left ', lat3d)
                            done=True

                    #right side
                    idx = np.argsort(abs(dif_lat_down))
                    dif_lat_down = dif_lat_down[idx]
                    lats = lats[idx]
                    clats = clats[idx,:]
                    # only take if dif_lat_down<100 and clue_y above
                    dif_lat_down = dif_lat_down[abs(dif_lat_down)<100]
                    if(len(dif_lat_down)>=1 and done==False):
                        idy = np.where(lats<=clue_y)[0]
                        if(len(idy)>=1 and done==False):
                            lat3d = np.zeros((3,1))
                            lat3d[0,0] = clats[idy[0],1]
                            lat3d[1,0] = clats[idy[0],0]
                            lat3d[2,0] = lats[idy[0]]
                            print('New Lat3d Right ', lat3d)
                            done=True
                    
                    popty=poptx.copy()
                    popty[0] = -popty[0]
                    c = lat3d[2,0] - popty[0]*lat3d[0,0]
                    popty[1] = c
                    fity_help=True
                    X = np.linspace(0,img_shape[0]-1,img_shape[0])
                    Y = lin_line(X,*popty)

                    lat_min = Y[-1]
                    lat_max = Y[0]
                    print('lat max/min help = ',lat_max,lat_min)
                    if(lat_max - lat_min > 3.5):
                        fity_help=False
                        fity_global=True                    

                    """
                    popty=poptx.copy()
                    popty[0] = -popty[0]
                    c = lat3d[2,0] - popty[0]*lat3d[0,0]
                    popty[1] = c
                    fity_help=True
                    X = np.linspace(0,img_shape[0]-1,img_shape[0])
                    Y = lin_line(X,*popty)         
                    lat_min = Y[-1]
                    lat_max = Y[0]
                    print('lat max/min help = ',lat_max,lat_min)
                    if(lat_max - lat_min > 3.5):
                        fity_help=False
                        fity_global=True
                    """
                        
            if(fitx_own == False and fitx_help==False):
                fitx_global = True
            if(fity_own == False and fity_help==False):
                fity_global = True

            print('Fit: xown,yown = ',fitx_own,', ',fity_own)
            print('Fit: xhelp,yhelp = ',fitx_help,', ',fity_help)
            print('Fit: xglobal,yglobal = ',fitx_global,', ',fity_global)

            if(redo == False):
                if(fitx_global==True or fity_global==True ):
                    redo=True
                    return redo   

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

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 3.5):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 3.5):
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
                            

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 3.5):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 3.5):
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

                        if(abs(abs(calc_lat[i]) - abs(clue_y) ) > 3.5):
                            calc_lat[i] = clue_y
                        if(abs(abs(calc_lon[i]) - abs(clue_x) ) > 3.5):
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
            return False 


import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

def lin_line(x, A, B): 
    return A*x + B


def main(lat3d,lon3d,training,img_shape,out_dir,image_path,image_dir,clue_x,clue_y,redo):
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
                    print('lon max/min own = ',lon_max,lon_min)
                    if(lon_max - lon_min > 3):
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
                    if(lat_max - lat_min > 3):
                        fity_own=False

            print('Checking Fits against each other') 
            if(fity_own == True and fitx_own == True):
                # meter per pixel in each direction
                deltay1 = 1e5 * (lat_max - lat_min) /img_shape[0] 
                deltax1 = 1e5 * (lon_max - lon_min) /img_shape[1] 
                print('meter per pixel y,x = ',deltay1,deltax1)
                if(abs(deltax1/deltay1)>2 or abs(deltax1/deltay1)<0.5):
                    print('Large dispersion.....checking x points against y fit')
                    # try fit x points using y fit
                    dist = np.zeros(2)
                    dist[0] = abs(abs(clue_x) - abs(lon3d[2,0]))
                    dist[1] = abs(abs(clue_x) - abs(lon3d[2,1]))
                    poptx1=popty.copy()
                    poptx1[0] = -poptx1[0]
                    # pick best x point i.e. the one closest to our key point
                    if(dist[0]<dist[1]):
                        c = lon3d[2,0] - poptx1[0]*lon3d[1,0]
                        poptx1[1] = c
                    else:
                        c = lon3d[2,1] - poptx1[0]*lon3d[1,1]
                        poptx1[1] = c
                    X = np.linspace(0,img_shape[1]-1,img_shape[1])
                    Y = lin_line(X,*poptx1)
                        
                    lon_max = Y[-1]
                    lon_min = Y[0]
                    deltax2 = 1e5 * (lon_max - lon_min) /img_shape[1] 
                    
                    if(abs(deltay1 / deltax2) < 2 and abs(deltay1/deltax2)>0.5 and (lon_max-lon_min)<3):
                        fitx2=True
                    else: 
                        fitx2=False
                    print('Lon point check with Y fit: ',fitx2)
                   
                    print('Large dispersion.....checking y points against X fit')
                    # we have 2 numbers, need to pick closest to clue y
                    dist = np.zeros(2)
                    dist[0] = abs(abs(clue_y) - abs(lat3d[2,0]))
                    dist[1] = abs(abs(clue_y) - abs(lat3d[2,1]))
                    if(dist[0]<dist[1]):
                        popty1=poptx.copy()
                        popty1[0] = -popty1[0]
                        c = lat3d[2,0] - popty1[0]*lat3d[0,0]
                        popty1[1] = c
                    else:
                        popty1=poptx.copy()
                        popty1[0] = -popty1[0]
                        c = lat3d[2,1] - popty1[0]*lat3d[0,1]
                        popty1[1] = c
                    X = np.linspace(0,img_shape[0]-1,img_shape[0])
                    Y = lin_line(X,*popty1)
                        
                    lat_min = Y[-1]
                    lat_max = Y[0]
                    deltay2 = 1e5 * (lon_max - lon_min) /img_shape[1] 
                    if(abs(deltax1 / deltay2) < 2 and abs(deltax1 / deltay2) > 0.5 and (lat_max-lat_min)<3):
                        fity2=True
                    else: 
                        fity2=False
                    print('Lat point check with X fit: ',fity2)

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
                        fitx_own = False 
                    elif(fitx2 == False and fity2==True):
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
                    X = np.linspace(0,img_shape[1]-1,img_shape[1])
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
                X = np.linspace(0,img_shape[0]-1,img_shape[0])
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
                    X = np.linspace(0,img_shape[0]-1,img_shape[0])
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
            return False 

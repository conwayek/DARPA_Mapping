from geopy.distance import distance
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


files = glob.glob('/scratch/e.conway/DARPA_MAPS/Results/*.csv')
print('Len Files = ',len(files))
meas_lon = []
meas_lat = []
calc_lon = []
calc_lat = []
name = []
rmse = []

for file in files:
    if('old' not in file):
        data = np.genfromtxt(file,delimiter=',')
        npts = data.shape[0]
        lon_m = []
        lat_m = []
        lon_c = []
        lat_c = []
        err_tot = []
        err = []
        f = []
        #if(file!='/scratch/e.conway/DARPA_MAPS/Results/GEO_0048.csv' and file!='/scratch/e.conway/DARPA_MAPS/Results/GEO_0047.csv'\\\n
        #  and file!='/scratch/e.conway/DARPA_MAPS/Results/GEO_0031.csv'):\n
        for i in range(npts):
                lon_m=data[i,3]
                lat_m=data[i,2]
                lon_c=data[i,5]
                lat_c=data[i,4]
                e = distance((lat_m,lon_m),(lat_c,lon_c)).m
                #print(e)
                err.append(e)
                #if(e>1e5):
                #    print(file)
        err = np.array(err,dtype=np.float64)
        #print(file,err)
        
        e = np.sqrt(np.sum(np.square(err)) / np.sum(np.isfinite(err)))
        print(file,e)
        
        
        rmse.append(e)

print(rmse)
rmse = np.array(rmse,dtype=np.float64)
print('Median RMSE [m] = ',np.nanmedian(rmse))
                
                
                
                
"""
        f.append(os.path.basename(file).split('.csv')[0].split('GEO_')[1])
        meas_lon.append(lon_m)
        meas_lat.append(lat_m)
        calc_lon.append(lon_c)
        calc_lat.append(lat_c)
        name.append(f)


nfile = len(name)
mlon_tot = []
mlat_tot = []
clon_tot = []
clat_tot = []
names = []
bad = [] 
bad_coord = []
for i in range(nfile):
        #if( '0086' not in name[i] and '0104' not in name[i]):
        for j in range(len(calc_lon[i])):
            mlon_tot.append(meas_lon[i][j])
            mlat_tot.append(meas_lat[i][j])
            clon_tot.append(calc_lon[i][j])
            clat_tot.append(calc_lat[i][j])
            names.append(name[i])
            if(1e5*abs(meas_lon[i][j] - calc_lon[i][j]) > 3e5):
                if(name[i] not in bad):
                    bad.append(name[i])
                    bad_coord.append(meas_lon[i][j])
            if(1e5*abs(meas_lat[i][j] - calc_lat[i][j]) > 3e5):
                if(name[i] not in bad):
                    bad.append(name[i])
                    bad_coord.append(meas_lat[i][j])
mlon_tot = np.array(mlon_tot,dtype=np.float64)
mlat_tot = np.array(mlat_tot,dtype=np.float64)
clon_tot = np.array(clon_tot,dtype=np.float64)
clat_tot = np.array(clat_tot,dtype=np.float64)

print(mlat_tot)
print(clat_tot)



rmse=[]
for i in range(5):#len(mlon_tot)):
    #err = mean_squared_error([mlon_tot[i],mlat_tot[i]],[clon_tot[i],clat_tot[i]])*1e5
    print((mlat_tot[i],mlon_tot[i]),(clat_tot[i],mlon_tot[i]))
    err_lat = distance((mlat_tot[i],mlon_tot[i]),(clat_tot[i],mlon_tot[i])).m
    print(err_lat)
    err_lon = distance((mlat_tot[i],mlon_tot[i]),(mlat_tot[i],clon_tot[i])).m
    #print((mlat_tot[i],mlon_tot[i]),(mlat_tot[i],clon_tot[i]))
    err = err_lat#mean_squared_error([err_lat],[err_lon])
    #print(err_lon)
    rmse.append(err)
    if(err>1e5):
        print(names[i])

rmse = np.array(rmse,dtype=np.float64)
print('Median RMSE [m] = ',np.nanmedian(rmse))
"""


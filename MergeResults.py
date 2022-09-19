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
Fuse together all (non-)completed .csv files for DARPA submission

"""

from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import glob

files_clue = glob.glob('/scratch/e.conway/DARPA_MAPS/clue_CSVs/*.csv')
results = glob.glob('/scratch/e.conway/DARPA_MAPS/ValidationResults/*.csv')
eval_files = glob.glob('/scratch/e.conway/DARPA_MAPS/Validation/*.csv')

nclue = len(files_clue)
nres = len(results)
neval = len(eval_files)

filename = []
x = []
y = []
lon = []
lat = []
for i in tqdm(range(neval)):
    # read eval points
    f= pd.read_csv(eval_files[i])
    x1 = f['col'].values
    y1 = f['row'].values
    name = f['raster_ID'].values
    npts = len(x1)

    # current eval file from DARPA
    namenow = os.path.basename(eval_files[i])

    # let us get us get clue data
    found_clue = False
    while found_clue == False:
        for j in range(nclue):
            if(os.path.basename(files_clue[j]) == namenow.split('.csv')[0]+'_clue.csv'):
                f = pd.read_csv(files_clue[j])
                x_clue = f['NAD83_x'].values
                y_clue = f['NAD83_y'].values
                found_clue = True
    #get results for file
    found=False
    done=False
    while done==False:
        for j in range(nres):
            if(os.path.basename(results[j]) == namenow):
                try:
                    data = np.genfromtxt(results[j],delimiter=',')
                    ncalc = data.shape[0]
                    row_calc = data[:,0]
                    col_calc = data[:,1]
                    lat_calc = data[:,2]
                    lon_calc = data[:,3]
                    done=True
                    found=True
                except:
                    done=False
        if(done==False):
            done=True
    if(found==True):
        if(npts==ncalc):
            for j in range(npts):
                if(row_calc[j] == y1[j] and col_calc[j] == x1[j]):
                    filename.append(namenow.split('.csv')[0])
                    x.append(col_calc[j])
                    y.append(row_calc[j])
                    lon.append(lon_calc[j])
                    lat.append(lat_calc[j])
                else:
                    print('error: mix up on file write',namenow)
                    filename.append(namenow.split('.csv')[0])
                    x.append(col_calc[j])
                    y.append(row_calc[j])
                    lon.append(x_clue)
                    lat.append(y_clue)      
    else: 
        for j in range(npts):
                    filename.append(namenow.split('.csv')[0])
                    x.append(x1[j])
                    y.append(y1[j])
                    lon.append(x_clue)
                    lat.append(y_clue)




y = np.array(y,dtype=int)
x = np.array(x,dtype=int)
lat = np.array(lat,dtype=np.float64)
lon = np.array(lon,dtype=np.float64)
filename=np.array(filename,dtype=object)

arr = np.array([filename,y,x,lon,lat])
print(arr.T.shape)
df = pd.DataFrame(arr.T,index=None,columns=['raster_ID','row','col','NAD83_x','NAD83_y'])




df.to_csv('KRI_GDC_III.csv')


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
To tile an image into equal sized images

Args:
image
bounds(for main)

Out:
top left position of tile
bottom right position of tile
tile

"""

import numpy as np

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
######################################################################

def main(bounds,img):
            nres=750

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
            
            return tl,br,tile
            



if __name__=="__main__":
    image_dir = '/scratch/e.conway/DARPA_MAPS/Training/'
    image_path='GEO_0012.tif'
    file = os.path.join(image_dir,image_path)
    with rasterio.open(file,'r') as f:
        img = f.read()
    img = img.transpose((1,2,0))

    tl,br,tiles = tileall(img)

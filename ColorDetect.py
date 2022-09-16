import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import rasterio
import time

def main(image_dir,out_dir,image_path,width=0):

    #img = cv2.imread(os.path.join(image_dir,image_path))
    with rasterio.open(os.path.join(image_dir,image_path)) as f:
        img = f.read()
    print(img.shape)
    if(img.shape[0]==3):
        img = img.transpose((1,2,0))
    elif(img.shape[0]>3):
        img = img[0:3,:,:].transpose((1,2,0))
    elif(img.shape[0]==2):
        x =img[0,:,:]
        img = np.concatenate((img,x[np.newaxis,:,:]),axis=0)    
        img = img.transpose((1,2,0))
    else:
        x =img[0,:,:]
        img = np.concatenate((img,x[np.newaxis,:,:]),axis=0)    
        img = np.concatenate((img,x[np.newaxis,:,:]),axis=0)    
        img = img.transpose((1,2,0))
        x=None
    print(img.shape)
    img = np.array(img,dtype=np.float64)
    img[img<=2] = np.nan
    img[img==255] = np.nan
    
    print('Image Sum 1 = ',np.sum(np.isfinite(img)))
    ip = np.where(img[:,:,2].flatten()>250)[0]# = np.nan

    r=img[:,:,0].flatten()
    g=img[:,:,1].flatten()
    b=img[:,:,2].flatten()

    r[ip] = np.nan
    g[ip] = np.nan
    b[ip] = np.nan

    r = r.reshape(img.shape[0],img.shape[1])
    g = g.reshape(img.shape[0],img.shape[1])
    b = b.reshape(img.shape[0],img.shape[1])

    img = np.stack((r[:,:],g[:,:]),axis=2)
    img = np.concatenate((img,b[:,:,np.newaxis]),axis=2)
    
    print('Image Sum 2 = ',np.sum(np.isfinite(img)))
    
    contin=False
    return_fail = False

    while contin == False:
        width=width+200
        npix=width**2
        s1 = np.nansum(np.isfinite(img[0:width,0:width,0])) / npix
        s2 = np.nansum(np.isfinite(img[-width:,0:width,0])) / npix
        s3 = np.nansum(np.isfinite(img[0:width,-width:,0])) / npix
        s4 = np.nansum(np.isfinite(img[-width:,-width:,0])) / npix

        print('width,sums = ',width,s1,s2,s3,s4)
        if(s1>0.05 and s2>0.05 and s3>0.05 and s4>0.05):
            if(s1>0.15 or s2>0.15 or s3>0.15 or s4>0.15):
                if(np.isfinite(s1) and np.isfinite(s2) and np.isfinite(s3) and np.isfinite(s4)):
                    contin=True

        if(width>=max(0.3*img.shape[0],0.3*img.shape[1])):
            return_fail = True
            contin=True

    if(return_fail == True):
        return return_fail,np.zeros((img.shape[0],img.shape[1])),np.array([np.nan,np.nan,np.nan,np.nan])

    
    mean1 = np.nanmean(np.nanmean(img[0:width,0:width,:],axis=0),axis=0)
    mean2 = np.nanmean(np.nanmean(img[-width:,0:width,:],axis=0),axis=0)
    mean3 = np.nanmean(np.nanmean(img[0:width,-width:,:],axis=0),axis=0)
    mean4 = np.nanmean(np.nanmean(img[-width:,-width:,:],axis=0),axis=0)
    
    print('means = ',mean1,mean2,mean3,mean4)


    meanr = min(mean1[0],mean2[0],mean3[0],mean4[0])
    meang = min(mean1[1],mean2[1],mean3[1],mean4[1])
    meanb = min(mean1[2],mean2[2],mean3[2],mean4[2])


    #img=np.array(img,dtype=np.uint8)
    

    xsec_x = img[:,:,:]
    xsec_xr = xsec_x[:,:,0]
    xsec_xg = xsec_x[:,:,1]
    xsec_xb = xsec_x[:,:,2]

    
    
    ipr = np.logical_and(xsec_xr.flatten()<meanr-5,xsec_xr.flatten()>10)
    x = xsec_xr.flatten()
    x[:] = 0
    xsec_xr = x
    xsec_xr[ipr] = 1


    ipg = np.logical_and(xsec_xg.flatten()<meang-5,xsec_xg.flatten()>10)
    x = xsec_xg.flatten()
    x[:] = 0
    xsec_xg = x
    xsec_xg[ipg] = 1

    ipb = np.logical_and(xsec_xb.flatten()<meanb-5,xsec_xb.flatten()>10)
    x = xsec_xb.flatten()
    x[:] = 0
    xsec_xb = x
    xsec_xb[ipb] = 1
    
    s1 = np.sum(np.isfinite(np.where(xsec_xr>0))[0])
    s2 = np.sum(np.isfinite(np.where(xsec_xg>0))[0])
    s3 = np.sum(np.isfinite(np.where(xsec_xb>0))[0])
    npix = img.shape[0]*img.shape[1]

    print(s1,s2,s3,npix)

    if(s1 > 0.6*npix):
        xsec_xr[:] = 0
    if(s2 > 0.6*npix):
        xsec_xg[:] = 0
    if(s3 > 0.6*npix):
        xsec_xb[:] = 0
    
    
    xsec_xr = xsec_xr.reshape(img.shape[0],img.shape[1])
    xsec_xg = xsec_xg.reshape(img.shape[0],img.shape[1])
    xsec_xb = xsec_xb.reshape(img.shape[0],img.shape[1])

    img_test = np.stack((xsec_xr,xsec_xg),axis=2)
    img_test = np.concatenate((img_test,xsec_xb[...,np.newaxis]),axis=2)


    img2 = np.nansum(img_test,axis=2)

    x=img2.flatten()
    x[x>0.9] = 1
    #x[x<0.9] = 0
    img2 = x.reshape(img.shape[0],img.shape[1])
    
    fig=plt.figure()
    plt.imshow(img2,aspect='auto')
    plt.savefig('img2.png',dpi=400)
    plt.close()


    idx=[]
    for i in range(img.shape[0]):
        slc = img2[i,:]
        idx.append([np.where(slc==1)[0]])
        
    tz = time.time()
    
    idx_final_y = []
    for i in range(len(idx)):
        if(len(idx[i][0])>1):
            # run ransac per slice to find the linear increasing line
            ransac = RANSACRegressor(residual_threshold=400)
            pt=np.linspace(0,len(idx[i][0])-1,len(idx[i][0]))
            rf = ransac.fit(pt.reshape(-1,1),idx[i][0].reshape(-1,1))
            rf_mask = rf.inlier_mask_
            rf_out_mask = np.logical_not(rf_mask)
            #x = np.arange(np.min(idx[i][0][rf_mask]),np.max(idx[i][0][rf_mask]),1)
            idx_final_y.append(idx[i][0][rf_mask])
            #idx_final_y.append(x)
        else:
            idx_final_y.append([])
            
            
        
    img_mask = np.zeros((img.shape[0:2]))
    for i in range(len(idx_final_y)):
        img_mask[i,idx_final_y[i]] = 1
        #img[i,idx_final[i]] = 0
    """
    fig=plt.figure()
    plt.imshow(img_mask,aspect='auto')
    plt.savefig('post_ransacy.png',dpi=400)
    plt.close()
    """
    idx=[]
    for i in range(img_mask.shape[1]):
        slc = img_mask[:,i]
        idx.append([np.where(slc>0)[0]])
        
    idx_final_x = []
    for i in range(len(idx)):
        if(len(idx[i][0])>1):
            # run ransac per slice to find the linear increasing line
            ransac = RANSACRegressor(residual_threshold=500)
            pt=np.linspace(0,len(idx[i][0])-1,len(idx[i][0]))
            rf = ransac.fit(pt.reshape(-1,1),idx[i][0].reshape(-1,1))
            rf_mask = rf.inlier_mask_
            rf_out_mask = np.logical_not(rf_mask)
            idx_final_x.append(idx[i][0][rf_mask])  
        else:
            idx_final_x.append([])
            
    img3 = np.zeros((img_mask.shape))
    for i in range(len(idx_final_x)):
        img3[idx_final_x[i],i] = 1

    """
    fig=plt.figure()
    plt.imshow(img3,aspect='auto')
    plt.savefig('post_ransacx.png',dpi=400)
    plt.close()
    """
        
    if(np.sum(img3)/(img3.shape[0]*img3.shape[1]) < 0.3):
        min_thresh_y = img.shape[0] * 0.1
        min_thresh_x = img.shape[1] * 0.1       
    else:
        min_thresh_y = img.shape[0] * 0.1
        min_thresh_x = img.shape[1] * 0.1

    idx = np.zeros(img.shape[1])
    idy = np.zeros(img.shape[0])

    for i in range(img3.shape[1]):
        idx[i] = np.sum(img3[:,i])
    idf = np.where(idx<min_thresh_x)[0]
    for i in range(len(idf)):
        img3[:,idf[i]] = 0


    for i in range(img3.shape[0]):
        idy[i] = np.sum(img3[i,:])
    idf = np.where(idy<min_thresh_y)[0]
    for i in range(len(idf)):
        img3[idf[i],:] = 0

    """
    #re-run ransac for remaining outliers
    idx=[]
    for i in range(img.shape[0]):
        slc = img3[i,:]
        idx.append([np.where(slc==1)[0]])

    idx_final_y = []
    for i in range(len(idx)):
        if(len(idx[i][0])>1):
            # run ransac per slice to find the linear increasing line
            ransac = RANSACRegressor(residual_threshold=100)
            pt=np.linspace(0,len(idx[i][0])-1,len(idx[i][0]))
            rf = ransac.fit(pt.reshape(-1,1),idx[i][0].reshape(-1,1))
            rf_mask = rf.inlier_mask_
            rf_out_mask = np.logical_not(rf_mask)
            #x = np.arange(np.min(idx[i][0][rf_mask]),np.max(idx[i][0][rf_mask]),1)
            idx_final_y.append(idx[i][0][rf_mask])
            #idx_final_y.append(x)
        else:
            idx_final_y.append([])


    img_mask = np.zeros((img.shape[0:2]))
    for i in range(len(idx_final_y)):
        img_mask[i,idx_final_y[i]] = 1



    idx=[]
    for i in range(img_mask.shape[1]):
        slc = img_mask[:,i]
        idx.append([np.where(slc>0)[0]])   


    idx_final_x = []
    for i in range(len(idx)):
        if(len(idx[i][0])>1):
            # run ransac per slice to find the linear increasing line
            ransac = RANSACRegressor(residual_threshold=500)
            pt=np.linspace(0,len(idx[i][0])-1,len(idx[i][0]))
            rf = ransac.fit(pt.reshape(-1,1),idx[i][0].reshape(-1,1))
            rf_mask = rf.inlier_mask_
            rf_out_mask = np.logical_not(rf_mask)
            idx_final_x.append(idx[i][0][rf_mask])  
        else:
            idx_final_x.append([])

    img3 = np.zeros((img_mask.shape))
    for i in range(len(idx_final_x)):
        img3[idx_final_x[i],i] = 1


    min_thresh_y = img.shape[0] * 0.3
    min_thresh_x = img.shape[1] * 0.3

    idx = np.zeros(img.shape[1])
    idy = np.zeros(img.shape[0])

    for i in range(img3.shape[1]):
        idx[i] = np.sum(img3[:,i])
    idf = np.where(idx<min_thresh_x)[0]
    for i in range(len(idf)):
        img3[:,idf[i]] = 0


    for i in range(img3.shape[0]):
        idy[i] = np.sum(img3[i,:])
    idf = np.where(idy<min_thresh_y)[0]
    for i in range(len(idf)):
        img3[idf[i],:] = 0
        
    """
    
    img_temp = img3.copy() 
    """
    fig=plt.figure()
    plt.imshow(img3,aspect='auto')
    plt.savefig('img3.png',dpi=400)
    plt.close()
    """

    # Count the start/end/sum of all strips of missing pixels. 
    # remove all but the biggest set

    idx=[]
    for i in range(img_temp.shape[1]):
        slc = img_temp[:,i]
        idx.append([np.where(slc>0)[0]])

    start_x = []
    stop_x = []
    sum_x_i = []
    min_thresh = 0.2*img_temp.shape[0]
    start = True
    end = False
    for i in range(img_temp.shape[1]):
        if(i==0 and np.sum(np.isfinite(idx[i]))==0 and start==True and end==False):
            start_x.append(i)
            start=False
        elif(i>0 and np.sum(np.isfinite(idx[i]))>0 and start==False and end==False):
            stop_x.append(i)
            sum_x_i.append(np.sum(np.isfinite(np.where(img_temp[:,start_x[-1]:stop_x[-1]]>0)[0])))
            start_x.append(i)
            end=True  
            start=True
        elif(i>0 and np.sum(np.isfinite(idx[i]))==0 and start==True and end==True):
            stop_x.append(i)
            sum_x_i.append(np.sum(np.isfinite(np.where(img_temp[:,start_x[-1]:stop_x[-1]]>0)[0])))
            start_x.append(i)
            end=False  
            start=True
        elif(i>0 and np.sum(np.isfinite(idx[i]))>0 and start==True and end==False):
            stop_x.append(i)
            start_x.append(i)
            start=False

        if(i==img_temp.shape[1]-1):
            stop_x.append(img_temp.shape[1])
            sum_x_i.append(i)

    if(stop_x[-1] == img_temp.shape[1] and len(start_x)!=len(stop_x)):
        start_x.append(img_temp.shape[1])
    if(start_x[-1] == img_temp.shape[1] and len(start_x)!=len(stop_x)):
        stop_x.append(img_temp.shape[1])
    delta = np.array(stop_x,dtype=int) - np.array(start_x,dtype=int)

    for i in range(len(delta)):
        if(delta[i]<0.05*img_temp.shape[1]):
            img_temp[:,start_x[i]:stop_x[i]] = 0
            
            
    # Count the start/end/sum of all strips of missing pixels. 
    # remove all but the biggest set
    """
    fig=plt.figure()
    plt.imshow(img_temp,aspect='auto')
    plt.savefig('img_tempx.png',dpi=400)
    plt.close()
    """
    idy=[]
    for i in range(img_temp.shape[0]):
        slc = img_temp[i,:]
        idy.append([np.where(slc>0)[0]])

    start_y = []
    stop_y = []
    sum_y_i = []
    min_thresh = 0.2*img_temp.shape[1]
    start = True
    end = False
    for i in range(img_temp.shape[0]):
        if(i==0 and np.sum(np.isfinite(idy[i]))==0 and start==True and end==False):
            start_y.append(i)
            start=False
        elif(i>0 and np.sum(np.isfinite(idy[i]))>0 and start==False and end==False):
            stop_y.append(i)
            sum_y_i.append(np.sum(np.isfinite(np.where(img_temp[start_y[-1]:stop_y[-1],:]>0)[0])))
            start_y.append(i)
            end=True  
            start=True
        elif(i>0 and np.sum(np.isfinite(idy[i]))==0 and start==True and end==True):
            stop_y.append(i)
            sum_y_i.append(np.sum(np.isfinite(np.where(img_temp[start_y[-1]:stop_y[-1],:]>0)[0])))
            start_y.append(i)
            end=False  
            start=True
        elif(i>0 and np.sum(np.isfinite(idy[i]))>0 and start==True and end==False):
            stop_y.append(i)
            start_y.append(i)
            start=False

        if(i==img_temp.shape[0]-1):
            stop_y.append(img_temp.shape[0])
            sum_y_i.append(i)
    if(stop_y[-1] == img_temp.shape[0] and len(start_y)!=len(stop_y)):
        start_y.append(img_temp.shape[0])
    if(start_y[-1] == img_temp.shape[0] and len(start_y)!=len(stop_y)):
        stop_y.append(img_temp.shape[0])
    delta = np.array(stop_y,dtype=int) - np.array(start_y,dtype=int)
    for i in range(len(delta)):
        if(delta[i]<0.05*img_temp.shape[0]):
            img_temp[start_y[i]:stop_y[i],:] = 0

    img3 = img_temp.copy()
    """
    fig=plt.figure()
    plt.imshow(img_temp,aspect='auto')
    plt.savefig('img_tempy.png',dpi=400)
    plt.close()
    """
    min_x = np.zeros(img.shape[0]) ; min_x[:] = np.nan
    max_x = np.zeros(img.shape[0]) ; max_x[:] = np.nan
    min_y = np.zeros(img.shape[1]) ; min_y[:] = np.nan
    max_y = np.zeros(img.shape[1]) ; max_y[:] = np.nan

    for i in range(img3.shape[0]):
        if(np.isfinite(np.nansum(img3[i,:]))==True):
            idx = np.where(img3[i,:]>0)[0]
            if(len(idx)>1):
                min_x[i] = idx[0]
                max_x[i] = idx[-1]
    #"""
    for i in range(img3.shape[1]):
        if(np.isfinite(np.nansum(img3[:,i]))==True):
            idy = np.where(img3[:,i]>0)[0]
            if(len(idy)>1):
                min_y[i] = idy[0]
                max_y[i] = idy[-1]
    #print(min_y,max_y,min_x,max_x)
    """
    fig=plt.figure()
    plt.plot(min_x)
    plt.savefig('min_x.png',dpi=400)
    plt.close()
    """
    bounds = np.array([np.nanmin(min_y),np.nanmax(max_y),np.nanmin(min_x),np.nanmax(max_x)])
    print(bounds)    
    out = image_path.split('.tif')[0]+'_Mask.txt'
    np.savetxt(os.path.join(out_dir,out),bounds)
    
    fig=plt.figure()
    plt.imshow(img3,aspect='auto')
    plt.hlines(bounds[0],bounds[2],bounds[3],'red')
    plt.hlines(bounds[1],bounds[2],bounds[3],'red')
    plt.vlines(bounds[2],bounds[0],bounds[1],'red')
    plt.vlines(bounds[3],bounds[0],bounds[1],'red')
    plt.savefig(os.path.join(out_dir,image_path.split('.tif')[0]+'_Bounds.png'),dpi=500)
    #plt.show()
    plt.close()

    return_fail = np.isfinite(np.sum(bounds))

    if(return_fail == True):
        return_fail = False
    else:
        return_fail = True
        
    return return_fail,img3,bounds

        
    
if __name__=="__main__":
    image_dir = '/scratch/e.conway/DARPA_MAPS/Validation/'
    image_path='GEO_0994.tif'
    out_dir = '/scratch/e.conway/DARPA_MAPS/ValidationResults/'
    ret_fail,mask,bounds = main(image_dir,out_dir,image_path)
    
    out = image_path.split('.tif')[0]+'_Mask.txt'
    np.savetxt(out,mask)
    
    #fig = plt.figure()
    #plt.imshow(mask)
    #plt.savefig(image_file.split('.tif')[0]+'_Mask.png',dpi=400)
    #plt.close()

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
To find the best pairs of coordinates

Args:
lat points
lon points
lat pixel coordinates
lon pixel coordinates
shape of image
clue lon
clue lat
bounds of map
max meter per pixel
min meter per pixel

Out:
best pair of lon plus pixel location
best pair of lat plus pixel location

"""

import numpy as np
from scipy.optimize import curve_fit


def lin_line(x, A, B): 
    return A*x + B

def main(lat,clat,lon,clon,img_shape,clue_x,clue_y,bounds,mpix_max,mpix_min):
            summ = np.sum(bounds)
            if(np.isfinite(summ)==False):
                top_left = [0,0]
                bot_left = [img_shape[0],0]
                bot_right = [img_shape[0],img_shape[1]]
                top_right = [0,img_shape[1]]
                loc_count = 0   
                print('Failed Case') 
            else:
                #calculate distance to our four corners
                top_left = [bounds[0],bounds[2]]
                bot_left = [bounds[1],bounds[2]]
                bot_right = [bounds[1],bounds[3]]
                top_right = [bounds[0],bounds[3]]
                loc_count = 1    
            
            
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
            #min_c_dist_lon

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
                                if(np.isfinite(summ)==False):
                                    #app = np.array([dist_ul_top,dist_ur_top,dist_top,dist_lr_bot,dist_ll_bot,dist_bot])
                                    app = np.array([dist_top,dist_bot])
                                else:
                                    app = np.array([dist_ul_top,dist_ur_top,dist_lr_bot,dist_ll_bot])
                                min_c = np.min(app)
                                loc = np.where(min_c == app)[0]
                                if(min_c<min_c_dist):
                                    min_c_dist = min_c
                                    if(loc<=loc_count):
                                        min_c_dist_lat = top_lat
                                        min_c_dist_cen = dup_lat_final_cen[p][stored_index_top[p][j]] #x,y format
                                    else:
                                        min_c_dist_lat = bot_lat
                                        min_c_dist_cen = dup_lat_final_cen[k][stored_index_top[k][i]] #x,y format
                                    
                                
                                delta_lat = top_lat - bot_lat
                                delta_pix = bot_point[1] - top_point[1]
                                meter_per_pix = 1e5*delta_lat/delta_pix
                                print(top_lat,bot_lat,delta_lat,top_point,bot_point,\
                                      delta_pix,dist_top,dist_bot,meter_per_pix,mpix_max,mpix_min)
                                if(delta_pix > 0 and delta_lat > 0 and done==False and (clue_y <= (top_lat+0.05))\
                                   and (clue_y >= (bot_lat-0.05)) and (meter_per_pix<=mpix_max) and (meter_per_pix>=mpix_min)):
                                    x = np.array([top_point[1],bot_point[1]],dtype=np.float64)
                                    y = np.array([top_lat,bot_lat],dtype=np.float64)
                                    popt,pcov = curve_fit(lin_line,x,y)
                                    max_lat = lin_line(top_left[1],*popt)
                                    min_lat = lin_line(bot_left[1],*popt)
                                    if(max_lat - min_lat <= 2):
                                        #done=True
                                        top_point_cen = dup_lat_final_cen[p][stored_index_top[p][j]]
                                        top_point_lat = dup_lat_final[p][stored_index_top[p][j]]
                                        bot_point_cen = dup_lat_final_cen[k][stored_index_bot[k][i]]
                                        bot_point_lat = dup_lat_final[k][stored_index_bot[k][i]]
                                        delta_pix = bot_point[1] - top_point_cen[1] 
                                        total = dist_top + dist_bot
                                        top_dist_corner = min(dist_ul_top,dist_ur_top)
                                        bot_dist_corner = min(dist_lr_bot,dist_ll_bot)
                                        if(np.isfinite(summ)==False):
                                            top_max = dist_top + min(top_point[0],abs(top_point[0]-top_right[1]))
                                            #top_max = min(dist_top,top_dist_corner)
                                            bot_max = dist_bot + min(bot_point[0],abs(bot_point[0]-bot_right[1]))#
                                            #bot_max = min(dist_bot,bot_dist_corner)
                                        else:
                                            top_max = max(dist_top,top_dist_corner)
                                            bot_max = max(dist_bot,bot_dist_corner)
                                        total = top_max+bot_max
                                        print(total,top_point_lat,bot_point_lat,top_point_cen,\
                                              bot_point_cen,top_max,bot_max,meter_per_pix,max_lat,min_lat)
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
                if(np.isfinite(summ)==False):
                    thresh = 1e5
                else:
                    thresh = 1000
                
                if(min_dist_sum<thresh):
                    lat3d = np.zeros((3,2))
                    lat3d[0,0] = dup_lat_final_cen[arr[2]][stored_index_top[arr[2]][arr[3]]][1] 
                    lat3d[1,0] =dup_lat_final_cen[arr[2]][stored_index_top[arr[2]][arr[3]]][0] 
                    lat3d[2,0] = dup_lat_final[arr[2]][stored_index_top[arr[2]][arr[3]]]

                    lat3d[0,1] = dup_lat_final_cen[arr[0]][stored_index_bot[arr[0]][arr[1]]][1]
                    lat3d[1,1] = dup_lat_final_cen[arr[0]][stored_index_bot[arr[0]][arr[1]]][0]
                    lat3d[2,1] = dup_lat_final[arr[0]][stored_index_bot[arr[0]][arr[1]]]
                elif(min_dist_sum>thresh  ):
                    # no good pair
                    # need to sort the lats by didstance to pick best match to a corner
                    lat3d = np.zeros((3,1))
                    lat3d[0,0] = min_c_dist_cen[1]
                    lat3d[1,0] = min_c_dist_cen[0]
                    lat3d[2,0] = min_c_dist_lat

            elif(len(dist_y_bot)==1):
                #here, we only have one set of lons, all possibly duplicated
                # we need to pick the one that is closest to a corner
                if(len(stored_dist_y_ur[0])>1):
                    print(stored_dist_y_ur)
                    arr = np.array([stored_dist_y_ul[0][0],stored_dist_y_ur[0][0],stored_dist_y_lr[0][0],stored_dist_y_ll[0][0]],dtype=np.float64)
                    arr_indx = np.array([stored_indx_y_ul[0][0],stored_indx_y_ur[0][0],stored_indx_y_lr[0][0],stored_indx_y_ll[0][0]],dtype=np.float64)
                    #print(arr)
                    #print(arr_indx)
                    best_lat = np.min(arr)
                    #print(best_lon)
                    idx = int(np.where(arr==best_lat)[0])
                    #print(idx)
                    lat3d=np.zeros((3,1))
                    lat3d[2,0] = dup_lat_final[0][idx]
                    lat3d[1,0] = dup_lat_final_cen[0][idx][0]
                    lat3d[0,0] = dup_lat_final_cen[0][idx][1]
                else:
                    lat3d=np.zeros((3,1))
                    lat3d[2,0] = dup_lat_final[0][0]
                    lat3d[1,0] = dup_lat_final_cen[0][0][0]
                    lat3d[0,0] = dup_lat_final_cen[0][0][1]                  
            else:
                lat3d = np.zeros((3,1))

            
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
                                if(np.isfinite(summ)==False):
                                    #app = np.array([dist_ul_left,dist_ll_left,dist_left,dist_ur_right,dist_lr_right,dist_right])
                                    app = np.array([dist_left,dist_right])
                                else:
                                    app = np.array([dist_ul_left,dist_ll_left,dist_ur_right,dist_lr_right])
                                min_c = np.min(app)
                                loc = np.where(min_c == app)[0]
                                if(min_c<min_c_dist):
                                    min_c_dist = min_c
                                    if(loc<=loc_count):
                                        min_c_dist_lon = left_lon
                                        min_c_dist_cen = dup_lon_final_cen[p][stored_index_left[p][j]]
                                    else:
                                        min_c_dist_lon = right_lon
                                        min_c_dist_cen = dup_lon_final_cen[k][stored_index_right[k][i]]
                                delta_lon = right_lon - left_lon
                                delta_pix = right_point[0] - left_point[0]
                                meter_per_pix = 1e5*delta_lon/delta_pix
                                print(right_lon,left_lon,delta_lon,right_point,left_point,delta_pix,meter_per_pix,mpix_max,mpix_min)
                                if(delta_pix > 0 and delta_lon > 0 and done==False and (clue_x <= (right_lon+0.05))\
                                   and (clue_x >= (left_lon-0.05)) and (meter_per_pix<=mpix_max) and (meter_per_pix>=mpix_min)):
                                    x = np.array([left_point[0],right_point[0]],dtype=int)
                                    y = np.array([left_lon,right_lon],dtype=np.float64)
                                    popt,pcov = curve_fit(lin_line,x,y)
                                    max_lon = lin_line(top_left[0],*popt)
                                    min_lon = lin_line(top_right[0],*popt)
                                    if(max_lon - min_lon <= 2):
                                        #done=True
                                        left_point_cen = dup_lon_final_cen[p][stored_index_left[p][j]]
                                        left_point_lon = dup_lon_final[p][stored_index_left[p][j]]
                                        right_point_cen = dup_lon_final_cen[k][stored_index_right[k][i]]
                                        right_point_lon = dup_lon_final[k][stored_index_right[k][i]]
                                        delta_pix = right_point[1] - left_point_cen[1] 

                                        total = dist_right + dist_left
                                        left_dist_corner = min(dist_ul_left,dist_ll_left)
                                        right_dist_corner = min(dist_ur_right,dist_lr_right)
                                        if(np.isfinite(summ)==False):
                                            left_max = min(dist_left,left_dist_corner)
                                            right_max = min(dist_right,right_dist_corner)
                                        else:
                                            left_max = max(dist_left,left_dist_corner)
                                            right_max = max(dist_right,right_dist_corner)
                                        total = left_max + right_max
                                        print(total,left_point_lon,right_point_lon,\
                                              left_point_cen,right_point_cen,left_max,\
                                              right_max,meter_per_pix,max_lon,min_lon)
                                        if(total<min_dist_sum):
                                            min_dist_sum = total
                                            arr = [k,i,p,j]
                                        if(dist_left<min_dist_left):
                                            min_dist_left = dist_left
                                        if(dist_right<min_dist_right):
                                            min_dist_right = dist_right
                                        delta_lon = right_point_lon - left_point_lon 
                                        meter_per_pix = 1e5*delta_lon/delta_pix
                """
                print(min_dist_sum)        
                print(min_dist_right,arr) 
                print(min_dist_left,arr) 
                print(dup_lon_final_cen[arr[2]][stored_index_left[arr[2]][arr[3]]])
                print(dup_lon_final_cen[arr[0]][stored_index_right[arr[0]][arr[1]]])
                print(dup_lon_final[arr[2]][stored_index_left[arr[2]][arr[3]]])
                print(dup_lon_final[arr[0]][stored_index_right[arr[0]][arr[1]]])
                """
                if(np.isfinite(summ)==False):
                    thresh = 1e5
                else:
                    thresh = 1000
                if(min_dist_sum<thresh):
                    lon3d = np.zeros((3,2))
                    lon3d[0,0] = dup_lon_final_cen[arr[2]][stored_index_left[arr[2]][arr[3]]][1] 
                    lon3d[1,0] =dup_lon_final_cen[arr[2]][stored_index_left[arr[2]][arr[3]]][0] 
                    lon3d[2,0] = dup_lon_final[arr[2]][stored_index_left[arr[2]][arr[3]]]

                    lon3d[0,1] = dup_lon_final_cen[arr[0]][stored_index_right[arr[0]][arr[1]]][1]
                    lon3d[1,1] = dup_lon_final_cen[arr[0]][stored_index_right[arr[0]][arr[1]]][0]
                    lon3d[2,1] = dup_lon_final[arr[0]][stored_index_right[arr[0]][arr[1]]]  

                elif(min_dist_sum>thresh ):
                    lon3d = np.zeros((3,1))
                    print(min_c_dist_cen)
                    lon3d[0,0] = min_c_dist_cen[1]
                    lon3d[1,0] = min_c_dist_cen[0]
                    lon3d[2,0] = min_c_dist_lon

                    
            elif(len(dist_x_right)==1):
                #here, we only have one set of lons, all possibly duplicated
                # we need to pick the one that is closest to a corner
                if(len(stored_dist_x_ur[0])>1):
                    print(stored_dist_x_ur)
                    arr = np.array([stored_dist_x_ul[0][0],stored_dist_x_ur[0][0],stored_dist_x_lr[0][0],stored_dist_x_ll[0][0]],dtype=np.float64)
                    arr_indx = np.array([stored_indx_x_ul[0][0],stored_indx_x_ur[0][0],stored_indx_x_lr[0][0],stored_indx_x_ll[0][0]],dtype=np.float64)
                    #print(arr)
                    #print(arr_indx)
                    best_lon = np.min(arr)
                    #print(best_lon)
                    idx = int(np.where(arr==best_lon)[0])
                    #print(idx)
                    lon3d=np.zeros((3,1))
                    lon3d[2,0] = dup_lon_final[0][idx]
                    lon3d[1,0] = dup_lon_final_cen[0][idx][0]
                    lon3d[0,0] = dup_lon_final_cen[0][idx][1]
                else:
                    lon3d=np.zeros((3,1))
                    lon3d[2,0] = dup_lon_final[0][0]
                    lon3d[1,0] = dup_lon_final_cen[0][0][0]
                    lon3d[0,0] = dup_lon_final_cen[0][0][1]                  
            else:
                lon3d=np.zeros((3,1))
                
                
            return lat3d,lon3d

if __name__=="__main__":
    file='/scratch/e.conway/DARPA_MAPS/Training/GEO_0089.tif'
    with rasterio.open(file,'r') as f:
        data=f.read()
    data = data.transpose((1,2,0))
    lon = [-120,-119]  
    lat = [40,39]
    clue_x = -120
    clue_y = 39  
    clon = [[10,20],[3000,4000]]
    clat = [[10,20],[3000,4000]]
    bounds = [np.nan,np.nan,np.nan,np.nan]


            
            
            

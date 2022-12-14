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
Considers all keywords and fuses nearby keywords together

Args:
keywords,
bboxes
centers

Out:
keywords,
bboxes,
centers


"""

import math
import numpy as np
from tqdm import tqdm
import time

def main(keywords,bboxes,centers,clue_x,clue_y):

            if(abs(int(clue_x))<100):
                clue_x = str(clue_x)[1:3]
            else:
                clue_x = str(clue_x)[1:4]

            clue_y = str(clue_y)[0:2]
            


            new_keywords = []
            new_bboxes = []
            new_centers=[]
            append = []
            tz = time.time()
            done=False
            success = False
            while done==False:
                # merge keywords if close together
                x = []
                for i in tqdm(range(len(keywords))):
                    tnow = (time.time() - tz)
                    app=True
                    contin = True
                    if(keywords[i]!=''):
                        if(clue_x in keywords[i] or clue_y in keywords[i]):
                            print(keywords[i])
                            top_left_1 = [bboxes[i][0][0],bboxes[i][0][1]]
                            top_right_1 = [bboxes[i][1][0],bboxes[i][0][1]]
                            bot_right_1 = [bboxes[i][1][0],bboxes[i][1][1]]
                            bot_left_1 = [bboxes[i][0][0],bboxes[i][1][1]]
                            tnow = (time.time() - tz)
                            if(tnow > 100):
                                break
                                done=True
                            loop_done = False
                            while loop_done == False:
                                for j in range(len(keywords)):
                                    if(keywords[j]!=''):
                                        if(('scale' in keywords[i]) or ('scale' in keywords[j])):
                                            tol=35
                                        else:
                                            tol=35
                                        top_left_2 = [bboxes[j][0][0],bboxes[j][0][1]]
                                        top_right_2 = [bboxes[j][1][0],bboxes[j][0][1]]
                                        bot_right_2 = [bboxes[j][1][0],bboxes[j][1][1]]
                                        bot_left_2 = [bboxes[j][0][0],bboxes[j][1][1]]
                                        # is box 2 close to box 1 on left side of 1
                                        if(math.isclose(bot_right_2[0],bot_left_1[0],abs_tol=tol) and \
                                          math.isclose(bot_right_2[1],bot_left_1[1],abs_tol=tol) and i!=j and \
clue_x not in keywords[j] and clue_y not in keywords[j]):
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
                                            loop_done=True
                                        # is box 2 close to box 1 on right side of 1
                                        if(math.isclose(bot_right_1[0],bot_left_2[0],abs_tol=tol) and \
                                          math.isclose(bot_right_1[1],bot_left_2[1],abs_tol=tol) and i!=j and \
clue_x not in keywords[j] and clue_y not in keywords[j]):
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
                                            loop_done=True
                                    if(j==len(keywords)-1):
                                        loop_done=True
                        if(contin == True):
                                for en in x:
                                    for yn in en:
                                        if(i == yn):
                                            app=False
                                if(app==True):
                                    new_keywords.append(keywords[i])
                                    new_bboxes.append(bboxes[i])
                                    new_centers.append(centers[i])
                if(tnow<=100):
                    success = True
                    done=True
                    print('MergeKeys Successful, Time =  ',(time.time() - tz))
                else:
                    break
                    print('MergeKeys Not Successful, Time =  ',(time.time() - tz))
            if(success==True):
                keywords = new_keywords
                bboxes = new_bboxes
                centers = new_centers
            
            return keywords,bboxes,centers

        
        
if __name__=="__main__":       
    
    fname = 'GEO_0050.txt'
    with open('/scratch/e.conway/DARPA_MAPS/TrainingKeys/'+fname,'r') as f:
        data = f.readlines()
    tot_numbers = []
    for x in data:
        tot_numbers.append(x.split('\n')[0])
        
    with open('/scratch/e.conway/DARPA_MAPS/TrainingCenters/'+fname,'r') as f:
        data = f.readlines()
    tot_centers = []
    for x in data:
        tot_centers.append([np.float64(x.split(' ')[0]),np.float64(x.split(' ')[1].split('\n')[0])])
        
    with open('/scratch/e.conway/DARPA_MAPS/TrainingBBoxes/'+fname,'r') as f:
        data = f.readlines()
    tot_bboxes = []
    for x in data:
        tot_bboxes.append([[np.float64(x.split(' ')[0]),np.float64(x.split(' ')[1])],\
                           [np.float64(x.split(' ')[2]),np.float64(x.split(' ')[3].split('\n')[0])]])

    clues = np.genfromtxt('/scratch/e.conway/DARPA_MAPS/CluesTesting/'+fname.split('.txt')[0]+'_clue.csv',delimiter=',')
    clue_x = clues[0]
    clue_y = clues[1]        
    
    keywords,bboxes,centers = main(tot_numbers,tot_bboxes,tot_centers,clue_x,clue_y)
        
    print('New Keys = ',keywords)
    """
    tot_numbers = ['105','030']
    print('Old Keys = ',tot_numbers)
    tot_num_centers = [[100,505],[120,505]]
    tot_num_boxes = [[[100,500],[110,510]],[[110,510],[120,520]]]
    clue_x = -105.3
    clue_y = 30
    keywords,bboxes,centers = main(tot_numbers,tot_num_boxes,tot_num_centers,clue_x,clue_y)
    print('New Keys = ',keywords)
    """

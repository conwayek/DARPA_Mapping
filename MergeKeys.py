import math
import numpy as np
from tqdm import tqdm
import time

def main(keywords,bboxes,centers):
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
                    if(keywords[i]!=''):
                        #print(keywords[i])
                        top_left_1 = [bboxes[i][0][0],bboxes[i][0][1]]
                        top_right_1 = [bboxes[i][1][0],bboxes[i][0][1]]
                        bot_right_1 = [bboxes[i][1][0],bboxes[i][1][1]]
                        bot_left_1 = [bboxes[i][0][0],bboxes[i][1][1]]
                        contin = True
                        tnow = (time.time() - tz)
                        if(tnow > 3600):
                            done=True
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
                if(tnow<=3600):
                    success = True
                    done=True
                    print('MergeKeys Successful, Time =  ',(time.time() - tz))
                else:
                    print('MergeKeys Not Successful, Time =  ',(time.time() - tz))
                
            if(success==True):
                keywords = new_keywords
                bboxes = new_bboxes
                centers = new_centers
            
            return keywords,bboxes,centers

        
        
if __name__=="__main__":        
    tot_numbers = ['hello','world']
    tot_num_centers = [[100,505],[120,505]]
    tot_num_boxes = [[[100,500],[110,510]],[[110,510],[120,520]]]
    keywords,bboxes,centers = main(tot_numbers,tot_num_boxes,tot_num_centers)
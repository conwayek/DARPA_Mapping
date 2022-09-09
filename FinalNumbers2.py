import os
import numpy as np
import math

def main(tot_numbers,tot_num_centers,tot_num_boxes,x,y):
            final_numbers=[]
            final_num_boxes = []
            final_num_centers = []
            x=np.array(x,dtype=np.float64)
            y=np.array(y,dtype=np.float64)
            idx = np.where(x==0)[0]
            x[idx] = 99999
            idy = np.where(y==0)[0]
            y[idy] = 99999 
            for j in range(len(tot_numbers)):
                word = tot_numbers[j]
                count=0
                pos=[]
                itert=-1
                done=False
                while done == False:
                    for i in range(1,len(word)):
                        if(done==False):
                            num = np.float64(word[0:i])
                            num_text = word[0:i]
                            #print(word,num,x[j],y[j])
                            if(math.isclose(abs(num),abs(x[j]),abs_tol=3) or math.isclose(abs(num),abs(y[j]),abs_tol=3)):
                                end_int = i 
                                if(len(word)==i):
                                    # no more chars
                                    final_numbers.append(num)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    done=True
                                # there one more character left
                                elif(len(word)==i+1):
                                    # is the next one a zero
                                    if(word[i] == '0'):
                                        final_numbers.append(num)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                        done=True
                                # there are two more character left
                                elif(len(word)==i+2):
                                    # if less than 60, it should be added interpreted as minutes
                                    if(np.float64(word[end_int:])<60):
                                        final_numbers.append(num+np.float64(word[end_int:])/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                        done=True

                                    else:
                                        # if less than 60, it should be added interpreted as minutes
                                        final_numbers.append(num+np.float64(word[end_int])/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                        done=True
                                # there are three more character left
                                elif(len(word)==i+3):
                                    # first 'minute' is zero
                                    if(word[end_int]=='0'):
                                        # if remainder less than 60, it should be added interpreted as minutes
                                        if(np.float64(word[end_int+1:])<60):
                                            final_numbers.append(num+np.float64(word[end_int+1:])/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                            done=True
                                        else:
                                            # if less than 60, it should be added interpreted as minutes
                                            final_numbers.append(num+np.float64(word[end_int+1])/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                            done=True
                                    # else first 'minute' is non-zero
                                    else:
                                        # if remainder less than 60, it should be added interpreted as minutes
                                        if(np.float64(word[end_int:end_int+2])<60):
                                            if(np.float64(word[end_int+2:])<60):
                                                final_numbers.append(num+np.float64(word[end_int:end_int+2])/60+np.float64(word[end_int+2:])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                            else:
                                                final_numbers.append(num+np.float64(word[end_int:end_int+2])/60+np.float64(word[end_int+2])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                        else:
                                            # if more than 60, it should be added interpreted as minutes
                                            if(np.float64(word[end_int+2:])<60):
                                                final_numbers.append(num+np.float64(word[end_int+1])/60+np.float64(word[end_int+2:])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True 
                                            else:                                              
                                                final_numbers.append(num+np.float64(word[end_int+1])/60+np.float64(word[end_int+2])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True 
                                # there are four or more character left
                                elif(len(word)>=i+4):
                                    # first 'minute' is zero
                                    if(word[end_int]=='0'):
                                        
                                        # if remainder less than 60, it should be added interpreted as minutes
                                        if(np.float64(word[end_int+1:end_int+3])<60):
                                            if(np.float64(word[end_int+3:])<60):
                                                final_numbers.append(num+np.float64(word[end_int+1:end_int+3])/60+np.float64(word[end_int+3:])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                            elif(np.float64(word[end_int+3:end_int+5])<60):
                                                final_numbers.append(num+np.float64(word[end_int+1:end_int+3])/60+np.float64(word[end_int+3:end_int+5])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                        else:
                                            # if less than 60, it should be added interpreted as minutes
                                            if(np.float64(word[end_int+2:])<60):
                                                final_numbers.append(num+np.float64(word[end_int+1])/60+np.float64(word[end_int+2:])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                            else:
                                                final_numbers.append(num+np.float64(word[end_int+1])/60+np.float64(word[end_int+2:end_int+4])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                    # else first 'minute' is non-zero
                                    else:
                                        # if remainder less than 60, it should be added interpreted as minutes
                                        if(np.float64(word[end_int:end_int+2])<60):
                                            if(np.float64(word[end_int+2:])<60):
                                                final_numbers.append(num+np.float64(word[end_int:end_int+2])/60+np.float64(word[end_int+2:])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                            else:
                                                final_numbers.append(num+np.float64(word[end_int:end_int+2])/60+np.float64(word[end_int+2:end_int+4])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                        else:
                                            # if more than 60, it should be added interpreted as minutes
                                            if(np.float64(word[end_int+1:])<60):
                                                final_numbers.append(num+np.float64(word[end_int])/60+np.float64(word[end_int+1:])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True
                                            else:    
                                                final_numbers.append(num+np.float64(word[end_int])/60+np.float64(word[end_int:end_int+3])/3600)
                                                final_num_centers.append(tot_num_centers[j])
                                                final_num_boxes.append(tot_num_boxes[j])
                                                done=True

                                
                    done=True
            return final_numbers,final_num_centers,final_num_boxes




if __name__=="__main__":
    
    
    tot_numbers = ['9045','91200','42080','9045', '423780','10907', '4100',\
                  '7630', '4037001','40331451','4137301','41130','830071307','8307130',\
                   '37022301','83071301','83007301','37022130','1070730','380750','3807300',\
                  '36045','116022130','11602230','11615','116015','10745','10737307','370521',\
                   '3752300','11930','3937000','4005230']
    tot_num_centers = [[100,500],[200,600],[300,700],[200,600],[300,700],[200,600],[300,700],\
                      [200,600],[300,700],[300,700],[300,700],[300,700],[300,700],[300,700]\
                       ,[300,700],[300,700],[300,700],[300,700],[300,700],[300,700],[300,700],\
                      [300,700],[300,700],[300,700],[300,700],[300,700],[300,700],[300,700],[300,700]\
                      ,[300,700],[300,700],[300,700],[300,700]]
    tot_num_boxes = [[[100,500],[110,510]],[[200,600],[210,610]],[[300,700],[310,710]],\
                     [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]]]
    cluex = [90,91.3,0,90.75,0,109.1,0,\
            76.5,0,0,0,0,83,83,\
            0,83,83,0,107.1,0,0,\
            0,116,116,116,116,107,107,0,\
            0,119,0,0]
    cluey = [0,0,42.1,0,42.5,0,41,\
            0,40.5,40.6,41.5,41.2,0,0,\
            37,0,0,37,0,38.1,38.1,\
            36.75,0,0,0,0,0,0,37,\
            37,0,39.6,40]
    
    final_keys,final_cen,final_bboxes = main(tot_numbers,tot_num_centers,tot_num_boxes,cluex,cluey)
    print(final_keys)

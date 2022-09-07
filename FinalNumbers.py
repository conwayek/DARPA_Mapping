import os
import numpy as np

def main(tot_numbers,tot_num_centers,tot_num_boxes):
            final_numbers=[]
            final_num_boxes = []
            final_num_centers = []
            for j in range(len(tot_numbers)):
                word = tot_numbers[j]
                count=0
                pos=[]
                itert=-1
                for x in word:
                    # count zeros
                    itert+=1
                    if(x=='0'):
                        count+=1
                        pos.append(itert)
                if count==0 and len(word)==5:
                    if(np.float64(word[3:])/60):
                            final_numbers.append(np.float64(word[0:3])+np.float64(word[3:])/60)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                    if(np.float64(word[2:4])/60):
                            final_numbers.append(np.float64(word[0:2])+np.float64(word[2:4])/60)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                        
                if count==1 and len(word)==5:
                    x=word
                    done=False
                    y = x.split('0')
                    
                    for i in range(len(x)):
                        # remove first '0' for 5 char nums with one zero found
                        if(i>0 and x[i] == '0' and done==False and i==pos[0]):
                            done=True
                            
                            if(i>2):

                                # remove the '0', make mins to dec deg
                                if(len(y[0]) ==1):
                                    final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                elif(len(y[0]) ==2):
                                    final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                elif(len(y[0]) ==3):
                                    left = (np.float64(y[0][0:2]))
                                    right =  (np.float64(y[0][2]+y[1])) / 60
                                    final_numbers.append(left+right)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                elif(len(y[1]) ==0):
                                    
                                    if(np.float64(word[2:4])<60):
                                        final_numbers.append(np.float64(word[0:2])+np.float64(word[2:4])/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                        final_numbers.append(np.float64(word[0:3])+np.float64(word[3:5])/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])

                            else:
                                if(len(y[1]) ==1):
                                    final_numbers.append(np.float64(y[0])+np.float64(y[1])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                elif(len(y[1]) ==2):
                                    final_numbers.append(np.float64(y[0])+np.float64(y[1])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                elif(len(y[1]) ==3):
                                    if(i==1):   
                                        if(np.float64(y[1][0:2])<60):
                                            final_numbers.append(np.float64(y[0]+'0')+np.float64(y[1][0:2])/60+np.float64(y[1][2])/3600)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                        if(np.float64(y[1][1:3])<60):
                                            final_numbers.append(np.float64(y[0]+'0'+y[1][0])+np.float64(y[1][1:])/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                elif(len(y[1]) ==0):
                                    final_numbers.append(np.float64(y[0]))
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                #"""
                elif count==1 and len(word)==4 :
                    x=word
                    done=False
                    for i in range(len(x)):
                                # remove first '0' for 4 char nums with one zeros found
                                done=True
                                y = x.split('0')
                                if(x[0]=='0'):
                                    final_numbers.append(np.float64(word[1:]))
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j]) 
                                    final_numbers.append(np.float64(word[1:3])+np.float64(word[3])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j]) 
                                elif(x[1]=='0'):
                                    final_numbers.append(np.float64(word[0:2])+np.float64(word[2:])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j]) 
                                elif(x[2]=='0'):
                                    final_numbers.append(np.float64(word[0:3])+np.float64(word[3:])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j]) 
                                elif(x[3]=='0'):
                                    final_numbers.append(np.float64(word[0:2])+np.float64(word[2:])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j]) 
                                """
                                if(len(y[0])>1):
                                    if(len(y[1]) ==1):
                                        final_numbers.append(np.float64(y[0]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    elif(len(y[1]) ==2):
                                        final_numbers.append(np.float64(y[0]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    elif(len(y[1]) ==3):
                                        final_numbers.append(np.float64(y[0]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    elif(len(y[1]) ==0):
                                        if(len(y[0])==3):               
                                            left = y[0][0:2]
                                            right=y[0][2]
                                            final_numbers.append(np.float64(left)+np.float64(right)/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                    """
    


                elif count==3 and len(word)==4 :
                    x=word
                    done=False
                    for i in range(len(x)):
                        # remove second '0' for 4 char nums with three zeros found
                        if(i>0 and x[i] == '0' and done==False and i==pos[1]):
                                done=True
                                if(x[0]!='0'):
                                    final_numbers.append(np.float64(x[0])*10)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(x[0])*100)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                elif(x[0]=='0' and x[1]!='0'):
                                    final_numbers.append(np.float64(x[1])*10)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(x[1])*100)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                elif(x[0]=='0' and x[1]=='0' and x[2]!='0'):
                                    final_numbers.append(np.float64(x[2])*10)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(x[2])*100)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                elif(x[0]=='0' and x[1]=='0' and x[2]=='0' and x[3]!='0'):
                                    final_numbers.append(np.float64(x[3])*10)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(x[3])*100)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                elif count==4 and len(word)==6:
                    x=word
                    done=False
                    y = word.split('0')
                    if(x[-3:]=='000'):
                        done=True
                        final_numbers.append(np.float64(word[0:3]))
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])

                elif count==2 and len(word)==4:
                    x=word
                    done=False
                    for i in range(len(x)):
                        y = x.split('0')
                        if(len(y[1])==2):
                                    final_numbers.append(np.float64(y[1]))
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(y[1])*10)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                        if(len(y[0])==2):
                                    final_numbers.append(np.float64(y[0]))
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(y[0])*10)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                        if(len(y[0])==1 and len(y[1])==1):
                               if(word[0]!='0' and word[1]=='0' and word[2]!='0'):
                                    final_numbers.append(np.float64(word[0:3]))
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(word[0:3])+np.float64(word[3:])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])

                elif count==1 and len(word)>=2 and len(word)<3:
                    x=word
                    done=False
                    final_numbers.append(np.float64(word)*10)
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j])
                    final_numbers.append(np.float64(word))
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j])
                elif count==2 and len(word)==3:
                    x=word
                    y = x.split('0')
                    done=False
                    if(x[0]=='0' and x[1]!='0'):
                        final_numbers.append(np.float64(x[1]))
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                        final_numbers.append(np.float64(x[1])*10)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                    elif(x[0]!='0' and x[1]!='0'):
                        final_numbers.append(np.float64(x[1]))
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                        final_numbers.append(np.float64(x[1])*10)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                    elif(x[0]=='0' and x[1]=='0'):
                        final_numbers.append(np.float64(x[2]))
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                        final_numbers.append(np.float64(x[2])*10)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                elif count==3 and len(word)==6:
                    x=word
                    done=False
                    y = word.split('0')
                    if(x[-2:]=='00'):
                        done=True
                        final_numbers.append(np.float64(word[0:3])+np.float64(word[-3:-1])/60)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                    for i in range(len(x)):
                        if(len(y[0])==1 and len(y[1])==2):
                            final_numbers.append(np.float64(y[0])*100+np.float64(y[1]))
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j]) 
                elif count==3 and len(word)>4 and len(word)<=5:
                    x=word
                    done=False
                    y = word.split('0')
                    if(len(y[0])==1 and len(y[1])==1):
                        done=True
                        final_numbers.append(np.float64(y[0]+y[1]))
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                    if(x[1]=='0' and x[0]!='0' and x[2]!='0' ):
                        done=True
                        final_numbers.append(np.float64(word[0:3]))
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])        
                    if(len(y[0])>1 ):
                        done=True
                        final_numbers.append(np.float64(y[0]))
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                elif count==0 and len(word)==4:
                    x=word
                    done=False
                    final_numbers.append(np.float64(word[0:3]))
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j])  
                    final_numbers.append(np.float64(word[0:3])+np.float64(word[3:])/60)
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j]) 
                    final_numbers.append(np.float64(word[0:2])+np.float64(word[2:])/60)
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j]) 


                elif count==0 and len(word)==2:
                    x=word
                    done=False
                    final_numbers.append(np.float64(word[0:2]))
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j])  
                    final_numbers.append(np.float64(word[0:1])+np.float64(word[1:])/60)
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j]) 
                elif count==1 and len(word)==3:
                    x=word
                    done=False
                    y=x.split('0')
                    final_numbers.append(np.float64(y[0]+y[1]))
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j])  
                    final_numbers.append(np.float64(word[0:2])+np.float64(word[2:])/60)
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j]) 
                    final_numbers.append(np.float64(word))
                    final_num_centers.append(tot_num_centers[j])
                    final_num_boxes.append(tot_num_boxes[j]) 
                elif count==2 and len(word)==5:
                    x=word
                    done=False
                    y = x.split('0')
                    
                    if(len(y[1]) ==1 and len(y[0])==1):
                                    if(pos[1]==4):
                                        final_numbers.append(np.float64(y[0])+np.float64(y[1]+'0')/60+np.float64(y[2][0:-1])/1000)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    elif(pos[1]==3 and pos[0]==1):
                                        final_numbers.append(np.float64(word[0:3])+np.float64(word[3:])/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                        if(np.float64(word[2:])<60):
                                            final_numbers.append(np.float64(word[0:2])+np.float64(word[2:4])/60+np.float64(word[4:])/3600)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                        
                                    else:
                                        if(len(y[0])==1 and word[0]=='0' and len(y[1]) >1 and word[2]=='0'):
                                            final_numbers.append(np.float64(word[1:3])+np.float64(word[3:5])/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])

                    elif(len(y[0]) ==2 and len(y[1])==1):
                                    
                                    
                                    if(pos[1]==4):
                                        if(word[3]!=0):
                                
                                            final_numbers.append(np.float64(y[0]+'0')+np.float64(y[1]+'0')/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])     
                                            if(word[2]=='0'):
                                                if(np.float64(word[-2:]) < 60):
                                                    final_numbers.append(np.float64(y[0]+'0')+np.float64(y[1]+'0')/60)
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                                                    final_numbers.append(np.float64(y[0])+np.float64(y[1]+'0')/60)
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                                                else:
                                                    final_numbers.append(np.float64(y[0]+'0')+np.float64(y[1])/60)
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                                                    final_numbers.append(np.float64(y[0])+np.float64(y[1])/60)
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                                                
                                        else:        
                                            final_numbers.append(np.float64(y[0]+'0')+np.float64(y[1])/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])


                    elif(len(y[1]) ==2 and len(y[0])==1):
                                    final_numbers.append(np.float64(y[0]+y[1]))
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(y[0])+np.float64(y[1])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(x[0:3])+np.float64(x[3:])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                    elif(len(y[1]) ==3 and len(y[0])==1):
                                    final_numbers.append(np.float64(y[0]+y[1]))
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(y[0]+y[1][0:2])+np.float64(y[1][2])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(x[:3])+np.float64(x[3:5])/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                    elif(len(y[1]) ==3 and word[0]=='0'):
                                    final_numbers.append(np.float64(y[1]))
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                    final_numbers.append(np.float64(y[1][0:2])+np.float64(y[1][2]+'0')/60)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                    elif(len(y[1]) ==0 and len(y[0])!=0):
                                    if(len(y[0])==3):
                                                    final_numbers.append(np.float64(word[0:3]))
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                                        
                                    else:
                                        if(word[1:3]=='00'):
                                                    final_numbers.append(np.float64(word[0:2]+'0')+np.float64(word[3:])/60)
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                                                    final_numbers.append(np.float64(word[0:2])+np.float64(word[3:])/60)
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                                        elif(word[2:4]=='00'):
                                                    final_numbers.append(np.float64(word[0:2]+'0'))
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                                                    final_numbers.append(np.float64(word[0:2]))
                                                    final_num_centers.append(tot_num_centers[j])
                                                    final_num_boxes.append(tot_num_boxes[j])
                    if(len(y)==3):
                        if(len(y[1]) ==0 and len(y[0])==0 and len(y[2])!=0):
                                        final_numbers.append(np.float64(y[2]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                        if(len(y[2])==3):
                                            final_numbers.append(np.float64(y[2][0:2])+np.float64(y[2][2])/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                            final_numbers.append(np.float64(y[2]))
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                        elif(len(y[2])==4):
                                            final_numbers.append(np.float64(y[2][0:3])+np.float64(y[2][3])/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])
                                            final_numbers.append(np.float64(y[2][0:3]/10)+np.float64(y[2][3])/60)
                                            final_num_centers.append(tot_num_centers[j])
                                            final_num_boxes.append(tot_num_boxes[j])

                elif count==5 and len(word)==9:
                    x=word
                    done=False
                    for i in range(len(x)):
                            # remove first '0' for 5 char nums with two zeros found
                            if(i>0 and x[i] == '0'):
                                y = x.split('0')
                                # last remaining digits == 0
                                if(len(y[1]) ==1 and len(y[0])==3):
                                    if(pos[0]==3):
                                        final_numbers.append(np.float64(y[0])+np.float64(y[1]+'0')/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    else:
                                        final_numbers.append(np.float64(y[0]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                elif count==2 and len(word)==9:
                    x=word
                    done=False
                    y = x.split('0')
                    # last remaining digits == 0
                
                    if(len(y[1]) ==1 and len(y[0])==3):
                        if(pos[0]==3):
                            final_numbers.append(np.float64(y[0])+np.float64(y[1]+'0')/60)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                        else:
                            final_numbers.append(np.float64(y[0]))
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                    elif(len(y[1]) ==4 and len(y[0])==2):
                        if(pos[0]==2):
                            final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                        else:
                            final_numbers.append(np.float64(y[0]))
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                    elif(len(y[1]) ==4 and len(y[0])==3):
                        if(pos[0]==3):
                            if(np.float64(y[1][0:2])<60):
                                final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60+np.float64(y[1][2:])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                            else:
                                final_numbers.append(np.float64(y[0])+np.float64(y[1][0])/60+np.float64(y[1][1:])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])

                elif count==3 and len(word)==9:
                    x=word
                    done=False
                    y = x.split('0')
                    # last remaining digits == 0
                    if(pos[0]==2):
                        if(pos[1]==3):
                            if(np.float64(word[3:5])<60):
                                final_numbers.append(np.float64(y[0])+np.float64(word[3:5])/60+np.float64(word[5:8])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                            else:
                                final_numbers.append(np.float64(y[0])+np.float64(word[3])/60+np.float64(word[4:7])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])                                
                elif count==4 and len(word)==9:
                    x=word
                    done=False
                    for i in range(len(x)):
                            # remove first '0' for 5 char nums with two zeros found
                            if(i>0 and x[i] == '0'):
                                y = x.split('0')
                                # last remaining digits == 0
                                if(len(y[1]) ==1 and len(y[0])==3):
                                    if(pos[0]==3):
                                        final_numbers.append(np.float64(y[0])+np.float64(y[1]+'0')/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    else:
                                        final_numbers.append(np.float64(y[0]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                elif(len(y[1]) ==4 and len(y[0])==2):
                                    if(pos[0]==2):
                                        final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    else:
                                        final_numbers.append(np.float64(y[0]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                elif count==2 and len(word)==10:
                    x=word
                    done=False
                    for i in range(len(x)):
                            # remove first '0' for 5 char nums with two zeros found
                            if(i>0 and x[i] == '0'):
                                y = x.split('0')
                                # last remaining digits == 0
                                if(len(y[1]) >=3 and len(y[0])==3):
                                    if(pos[0]==3):
                                        final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60+np.float64(y[1][2])/1000)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    else:
                                        final_numbers.append(np.float64(y[0]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                elif(len(y[1]) ==4 and len(y[0])==2):
                                    if(pos[0]==2):
                                        final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60)
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                                    else:
                                        final_numbers.append(np.float64(y[0]))
                                        final_num_centers.append(tot_num_centers[j])
                                        final_num_boxes.append(tot_num_boxes[j])
                elif count==1 and len(word)==7:
                    x=word
                    done=False
                    y = x.split('0')
                    # last remaining digits == 0
                    
                    done=False
                    if(len(y)==2):
                        if(len(y[1]) >=3 and len(y[0])==3):
                            if(pos[0]==3):
                                done=True
                                final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60+np.float64(y[1][2])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])

                                final_numbers.append(np.float64(y[0][0:2])+np.float64(y[0][2]+'0')/60+np.float64(y[1][0:2])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                        elif(pos[0]>=5 and len(y[1]) ==1 ):
                                final_numbers.append(np.float64(word[0:2])+np.float64(word[2:4])/60+np.float64(word[4:6])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                                if(np.float64(word[3:5])<60):
                                    final_numbers.append(np.float64(word[0:3])+np.float64(word[3:5])/60+np.float64(word[5:7])/3600)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                else:
                                    final_numbers.append(np.float64(word[0:3])+np.float64(word[3:4])/60+np.float64(word[4:6])/3600)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])

                                        
                elif count==2 and len(word)==7:
                    x=word
                    done=False
                    y = x.split('0')
                    print(word,y,pos,len(y))
                    if(len(y)==3):
                        if(len(y[0])==2 and len(y[1])==3):
                            if(word[2]=='0' and word[-1]=='0'):
                                if(np.float64(y[1][0:2])<60):
                                    final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60+np.float64(y[1][2]+'0')/3600)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                else:
                                    final_numbers.append(np.float64(y[0])+np.float64('0'+y[1][0])/60+np.float64(y[1][1:])/3600)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])  
                        if(len(y[0])==5 and len(y[1])==0):
                            if(word[5]=='0'):
                                if(np.float64(y[0][2:4])<60):
                                    final_numbers.append(np.float64(y[0][0:2])+np.float64(y[0][2:4])/60+np.float64(y[0][4:5]+'0')/3600)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])
                                else:
                                    final_numbers.append(np.float64(y[0])+np.float64('0'+y[1][0])/60+np.float64(y[1][1:])/3600)
                                    final_num_centers.append(tot_num_centers[j])
                                    final_num_boxes.append(tot_num_boxes[j])  
                elif count==3 and len(word)==7:
                    x=word
                    done=False
                    y = x.split('0')
                    
                    if(pos[0]==1 and pos[1]==4 and pos[1]==5):
                        final_numbers.append(np.float64(word[0:2])+np.float64(word[2:4])/60+np.float64(word[4:6])/3600)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                    elif(pos[0]==1 and pos[1]==3 and pos[2]==6 and np.float64(word[3:5])<60):
                        final_numbers.append(np.float64(word[0:3])+np.float64(word[3:5])/60+np.float64(word[5:7])/3600)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j]) 
                    elif(pos[0]==2 and pos[1]==5 and pos[2]==6):
                        if(np.float64(word[3:5])<60):
                            final_numbers.append(np.float64(word[0:3])+np.float64(word[3:5])/60+np.float64(word[5:7])/3600)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j]) 
                        else:
                            final_numbers.append(np.float64(word[0:2])+np.float64(word[2:4])/60+np.float64(word[4:6])/3600)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j]) 
                elif count==2 and len(word)==6:
                    x=word
                    done=False
                    y = x.split('0')
                    
                    # last remaining digits == 0
                    if(len(y[0]) ==2 and len(y[1])==1 and word[3]!='0'):
                        if(pos[0]==2):
                            final_numbers.append(np.float64(y[0])+np.float64(y[1]+'0')/60)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                    elif(len(y[0]) ==2 and len(y[1])==2):
                        if(pos[0]==2 and np.float64(y[1])<60):
                            final_numbers.append(np.float64(y[0])+np.float64(y[1])/60+np.float64(word[4:])/3600)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                        elif(pos[0]==2 and np.float64(y[1])>60):
                            final_numbers.append(np.float64(y[0])+np.float64(y[1][0])/60+np.float64(word[4:])/3600)
                            final_num_centers.append(tot_num_centers[j])
                            final_num_boxes.append(tot_num_boxes[j])
                elif count==1 and len(word)==6:
                    x=word
                    done=False
                    # remove first '0' for 5 char nums with two zeros found
                    y = x.split('0')
                    
                    if (len(y)==2):
                        if(len(y[0])==5):
                            if(np.float64(word[2:4])<60):
                                final_numbers.append(np.float64(word[0:2])+np.float64(word[2:4])/60+np.float64(word[4:])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                            else:
                                final_numbers.append(np.float64(word[0:2])+np.float64(word[2:3])/60+np.float64(word[4:6])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                        elif(len(y[0])==3 and len(y[1])==2):
                            if(np.float64(y[1])<60):
                                final_numbers.append(np.float64(y[0])+np.float64(y[1])/60)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                            else:
                                final_numbers.append(np.float64(word[0:2])+np.float64(word[2:3])/60+np.float64(word[4:6])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                        elif(len(y[0])==2 and len(y[1])==3):
                            if(np.float64(y[1][0:2])<60):
                                final_numbers.append(np.float64(y[0])+np.float64(y[1][0:2])/60)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                            else:
                                final_numbers.append(np.float64(word[0:2])+np.float64(word[2:3])/60+np.float64(word[4:6])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                elif count==0 and len(word)==8:
                    x=word
                    done=False
                    for i in range(len(x)):
                        final_numbers.append(np.float64(x[0:2])+np.float64(x[2:4])/60+np.float64(x[4:6])/3600)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])

                        final_numbers.append(np.float64(x[0:3])+np.float64(x[3:5])/60+np.float64(x[5:7])/3600)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                elif count==1 and len(word)==8:
                    x=word
                    done=False
                    if(word[1]=='0'):
                        final_numbers.append(np.float64(x[0:2])+np.float64(x[2:4])/60+np.float64(x[4:6])/3600)
                        final_num_centers.append(tot_num_centers[j])
                        final_num_boxes.append(tot_num_boxes[j])
                elif count==2 and len(word)==8:
                    x=word
                    y = x.split('0')
                    done=False
                    
                    if(pos[0]==2 and len(y[1])==3):
                        if(np.float64(y[1][0:2])<60):
                            if(pos[1]==6):
                                final_numbers.append(np.float64(x[0:2])+np.float64(y[1][0:2])/60+np.float64(y[1][2]+'0')/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                        else:
                            if(pos[1]==6):
                                final_numbers.append(np.float64(x[0:2])+np.float64(y[1][0:1])/60+np.float64(y[1][1:])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                    elif(pos[0]==2 and len(y[1])==4):
                        if(np.float64(y[1][0:2])<60):
                            if(pos[1]==7):
                                final_numbers.append(np.float64(x[0:2])+np.float64(y[1][0:2])/60+np.float64(y[1][2]+'0')/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                        else:
                            if(pos[1]==7):
                                final_numbers.append(np.float64(x[0:2])+np.float64(y[1][0:1])/60+np.float64(y[1][1:])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                    elif(pos[0]==3 and len(y[0])==3):
                        if(np.float64(y[1][0:2])<60):
                            if(pos[1]==7):
                                final_numbers.append(np.float64(x[0:3])+np.float64(y[1][0:2])/60+np.float64(y[1][2]+'0')/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                        else:
                            if(pos[1]==7):
                                final_numbers.append(np.float64(x[0:2])+np.float64(y[1][0:1])/60+np.float64(y[1][1:])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                    elif(pos[0]==1 and len(y[0])==3):
                        
                        if(np.float64(y[1][0:2])<60):
                            if(pos[1]==7):
                                final_numbers.append(np.float64(x[0:3])+np.float64(y[1][0:2])/60+np.float64(y[1][2]+'0')/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                        else:
                            if(pos[1]==7):
                                final_numbers.append(np.float64(x[0:2])+np.float64(y[1][0:1])/60+np.float64(y[1][1:])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
                    elif(pos[0]==1 and len(y[0])==1):
                        print(word,y,pos)
                        if(pos[1]==6):
                            if(np.float64(word[3:5])<60):
                                final_numbers.append(np.float64(x[0:3])+np.float64(word[3:5])/60+np.float64(word[5:7])/3600)
                                final_num_centers.append(tot_num_centers[j])
                                final_num_boxes.append(tot_num_boxes[j])
            return final_numbers,final_num_centers,final_num_boxes




if __name__=="__main__":
    
    
    tot_numbers = ['9045','91200','42080','9045', '423780','10907', '4100',\
                  '7630', '4037001','40331451','4137301','41130','830071307','8307130',\
                   '37022301','83071301','83007301','37022130','1070730','380750','3807300',\
                  '36045','116022130','11602230','11615','116015','10745','10737307','370521','3752300']
    tot_num_centers = [[100,500],[200,600],[300,700],[200,600],[300,700],[200,600],[300,700],\
                      [200,600],[300,700],[300,700],[300,700],[300,700],[300,700],[300,700]\
                       ,[300,700],[300,700],[300,700],[300,700],[300,700],[300,700],[300,700],\
                      [300,700],[300,700],[300,700],[300,700],[300,700],[300,700],[300,700],[300,700]\
                      ,[300,700]]
    tot_num_boxes = [[[100,500],[110,510]],[[200,600],[210,610]],[[300,700],[310,710]],\
                     [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],[[300,700],[310,710]],\
                    [[300,700],[310,710]]]
    
    final_keys,final_cen,final_bboxes = main(tot_numbers,tot_num_centers,tot_num_boxes)
    print(final_keys)

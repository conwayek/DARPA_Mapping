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
Accounts for irregularities in Keras-OCR text

Args:
keywords,
centers,
bboxes,
clue_lon,
clue_lat

Out:
keywords,
centers,
bboxes


"""

def main(keywords,centers,bboxes,clue_x,clue_y):
    
            for i in range(len(keywords)):
                for j in range(len(keywords[i])):
                    if(keywords[i][j]=='o'):
                        if(j==0):
                            new = '0'+keywords[i][j+1:]
                            keywords[i] = new
                        elif(j<(len(keywords[i])-1)):
                            new = keywords[i][0:j]+'0'+keywords[i][j+1:]
                            keywords[i] = new
                        elif(j==(len(keywords[i])-1)):
                            new = keywords[i][0:j]+'0'
                            keywords[i] = new
                    if(keywords[i][j]=='z'):
                        if(j==0):
                            new = '7'+keywords[i][j+1:]
                            keywords[i] = new
                        elif(j<(len(keywords[i])-1)):
                            new = keywords[i][0:j]+'7'+keywords[i][j+1:]
                            keywords[i] = new
                        elif(j==(len(keywords[i])-1)):
                            new = keywords[i][0:j]+'7'
                            keywords[i] = new
                            
            if('5' in str(clue_x) or '5' in str(clue_y)):
                add_5 = True
            else:
                add_5 = False

            if('3' in str(clue_x) or '3' in str(clue_y)):
                add_3 = True
            else:
                add_3=False

            if('8' in str(clue_x) or '8' in str(clue_y)):
                add_8 = True
            else:
                add_8=False
            #print(add_3,add_5,add_8)



            for j in range(len(keywords)):
                word = keywords[j]
                for i in range(len(word)):
                    x=word[i]
                    key=word[i]
                    if(key=='l' and i==0):
                        word = '4'+word[1:]
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
                    elif(key=='l' and i>0 and i<len(word)-1):
                        word = word[:i]+'4'+word[i+1:]
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
                    elif(key=='l' and i>0 and i==len(word)-1):
                        word = word[:i]+'4'
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
            
                    if(key=='8' and i==0):
                        word = '3'+word[1:]
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
                    elif(key=='8' and i>0 and i<len(word)-1):
                        word = word[:i]+'3'+word[i+1:]
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
                    elif(key=='8' and i>0 and i==len(word)-1):
                        word = word[:i-1]+'3'
                        keywords.append(word)
                        centers.append(centers[j])
                        bboxes.append(bboxes[j])
            
                    if(key=='s' and i==0):
                        if(add_5==True):
                            word = '5'+word[1:]
                            keywords.append(word)
                            centers.append(centers[j])
                            bboxes.append(bboxes[j])
                        if(add_3==True):
                                y = '3'+word[1:]
                                keywords.append(y)
                                centers.append(centers[j])
                                bboxes.append(bboxes[j])

                    elif(key=='s' and i>0 and i<len(word)-1):
                        if(add_5==True):
                            word = word[:i]+'5'+word[i+1:]
                            keywords.append(word)
                            centers.append(centers[j])
                            bboxes.append(bboxes[j])

                        if(add_3==True):
                                y = word[:i]+'3'+word[i+1:]
                                keywords.append(y)
                                centers.append(centers[j])
                                bboxes.append(bboxes[j])
            
                    elif(key=='s' and i>0 and i==len(word)-1):
                        if(add_5==True):
                            word = word[:i-1]+'5'
                            word = word[:i]+'5'+word[i+1:]
                            keywords.append(word)
                            centers.append(centers[j])
                            bboxes.append(bboxes[j])
                        if(add_3==True):
                                y = word[:i-1]+'3'
                                keywords.append(y)
                                centers.append(centers[j])
                                bboxes.append(bboxes[j])
            

            tot_numbers = []
            tot_num_centers = []
            tot_num_boxes = []
            for i in range(len(keywords)):
                word = keywords[i]
                #print(word)
                #if('r' in word):
                #    print(word,centers[i])
                if(word.isdigit() and len(word)>1 and len(word)<=12):
                    tot_numbers.append(word)
                    tot_num_centers.append(centers[i])
                    tot_num_boxes.append(bboxes[i])
                elif(len(word)>1):
                    num_true = False
                    num_start = False
                    done=False
                    num_now = []
                    for x in word:
                        if(x.isdigit() and num_start==False):
                            num_start=True
                        if(x.isdigit()==False and num_start==True):
                            done=True
                        if(num_start==True and x.isdigit()==True and done==False):
                            num_now.append(x)
                    if(done==True):
                        tot_numbers.append(''.join(num_now))
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
    
    
            return tot_numbers,tot_num_centers,tot_num_boxes




if __name__=="__main__":
    keywords = ['9o45','912oof']
    print('Old Keys = ',keywords)
    centers = [[100,500],[200,600]]
    bboxes = [[[100,500],[110,510]],[[200,600],[210,610]],[[300,700],[310,710]]]
    clue_x = [90,91.3]
    clue_y = [0,0]
    
    tot_numbers,tot_num_centers,tot_num_boxes = main(keywords,centers,bboxes,clue_x,clue_y)
    
    print('New Keys = ',tot_numbers)

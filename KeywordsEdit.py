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
            print(add_3,add_5,add_8)



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
                elif(len(word)==3):
                    if(word[0:-1].isdigit()):
                        tot_numbers.append(word[0:-1])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[1:].isdigit()):
                        tot_numbers.append(word[1:])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                elif(len(word)==4):
                    if(word[0:3].isdigit()):
                        tot_numbers.append(word[0:3])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[1:].isdigit()):
                        tot_numbers.append(word[1:])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[0:2].isdigit()):
                        if(word[-2:] == 'oo'):
                            word = word[0:2]+'00'
                            tot_numbers.append(word)
                            tot_num_centers.append(centers[i])
                            tot_num_boxes.append(bboxes[i])
                        elif(word[-2] == 'o'):
                            word = word[0:2]+'0'
                            tot_numbers.append(word)
                            tot_num_centers.append(centers[i])
                            tot_num_boxes.append(bboxes[i])
                        else:
                            tot_numbers.append(word[0:2])
                            tot_num_centers.append(centers[i])
                            tot_num_boxes.append(bboxes[i])
                elif(len(word)==8):
                    for j in range(len(word)):
                        x = word[j]
                        if(x=='o' and j<(len(word)-1)):
                            word=word[:j]+'0'+word[j+1:]
                        elif(x=='o' and j==(len(word)-1)):
                            word=word[:j]+'0'
                    if(word.isdigit()):
                        tot_numbers.append(word)
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                elif(len(word)==7):
                    not_number=True
                    for x in word:
                        if(x.isdigit() == True):
                            not_number = False
                    if(not_number == False):
                        for j in range(len(word)):
                            x = word[j]
                            if(x.isdigit()==False and j<(len(word)-1)):
                                word=word[:j]+'0'+word[j+1:]
                            elif(x.isdigit()==False and j==(len(word)-1)):
                                word=word[:j]+'0'
                        if(word.isdigit()):
                            tot_numbers.append(word)
                            tot_num_centers.append(centers[i])
                            tot_num_boxes.append(bboxes[i])
                elif(len(word)==5):
                    if(word[0:-1].isdigit()):
                        tot_numbers.append(word[0:-1])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[1:].isdigit()):
                        tot_numbers.append(word[1:])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[0:2].isdigit()):
                        tot_numbers.append(word[0:2])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                elif(len(word)==6):
                    if(word[0:-1].isdigit()):
                        tot_numbers.append(word[0:-1])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[1:].isdigit()):
                        tot_numbers.append(word[1:])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[0:4].isdigit()):
                        tot_numbers.append(word[0:4])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
                    elif(word[0:3].isdigit()):
                        tot_numbers.append(word[0:3])
                        tot_num_centers.append(centers[i])
                        tot_num_boxes.append(bboxes[i])
    
    
            return tot_numbers,tot_num_centers,tot_num_boxes




if __name__=="__main__":
    
    tot_numbers = ['9045','91200']
    tot_num_centers = [[100,500],[200,600]]
    tot_num_boxes = [[[100,500],[110,510]],[[200,600],[210,610]],[[300,700],[310,710]]]
    cluex = [90,91.3]
    cluey = [0,0]
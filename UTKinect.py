import os
from sklearn.utils import shuffle
import numpy
import math
import torch
from scipy import ndimage as ndimage            
features = []
actions = []
labels = []
dir_ntu = r"A:\ActionRecognition\skeletons2"
fine_frame = []
def loop_actions():
    count_joint = -1
    count_pass = 0
    frames = []
    all_frames=[]
    global features
    global fine_frame
    global actions
    for file in os.listdir(dir_ntu):
        flag = False
        prev = 0
        prev_spalla = 0
        index = -1
        prev_piedi = 0
        prev_gomiti = 0
        prev_ginocchia = 0
        prev_ginocchioanklepiededx = 0
        prev_ginocchioanklepiedesx = 0
        prev_spallapolsomanosx = 0
        prev_spallapolsomanodx = 0
        prev_manohipmanodx = 0
        prev_manohipmanosx = 0
        prev_manomanopiededx = 0
        prev_manomanopiedesx = 0
        prev_kneehippiedesx = 0
        prev_kneehippiededx = 0
        with open(os.path.join(dir_ntu, file), 'r') as f:
            final_linea = []
            stringa_file = str(f)
            num_label = stringa_file[78:80]
            readline = f.readlines()
            readline.append("")
            flag_dueskeleton= readline[1].strip()
            if(flag_dueskeleton == "1" and int(num_label) <50):
                labels.append(int(num_label))            
                for idx,line in enumerate(readline[1:]):
                    if(flag_dueskeleton =="1"): # solo quelle con una persona
                        
                        if(int(num_label) <50): # le altre sono con due persone
                            if(count_joint==25):    
                                count_pass+=1
                                count_joint=-1
                                index+=1
                                if(flag == False): # se non ci sono due scheletri
                                    frames.append(final_linea)

                        
                                final_linea = []
                            elif(count_joint == -1): # dopo ogni frame, nel dataset ci sono 3 righe contenti altri valori
                                if(count_pass==0): # controllo se il numero di skeleton diventa due
                                    if(line.strip == "2"):
                                        flag = True
                                        labels.pop()
                                count_pass+=1
                                if(count_pass==3):
                                    count_pass = 0
                                    count_joint+=1
                            else:
                                temp = line.split(" ")
                                if(len(temp) > 1):
                                    final_linea.append(temp[0]) # i primi tre valori della riga sono le coordinate x,y,z
                                    final_linea.append(temp[1])
                                    final_linea.append(temp[2])
                                    count_joint+=1
                all_frames.append(frames)
                frames = []
    for azione in all_frames:
        prev = 0
        prev_spalla = 0
        prev_piedi = 0
        prev_gomiti = 0
        prev_ginocchia = 0
        prev_ginocchioanklepiededx = 0
        prev_ginocchioanklepiedesx = 0
        prev_spallapolsomanosx = 0
        prev_spallapolsomanodx = 0
        prev_manohipmanodx = 0
        prev_manohipmanosx = 0
        prev_manomanopiededx = 0
        prev_manomanopiedesx = 0
        prev_kneehippiedesx = 0
        prev_kneehippiededx = 0
        for count,linea in enumerate(azione):
            if(count==0):
                pre_linea = linea
            else:
                pre_linea = azione[count-1]

            try:
                for i in range(0,len(linea)):
                    features.append(float(linea[i]))
                    features.append(float(linea[i]) - float(pre_linea[i])) #differenza con il frame precedente
            except:
                print(len(pre_linea))
                print(i)
                print(pre_linea)
            distanza_mani = get_distance(linea,7,11)
            features.append(distanza_mani - prev)
            prev = distanza_mani
            distanza_piedi = get_distance(linea,15,19)
            features.append(distanza_piedi - prev_piedi)
            prev_piedi = distanza_piedi

            distanza_ginocchia = get_distance(linea,18,14)
            features.append(distanza_ginocchia- prev_ginocchia)
            prev_ginocchia = distanza_ginocchia

            distanza_gomiti = get_distance(linea,10,6)
            features.append(distanza_gomiti - prev_gomiti)
            prev_gomiti = distanza_gomiti

            distanza_spalla = get_distance(linea,9,5)
            features.append(distanza_spalla - prev_spalla)
            prev_spalla = distanza_spalla


            distanza_ginocchioanklepiededx = angle_between(linea,19,18,20)
            distanza_ginocchioanklepiedesx = angle_between(linea,16,15,14)
            distanza_spallapolsomanosx = angle_between(linea,7,6,5)
            distanza_spallapolsomanodx = angle_between(linea,11,10,9)
            distanza_manohipmanodx = angle_between(linea,7,12,13)
            distanza_manohipmanosx = angle_between(linea,17,11,7)
            distanza_manomanopiededx = angle_between(linea,7,11,19)
            distanza_manomanopiedesx = angle_between(linea,7,11,15)
            distanza_kneehippiedesx = angle_between(linea,13,14,15)
            distanza_kneehippiededx = angle_between(linea,19,18,17)


            features.append(distanza_ginocchioanklepiededx) #angolo ginocchio ankle piede dx
            features.append(distanza_ginocchioanklepiedesx) # angolo ginocchio ankle piede sx
            features.append(distanza_spallapolsomanosx) # angolo spalla polso mano sx
            features.append(distanza_spallapolsomanodx) # angolo spalla polso mano dx
            features.append(distanza_manohipmanodx) #angolo mano sinistra, hip sinistro e mano destra per capire se cammina
            features.append(distanza_manohipmanosx) #angolo mano destra, hip destro e mano sinistra per capire se cammina
            features.append(distanza_manomanopiededx) #angolo mano destra, mano sinistra e piede destro per capire se cammina
            features.append(distanza_manomanopiedesx) #angolo mano destra, mano sinistra e piede sinistro per capire se cammina
            features.append(distanza_kneehippiedesx) # angolo knee hip foot sx
            features.append(distanza_kneehippiededx) # angolo knee hip foot dx
            

            
            features.append(distanza_ginocchioanklepiededx - prev_ginocchioanklepiededx)
            prev_ginocchioanklepiededx = distanza_ginocchioanklepiededx

            features.append(distanza_ginocchioanklepiedesx - prev_ginocchioanklepiedesx)
            prev_ginocchioanklepiedesx = distanza_ginocchioanklepiedesx

            features.append(distanza_spallapolsomanosx - prev_spallapolsomanosx)
            prev_spallapolsomanosx = distanza_spallapolsomanosx

            features.append(distanza_spallapolsomanodx - prev_spallapolsomanodx)
            prev_spallapolsomanodx = distanza_spallapolsomanodx

            features.append(distanza_manohipmanodx - prev_manohipmanodx)
            prev_manohipmanodx = distanza_manohipmanodx

            features.append(distanza_manohipmanosx - prev_manohipmanosx)
            prev_manohipmanosx = distanza_manohipmanosx

            features.append(distanza_manomanopiededx - prev_manomanopiededx)
            prev_manomanopiededx = distanza_manomanopiededx

            features.append(distanza_manomanopiedesx - prev_manomanopiedesx)
            prev_manomanopiedesx = distanza_manomanopiedesx

            features.append(distanza_kneehippiedesx - prev_kneehippiedesx)
            prev_kneehippiedesx = distanza_kneehippiedesx

            features.append(distanza_kneehippiededx - prev_kneehippiededx)
            prev_kneehippiededx = distanza_kneehippiededx
            for v in range(0,20): # distanza tra testa e joints
                    if(v!=3):
                        features.append(get_distance(linea,4,v))
            
            features.append(get_distance(linea,15,19))
            features.append(get_distance(linea,7,11))
            features.append(get_distance(linea,6,10))# distanza gomiti
            features.append(get_distance(linea,18,14))# distanza ginocchia
            features.append(get_distance(linea,18,11))# distanza ginocchia mano per capire se è seduto
            features.append(get_distance(linea,14,7))# distanza ginocchia mano per capire se è seduto
            #features.append(get_distance(linea[1:],3,7)) # distanza mano testa
            #features.append(get_distance(linea[1:],3,11)) # distanza mano testa
            features.append(get_distance(linea,5,6)) # distanza spalla gomito sx
            features.append(get_distance(linea,9,10)) # distanza spalla gomito dx
            features.append(get_distance(linea,3,1)) # distanza spine hip center
            features.append(get_distance(linea,7,17)) # mano sinistra hip destro
            features.append(get_distance(linea,11,13)) # mano destra hip sinistro


            





            a = get_distance(linea,19,15)
            b = get_distance(linea,14,18)
            features.append(a/b) # se è seduto o in piedi
            a = get_distance(linea,3,1)
            b = get_distance(linea,10,6)
            features.append(a/b) # se è seduto o in piedi

            

            
            #calcolare centro di gravità #segmental method
            trunk = get_middlepoint(linea,0,1,0.449) # location centro di gravità % di lunghezza
            upperarmR = get_middlepoint(linea,8,9,0.557)
            upperarmL = get_middlepoint(linea,4,5,0.557)
            lowerarmR = get_middlepoint(linea,5,6,0.557) 
            lowerarmL = get_middlepoint(linea,9,10,0.457)
            tighR = get_middlepoint(linea,12,13,0.41) 
            tighL = get_middlepoint(linea,16,17,0.41) 
            shankR = get_middlepoint(linea,17,18,0.446) 
            shankL = get_middlepoint(linea,13,14,0.446) 



            trunkx = trunk[0] * 43.46 # massa
            trunky = trunk[1] * 43.46
            trunkz = trunk[2] * 43.46
            upperarmRx = upperarmR[0]*5.42
            upperarmRy = upperarmR[1]*5.42
            upperarmRz = upperarmR[2]*5.42

            upperarmLx = upperarmL[0] *5.42
            upperarmLy = upperarmL[1] *5.42
            upperarmLz = upperarmL[2] *5.42

            lowerarmRx = lowerarmR[0] *3.24
            lowerarmRy = lowerarmR[1] *3.24
            lowerarmRz = lowerarmR[2] *3.24
            lowerarmLx = lowerarmL[0]*3.24
            lowerarmLy = lowerarmL[1]*3.24
            lowerarmLz = lowerarmL[2]*3.24
            tighRx = tighR[0]*28.32
            tighRy = tighR[1]*28.32
            tighRz = tighR[2]*28.32
            tighLx = tighL[0]*28.32
            tighLy = tighL[1]*28.32
            tighLz = tighL[2]*28.32
            shankRx = shankR[0]* 8.66
            shankRy = shankR[1]* 8.66
            shankRz = shankR[2]* 8.66
            shankLx = shankL[0]* 8.66
            shankLy = shankL[1]* 8.66
            shankLz = shankL[2]* 8.66

            centro_gravitax = (trunkx+upperarmLx+upperarmRx+lowerarmLx+lowerarmRx+tighRx+tighLx+shankLx+shankRx) / 100
            centro_gravitay = (trunky+upperarmLy+upperarmRy+lowerarmLy+lowerarmRy+tighRy+tighLy+shankLy+shankRy) / 100
            centro_gravitaz = (trunkz+upperarmLz+upperarmRz+lowerarmLz+lowerarmRz+tighRz+tighLz+shankLz+shankRz) / 100
            features.append(centro_gravitax)
            features.append(centro_gravitay)
            features.append(centro_gravitaz)
    

            cross_limb_distanceLR = distance_with_points_avg(average(linea,6,5,7),average(linea,18,17,19))
            cross_limb_distanceRL = distance_with_points_avg(average(linea,10,9,11),average(linea,14,13,15))
            cldr = 1/(1+cross_limb_distanceLR - cross_limb_distanceRL) # controlla oscillazione mano gambe per camminata

            features.append(cldr)
            if(count < len(azione)-1):
                next_linea = azione[count+1]
            else:
                next_linea = azione[count]

            LOA_arm_left = (get_distance_frames(linea,next_linea,6) / 3) + (get_distance_frames(linea,next_linea,5) / 3) + (get_distance_frames(linea,next_linea,7) / 3) # LOA su spalla gomito mano sx
            LOA_arm_right = (get_distance_frames(linea,next_linea,10) / 3) + (get_distance_frames(linea,next_linea,9) / 3) + (get_distance_frames(linea,next_linea,11) / 3) # LOA su spalla gomito mano dx
            LOA_leg_left = (get_distance_frames(linea,next_linea,14) / 3) + (get_distance_frames(linea,next_linea,13) / 3) + (get_distance_frames(linea,next_linea,15) / 3) # LOA su hip knee piede sx
            LOA_leg_right = (get_distance_frames(linea,next_linea,18) / 3) + (get_distance_frames(linea,next_linea,17) / 3) + (get_distance_frames(linea,next_linea,19) / 3) # LOA su hip knee piede dx

            features.append(LOA_arm_left)
            features.append(LOA_arm_right)
            features.append(LOA_leg_left)
            features.append(LOA_leg_right)

            features.append((get_distance(linea,1,7) + get_distance(linea,1,11) + get_distance(linea,1,15)+ get_distance(linea,1,19)) / 4) # distana media tra hip e mani/piedi

            fine_frame.append(features)
            features = []
        actions.append(fine_frame)
        fine_frame = []

#distanza tra la media di due punti
def distance_with_points_avg(a,b):
    d = pow((pow((b[0] - a[0]),2) + pow((b[1]-a[1]),2) + pow((b[2]-a[2]), 2)),0.5)
    return d

#media fra tre punti
def average(frame,a,b,c):
    res = []
    x = (float(frame[a*3]) + float(frame[b*3]) + float(frame[c*3])) / 3
    y = (float(frame[a*3+1]) + float(frame[b*3+1]) + float(frame[c*3+1])) / 3
    z = (float(frame[a*3+2]) + float(frame[b*3+2]) + float(frame[c*3+2])) / 3
    res.append(x)
    res.append(y)
    res.append(z)
    return res
    
#punto a metà tra a ed b
def get_middlepoint(frame,a,b,perc):
    return (float(frame[a*3]) + float(frame[b*3])) * perc, (float(frame[a*3+1]) + float(frame[b*3+1])) * perc,(float(frame[a*3+2]) + float(frame[b*3+2])) * perc
#distanza tra a e b
def get_distance(frame,a,b):
    d = pow((pow((float(frame[b*3]) - float(frame[a*3])),2) + pow((float(frame[b*3+1])-float(frame[a*3+1])),2) + pow((float(frame[b*3+2])-float(frame[a*3+2])), 2)),0.5)
    return d
#distanza di un punto col frame adiacente
def get_distance_frames(frame,frame_next,a):
    d = pow((pow((float(frame_next[a*3]) - float(frame[a*3])),2) + pow((float(frame_next[a*3+1])-float(frame[a*3+1])),2) + pow((float(frame_next[a*3+2])-float(frame[a*3+2])), 2)),0.5)
    return d

def calculate_area(a,b,c):  
    s = (a + b + c) / 2   
    area = (s*(s-a) * (s-b)*(s-c)) ** 0.5        
    return area



def angle_between(frame, o, m, n):
    v_distale = (float(frame[n * 3]) - float(frame[m * 3]), float(frame[n * 3 + 1]) - float(frame[m * 3 + 1]), float(frame[n * 3 + 2]) - float(frame[m * 3 + 2]))
    v_intermedio = (float(frame[m * 3]) - float(frame[o * 3]), float(frame[m * 3 + 1]) - float(frame[o * 3 + 1]), float(frame[m * 3 + 2]) - float(frame[o * 3 + 2]))

    dot_prod = v_distale[0] * v_intermedio[0] + v_distale[1] * v_intermedio[1] +v_distale[2] * v_intermedio[2]
    norma_distale = math.sqrt(pow(float(v_distale[0]), 2) + pow(float(v_distale[1]), 2) + pow(float(v_distale[2]), 2))
    norma_intermedia = math.sqrt(pow(float(v_intermedio[0]), 2) + pow(float(v_intermedio[1]), 2) + pow(float(v_intermedio[2]), 2))
    if(norma_distale * norma_intermedia <=0):
        return 0.0
    angle = math.acos(dot_prod / (norma_distale * norma_intermedia))
    return angle



def resize_length(n_chann,max):
    global actions
    temp = []
    for i in actions:
        diff = max - len(i)
        for j in range(0,diff):
            for k in range(0,n_chann): #numero feature
                temp.append(0)
            i.insert(0,temp)
            temp = []
    return actions

def to_Train(n_chann):
    loop_actions()
    global actions
    global labels
    x = 0
    two_actions = []
    max = 0
    for i in actions:
        if(len(i) == 0): # toglie azioni con piu persone
            two_actions.append(x)
        if(len(i) > max):
             max=len(i)
        x+=1
    count = 0
    for b in two_actions:        
        actions.pop(b-count)
        #labels.pop(b-count)
        count+=1

    print(str(len(actions)) + "   " + str(len(labels)))
    print("max " + str(max))



    actions = [[[float(k) for k in i] for i in j] for j in actions]
    a_action = []
    a_label = []
    
    for a,l in zip(actions,labels):
        azione = a
        for i in range(2):
            noise = numpy.random.normal(0,0.1,[len(azione),len(azione[0])])
            res = azione + noise
            res = res.tolist()
            a_action.append(res)
            a_label.append(l)
            el = res

    actions += a_action
    labels += a_label

    actions = resize_length(n_chann,max) # aggiunta di zeri all'inizio

    actions,labels = shuffle(actions,labels)
    #train_set, test_set, train_labels, test_labels = model_selection.train_test_split(actions, labels, train_size=0.80,test_size=0.20, random_state=101)
    #"""
    train_size = int(len(actions)/100 *80)
    test_size = int(len(actions) - train_size)

    test_set = actions[train_size:]
    test_labels = labels[train_size:]
  

    train_set = actions[:train_size]
    train_labels = labels[:train_size]



    train_set,test_set, train_labels,test_labels = convert_to_pytorch_tensors(train_set, test_set, train_labels, test_labels)
    return train_set, test_set, train_labels, test_labels
    
def convert_to_pytorch_tensors(train_set, test_set, train_labels, test_labels):
    train_labels, test_labels = numpy.array(train_labels), numpy.array(test_labels)
    #train_set, test_set = numpy.array(train_set), numpy.array(test_set)
    train_labels, test_labels = train_labels - 1, test_labels - 1

    train_set,test_set= torch.FloatTensor(train_set),torch.FloatTensor(test_set),

    
    #train_set, test_set = torch.from_numpy(train_set), torch.from_numpy(test_set)
    train_labels, test_labels = torch.from_numpy(train_labels), torch.from_numpy(test_labels)

    train_set, test_set = train_set.type(torch.FloatTensor), test_set.type(torch.FloatTensor)
    train_labels, test_labels = train_labels.type(torch.LongTensor), test_labels.type(torch.LongTensor)
  
    return train_set, test_set, train_labels, test_labels



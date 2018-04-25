import os 
from PIL import Image
import numpy as np
import pandas as pd

debug = False
shuffle = False

label_dict = {"go": 0, "goforward":1, "goleft":2, "warning":3, "warningleft":4, "stop":5, "stopleft":6}

count = 0

# convert Day footage
for day in os.listdir('./data/dayTrain/'):
    images = []
    labels = []
    # get the box and bulb markers
    box = pd.read_csv('./Annotations/dayTrain/' + day + '/frameAnnotationsBOX.csv', delimiter=';')
    bulb = pd.read_csv('./Annotations/dayTrain/' + day + '/frameAnnotationsBULB.csv', delimiter=';')
    
    # Concat them, sort values by filename, drop unneccessary columns and reset index
    boxes = pd.concat([box, bulb]).sort_values(by=['Filename']).drop(['Origin file', 
                      'Origin frame number', 'Origin track', 'Origin track frame number'], axis=1)
    boxes = boxes.reset_index(drop=True)
    
    # convert annotation to number and clean Filename
    for i, row in boxes.iterrows():
        boxes.loc[i, 'Annotation tag'] = label_dict[row['Annotation tag'].lower().lstrip()]
        boxes.loc[i, 'Filename'] = row['Filename'].replace('dayTraining/', '')
        
    # open Images and store labels in numpy array
    for filename in os.listdir('./data/dayTrain/' + day + '/frames'):
        img = np.asarray(Image.open('./data/dayTrain/' + day + '/frames/' + filename), dtype=np.uint8)
        images.append(img)
        labels.append(np.array(boxes.loc[boxes['Filename'] == filename].drop('Filename', axis = 1)))
        
    # collect all footage into one big numpy array (labels, images)   
    imagesday = np.array(images)
    labelsday = np.array(labels)   
    np.savez(day, images=imagesday, boxes=labelsday)
    count += 1
    print(day, " completed")

count = 0

# convert Night footage
for night in os.listdir('./data/nightTrain/'):
    images = []
    labels = []
    # get the box and bulb markers
    box = pd.read_csv('./Annotations/nightTrain/' + night + '/frameAnnotationsBOX.csv', delimiter=';')
    bulb = pd.read_csv('./Annotations/nightTrain/' + night + '/frameAnnotationsBULB.csv', delimiter=';')
    
    # Concat them, sort values by filename, drop unneccessary columns and reset index
    boxes = pd.concat([box, bulb]).sort_values(by=['Filename']).drop(['Origin file', 
                      'Origin frame number', 'Origin track', 'Origin track frame number'], axis=1)
    boxes = boxes.reset_index(drop=True)
    
    # convert annotation to number and clean Filename
    for i, row in boxes.iterrows():
        boxes.loc[i, 'Annotation tag'] = label_dict[row['Annotation tag'].lower().lstrip()]
        boxes.loc[i, 'Filename'] = row['Filename'].replace('nightTraining/', '')
        
    # open Images and store labels in numpy array
    for filename in os.listdir('./data/nightTrain/' + night + '/frames'):
        img = np.asarray(Image.open('./data/nightTrain/' + night + '/frames/' + filename), dtype=np.uint8)
        images.append(img)
        labels.append(np.array(boxes.loc[boxes['Filename'] == filename].drop('Filename', axis = 1)))
        
    # collect all footage into one big numpy array (labels, images)   
    imagesnight = np.array(images)
    labelsnight = np.array(labels)   
    np.savez(night, images=imagesnight, boxes=labelsnight)
    count += 1
    print(night, " completed")


dayClip1 = np.load('./data/progressed/dayClip1.npz')
dayClip2 = np.load('./data/progressed/dayClip2.npz')
dayClip3 = np.load('./data/progressed/dayClip3.npz')
dayClip4 = np.load('./data/progressed/dayClip4.npz')
dayClip5 = np.load('./data/progressed/dayClip5.npz')
dayClip6 = np.load('./data/progressed/dayClip6.npz')
dayClip7 = np.load('./data/progressed/dayClip7.npz')
dayClip8 = np.load('./data/progressed/dayClip8.npz')
dayClip9 = np.load('./data/progressed/dayClip9.npz')
dayClip10 = np.load('./data/progressed/dayClip10.npz')
dayClip11 = np.load('./data/progressed/dayClip11.npz')
dayClip12 = np.load('./data/progressed/dayClip12.npz')
dayClip13 = np.load('./data/progressed/dayClip13.npz')
nightClip1 = np.load('./data/progressed/nightClip1.npz')
nightClip2 = np.load('./data/progressed/nightClip2.npz')
nightClip3 = np.load('./data/progressed/nightClip3.npz')
nightClip4 = np.load('./data/progressed/nightClip4.npz')
nightClip5 = np.load('./data/progressed/nightClip5.npz')

first = np.concatenate((dayClip1['images'], dayClip2['images'], dayClip3['images'], 
                         dayClip4['images'], dayClip5['images'], dayClip6['images'], 
                         dayClip7['images'], dayClip8['images'], dayClip9['images']), axis=0)
second = np.concatenate((nightClip1['images'], nightClip2['images'], nightClip3['images'], 
                         nightClip4['images'], nightClip5['images'], dayClip10['images'], 
                         dayClip11['images'], dayClip12['images'], dayClip13['images']), axis=0)
firstlab = np.concatenate((dayClip1['boxes'], dayClip2['boxes'], dayClip3['boxes'], 
                           dayClip4['boxes'], dayClip5['boxes'], dayClip6['boxes'], 
                           dayClip7['boxes'], dayClip8['boxes'], dayClip9['boxes']), axis=0)
secondlab = np.concatenate((nightClip1['boxes'], nightClip2['boxes'], nightClip3['boxes'], 
                            nightClip4['boxes'], nightClip5['boxes'], dayClip10['boxes'], 
                            dayClip11['boxes'], dayClip12['boxes'], dayClip13['boxes']), axis=0)


images = np.concatenate((first, second), axis=0)
labels = np.concatenate((firstLab, secondLab), axis=0)

np.savez("./data/progressed/full_data", images=images, boxes=labels)
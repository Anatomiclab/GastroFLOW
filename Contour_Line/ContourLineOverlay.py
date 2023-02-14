# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from PIL import Image
from PIL import ImageDraw, ImageFont
from os import walk

f=[]
path=r'./PredictedData/'
for _,_,files in walk(path):
    for i in files:
        if i[-3:]=='csv':
            f+=[i]

for file in f:
    test=pd.read_csv(path+file)
    test=test.drop(['Nucleus: Area', 'Nucleus: Perimeter',\
           'Nucleus: Circularity', 'Nucleus: Max caliper', 'Nucleus: Min caliper',\
           'Nucleus: Eccentricity', 'Nucleus: Hematoxylin OD mean',\
           'Nucleus: Hematoxylin OD sum', 'Nucleus: Hematoxylin OD std dev',\
           'Nucleus: Hematoxylin OD max', 'Nucleus: Hematoxylin OD min',\
           'Nucleus: Hematoxylin OD range', 'Nucleus: Eosin OD mean',\
           'Nucleus: Eosin OD sum', 'Nucleus: Eosin OD std dev',\
           'Nucleus: Eosin OD max', 'Nucleus: Eosin OD min',\
           'Nucleus: Eosin OD range', 'Cell: Area', 'Cell: Perimeter',\
           'Cell: Circularity', 'Cell: Max caliper', 'Cell: Min caliper',\
           'Cell: Eccentricity', 'Cell: Hematoxylin OD mean',\
           'Cell: Hematoxylin OD std dev', 'Cell: Hematoxylin OD max',\
           'Cell: Hematoxylin OD min', 'Cell: Eosin OD mean',\
           'Cell: Eosin OD std dev', 'Cell: Eosin OD max', 'Cell: Eosin OD min',\
           'Cytoplasm: Hematoxylin OD mean', 'Cytoplasm: Hematoxylin OD std dev',\
           'Cytoplasm: Hematoxylin OD max', 'Cytoplasm: Hematoxylin OD min',\
           'Cytoplasm: Eosin OD mean', 'Cytoplasm: Eosin OD std dev',\
           'Cytoplasm: Eosin OD max', 'Cytoplasm: Eosin OD min',\
           'Nucleus/Cell area ratio'], axis=1)
    
    scatter=test[test['prediction']>.8]
    
    Xc=scatter["Centroid X Âµm"].values
    Yc=scatter["Centroid Y Âµm"].values
    
    X, Y = np.mgrid[min(Xc):max(Xc):200j, min(Yc):max(Yc):200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([Xc, Yc])                                                                        
    kernel = stats.gaussian_kde(values)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)
    maxZ=max(b for a in Z for b in a)
    
    fig, ax = plt.subplots(figsize=(20, 20))                   

    # Add contour lines
    plt.gca().invert_yaxis()
    #levels=[.5*maxZ,.8*maxZ]
    print(maxZ)
    img = plt.contour(X, Y, Z / maxZ,levels=[0.75,0.9]) # 0.9,0.75
    #img=plt.contour(X, Y, Z/maxZ)
    #plt.clabel(img,inline=True,fontsize=20)
    #plt.savefig(r'./Contour_line//2line_'+file.replace('csv','png'))
    plt.close()
    
    img2 = Image.open(r'./Contour_Line/original/'+file.replace('csv','png')).convert('L').convert('RGB')
    draw = ImageDraw.Draw(img2)
    i=0
    #print(img.collections)
    for lr in img.collections[::-1]:
        if lr.get_paths()==[]:
            pass
        else:
            if i==0:
                for lc in lr.get_paths():
                    X=(lc.vertices[:,0])/2
                    Y=(lc.vertices[:,1])/2
                    Y = list(map(int, Y))
                    X = list(map(int, X))
                    XY=list(zip(X,Y))
                    draw.line(XY, fill='red', width=10,joint="curve")
                i+=1
            elif i==1:
                for lc in lr.get_paths():
                    X=(lc.vertices[:,0])/2
                    Y=(lc.vertices[:,1])/2
                    Y = list(map(int, Y))
                    X = list(map(int, X))
                    XY=list(zip(X,Y))
                    draw.line(XY, fill='yellow', width=10,joint="curve")
                i+=1
    
    
    #img2=img2.rotate(270, expand=True)
    sizeX=img2.size[0]
    sizeY=img2.size[1]
    draw1 = ImageDraw.Draw(img2)    
    img2.save('./ExportImage2/'+file.replace('csv','png'))



# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:47:01 2021

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from PIL import Image
from PIL import ImageDraw
from os import walk
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm

f=[]
path=r'/home/htihe/Gastric_Flow_verify/Gastric_Flow/Code_Integration/Contour_Line/PredictedData/FN//'
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
    Z=Z/max(b for a in Z for b in a)
    
    fig, ax = plt.subplots(figsize=(20, 20))                   
    
    plt.gca().invert_yaxis()
    levels = np.linspace(0,1,100)
    img=plt.contourf(X, Y, Z,cmap='viridis',levels=levels)
    plt.colorbar(img)
                                           
    lvl_lookup = dict(zip(img.collections, img.levels))
    
    
    img2 = Image.open(r'/home/htihe/Gastric_Flow_verify/Gastric_Flow/Code_Integration/Contour_Line/original/FN/'+file.replace('csv','png')).convert("RGBA")
    canva = Image.new("RGBA", img2.size,'#FFFFFF00')
    draw = ImageDraw.Draw(canva, "RGBA")

    PolyList=[]
    for col in img.collections:
        z=lvl_lookup[col] 
        for contour_path in col.get_paths():
            for ncp,cp in enumerate(contour_path.to_polygons()):
                X=np.vstack(cp)[:,0]/2
                Y=np.vstack(cp)[:,1]/2
                if z > 0:
                    draw.polygon(list(zip(X,Y)), tuple(list(int(x*225) for x in cm.get_cmap('viridis')(z))[:3]+[125]))
                else:
                    pass
    img2.paste(canva,None,canva)
    img2.save('/home/htihe/Gastric_Flow_verify/Gastric_Flow/Code_Integration/Contour_Line/ExportHeatmap/FN/'+file.replace('csv','png'))
    plt.close()





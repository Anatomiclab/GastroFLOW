import glob

import pandas as pd
from os import walk


path=r'./RawData'
outPath=r'./aggregateddata.csv'
empty_frame = pd.DataFrame()
file = []
columns =['id', 'Nucleus: Area', 'Nucleus: Perimeter', 'Nucleus: Circularity', 'Nucleus: Max caliper',
       'Nucleus: Min caliper', 'Nucleus: Eccentricity', 'Nucleus: Hematoxylin OD mean', 'Nucleus: Hematoxylin OD sum',
       'Nucleus: Hematoxylin OD std dev', 'Nucleus: Hematoxylin OD max', 'Nucleus: Hematoxylin OD min', 'Nucleus: Hematoxylin OD range',
       'Nucleus: Eosin OD mean', 'Nucleus: Eosin OD sum', 'Nucleus: Eosin OD std dev', 'Nucleus: Eosin OD max',
       'Nucleus: Eosin OD min', 'Nucleus: Eosin OD range', 'Cell: Area', 'Cell: Perimeter',
       'Cell: Circularity', 'Cell: Max caliper', 'Cell: Min caliper',
       'Cell: Eccentricity', 'Cell: Hematoxylin OD mean', 'Cell: Hematoxylin OD std dev',
       'Cell: Hematoxylin OD max', 'Cell: Hematoxylin OD min', 'Cell: Eosin OD mean', 'Cell: Eosin OD std dev',
       'Cell: Eosin OD max', 'Cell: Eosin OD min', 'Cytoplasm: Hematoxylin OD mean',
       'Cytoplasm: Hematoxylin OD std dev', 'Cytoplasm: Hematoxylin OD max', 'Cytoplasm: Hematoxylin OD min',
       'Cytoplasm: Eosin OD mean', 'Cytoplasm: Eosin OD std dev', 'Cytoplasm: Eosin OD max',
       'Cytoplasm: Eosin OD min', 'Nucleus/Cell area ratio']

print(path+'/*')
for filename in glob.glob(path+'/*'):
        print(filename)
        if filename[-3:] == 'txt':
            file.append(filename.split("/")[-1])

error=[]

for f in file:
        temp_df = pd.read_csv(path+f, sep='\t',encoding="ISO-8859–1")
        print(temp_df.columns)
        if list(temp_df.columns) == ['Image','Name','Class', 'Parent', 'ROI', 'Centroid X Âµm', 'Centroid Y Âµm', 'Nucleus: Area', 'Nucleus: Perimeter', 'Nucleus: Circularity', 'Nucleus: Max caliper',
           'Nucleus: Min caliper', 'Nucleus: Eccentricity', 'Nucleus: Hematoxylin OD mean', 'Nucleus: Hematoxylin OD sum',
           'Nucleus: Hematoxylin OD std dev', 'Nucleus: Hematoxylin OD max', 'Nucleus: Hematoxylin OD min', 'Nucleus: Hematoxylin OD range',
           'Nucleus: Eosin OD mean', 'Nucleus: Eosin OD sum', 'Nucleus: Eosin OD std dev', 'Nucleus: Eosin OD max',
           'Nucleus: Eosin OD min', 'Nucleus: Eosin OD range', 'Cell: Area', 'Cell: Perimeter',
           'Cell: Circularity', 'Cell: Max caliper', 'Cell: Min caliper',
           'Cell: Eccentricity', 'Cell: Hematoxylin OD mean', 'Cell: Hematoxylin OD std dev',
           'Cell: Hematoxylin OD max', 'Cell: Hematoxylin OD min', 'Cell: Eosin OD mean', 'Cell: Eosin OD std dev',
           'Cell: Eosin OD max', 'Cell: Eosin OD min', 'Cytoplasm: Hematoxylin OD mean',
           'Cytoplasm: Hematoxylin OD std dev', 'Cytoplasm: Hematoxylin OD max', 'Cytoplasm: Hematoxylin OD min',
           'Cytoplasm: Eosin OD mean', 'Cytoplasm: Eosin OD std dev', 'Cytoplasm: Eosin OD max',
           'Cytoplasm: Eosin OD min', 'Nucleus/Cell area ratio']:
            columns_c = True
        else:
            columns_c = False
        
        temp_df=temp_df.drop(columns=['Image','Name','Class', 'Parent', 'ROI', 'Centroid X Âµm', 'Centroid Y Âµm'])
        temp_id=pd.Series({'id':f})
    
        temp_mean=temp_df.mean()
        temp_combined=temp_id.append(temp_df.mean())
        print(temp_df.columns)
    
        temp_combined.index = columns
        print(temp_combined)
        if columns_c == True:
            empty_frame=empty_frame.append(temp_combined,ignore_index=True)
        elif columns_c == False:
            empty_frame=empty_frame.append(temp_combined,ignore_index=True)
        


empty_frame=empty_frame[['id', 'Nucleus: Area', 'Nucleus: Perimeter', 'Nucleus: Circularity', 'Nucleus: Max caliper',
       'Nucleus: Min caliper', 'Nucleus: Eccentricity', 'Nucleus: Hematoxylin OD mean', 'Nucleus: Hematoxylin OD sum',
       'Nucleus: Hematoxylin OD std dev', 'Nucleus: Hematoxylin OD max', 'Nucleus: Hematoxylin OD min', 'Nucleus: Hematoxylin OD range',
       'Nucleus: Eosin OD mean', 'Nucleus: Eosin OD sum', 'Nucleus: Eosin OD std dev', 'Nucleus: Eosin OD max',
       'Nucleus: Eosin OD min', 'Nucleus: Eosin OD range', 'Cell: Area', 'Cell: Perimeter',
       'Cell: Circularity', 'Cell: Max caliper', 'Cell: Min caliper',
       'Cell: Eccentricity', 'Cell: Hematoxylin OD mean', 'Cell: Hematoxylin OD std dev',
       'Cell: Hematoxylin OD max', 'Cell: Hematoxylin OD min', 'Cell: Eosin OD mean', 'Cell: Eosin OD std dev',
       'Cell: Eosin OD max', 'Cell: Eosin OD min', 'Cytoplasm: Hematoxylin OD mean',
       'Cytoplasm: Hematoxylin OD std dev', 'Cytoplasm: Hematoxylin OD max', 'Cytoplasm: Hematoxylin OD min',
       'Cytoplasm: Eosin OD mean', 'Cytoplasm: Eosin OD std dev', 'Cytoplasm: Eosin OD max',
       'Cytoplasm: Eosin OD min', 'Nucleus/Cell area ratio']]
empty_frame.to_csv(outPath, index=False)


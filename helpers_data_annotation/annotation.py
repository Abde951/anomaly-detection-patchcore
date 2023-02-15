import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from matplotlib import image
import timeit

start = timeit.default_timer()

df = pd.read_excel('all_data_scores.xlsx')
images = df[['images_name','labels_gt','predictions']]

classes = ['good','deformed']
fods = ['NoFod','Fod']
colors = ['red','blue']
imgs = []
path = "./bin/mvtec/client_carbon/EmptyFabric/test"
valid_images = [".jpg",".gif",".png",".tga"]
i = 0
size = len(images)
for index, row in images.iterrows():
    
    classe = classes[int(row['labels_gt'])]
    color = colors[int(row['labels_gt']) == int(row['predictions'])]
    prediction = fods[int(row['predictions'])]
    label = fods[int(row['labels_gt'])]
    f = row['images_name']
    img_name = f'/{classe}/0004_{f}.png'
    file_path = path + img_name
    img = image.imread(file_path)

    plt.imshow(img)
    plt.axis('off')
    title = f'Label: {label} | Prediction: {prediction}'
    plt.title(title,fontsize='16', color=color)
    plt.savefig(f'./images_saved/EmptyFabric/0004_{f}.png', dpi=150,facecolor='white')
    i = i + 1
    print('progress : ',i*100/size,'%')

stop = timeit.default_timer()

print('Time: ', stop - start) 
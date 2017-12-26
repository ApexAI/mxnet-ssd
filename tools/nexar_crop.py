from PIL import Image
import numpy as np
from PIL import ImageDraw
from PIL import ImageFont
import json
import os
import random
import math
import argparse


parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

parser.add_argument('--anno_path', dest='anno_path', help='nexar annotations folder',
                    default="/media/tapir/Data/Thesis/Datasets/nexar/annotations", type=str)
parser.add_argument('--images_path', dest='images_path', help='path that contains original nexar images',
                    default="/media/tapir/Data/Thesis/Datasets/nexar/images", type=str)
parser.add_argument('--output_folder', dest='output_folder', help='folder to contain cropped nexar images',
                    default="/media/tapir/Data/Thesis/Datasets/nexar/nexar_cropped_noisy", type=str)
parser.add_argument('--add_noise', dest='add_noise', help='random resize bounding boxes',
                    default=True, type=bool)
parser.add_argument('--validation', dest='validation', help='If set to true, only the last 10k images will be used',
                    default=False, type=bool)

pargs = parser.parse_args()

start,end= (40000,50000) if(pargs.validation) else (0,40000)

widths=[]
heights=[]

images_with_lights=0
num_total_lights=0

counter={"tl_green":0,"tl_yellow":0,"tl_red":0,"none":0}

labels={"tl_green":0,"tl_yellow":1,"tl_red":2,"none":3}
tags=["tl_green","tl_yellow","tl_red"]
lst_data=""
empty_count=0
for i in range(start,end):
    annofile=pargs.anno_path+"/"+str(i)+".json"
    imagefile=pargs.images_path+"/"+str(i)+".jpg"
    num_lights = 0
    if os.path.isfile(annofile):
        anno = json.load(open(annofile))
        num_lights=len(anno)

    if num_lights>0 and os.path.isfile(imagefile):
        image = Image.open(imagefile)
        im_w,im_h=image.size


        for idx_light in range (num_lights):
            x0=anno[idx_light]['type_representation']['x0']
            y0=anno[idx_light]['type_representation']['y0']
            x1=anno[idx_light]['type_representation']['x1']
            y1=anno[idx_light]['type_representation']['y1']
#
            width=x1-x0
            height=y1-y0
            if(pargs.add_noise):
                x0=random.randrange(x0-2*width,x0+width//4)
                x1=random.randrange(x1-width//4,x1+2*width)
                y0=random.randrange(y0-2*height,y0+height//4)
                y1=random.randrange(y1-height//4,y1+2*height)

                x0 =max(x0,0)
                x1 =min(x1,im_w-1)
                y0 =max(y0,0)
                y1 =min(y1,im_h-1)

            #widths.append(x1-x0)
            #heights.append(y1-y0)

            res=image.crop((x0,y0,x1,y1))
            res.save(pargs.output_folder+"/{}_{}.jpg".format(i,idx_light))
            tag=anno[idx_light]['class_name']['detected_object_class']['tag']
            counter[tag]+=1
            lst_data+=str(num_total_lights)+"\t"+ str(labels[tag])  +"\t"+"{}_{}.jpg".format(i,idx_light)+"\n"

            num_total_lights+=1

        images_with_lights+=1
    elif num_lights==0 and os.path.isfile(imagefile):
        if(counter["none"]%3==0):
            image = Image.open("/media/tapir/Data/Thesis/Datasets/nexar/images/"+str(i)+".jpg")

            x0= random.randrange(0,image.width-100)
            x1= x0 + int(math.floor(random.normalvariate(40,7)))
            y0 = random.randrange(0, image.height - 100)
            y1 = y0 + int(math.floor(random.normalvariate(60, 7)))
            res = image.crop((x0, y0, x1, y1))
            res.save("/media/tapir/Data/Thesis/Datasets/nexar/nexar_cropped_noisy/{}_none.jpg".format(i))

            lst_data += str(num_total_lights) + "\t" + str(labels["none"]) + "\t" + "{}_none.jpg".format(i) + "\n"
            num_total_lights += 1
        counter["none"]+=1


mode= "val" if pargs.validation else "train"
list_file=pargs.output_folder+"/nexar_"+mode+".lst"
with open(list_file, "w") as text_file:
    text_file.write(lst_data)
print(num_total_lights)


#print("mean width:")
#print(np.mean(widths))
#print("variance width:")
#print(np.var(widths))
#print("stdev width:")
#print(np.std(widths))
#print("---------------------------")
#print("mean height:")
#print(np.mean(heights))
#print("variance height:")
#print(np.var(heights))
#print("stdev height:")
#print(np.std(heights))

#print("total num of lights: %d"%num_total_lights)
#print("total num of images with lights: %d"%images_with_lights)

for key,val in counter.iteritems():
    print("number of {}: {} ".format(key,val))
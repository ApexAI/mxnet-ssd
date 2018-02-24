import csv
import requests
import argparse
import os

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
parser.add_argument('--set', dest='set', help='train or val or test',
                    default="train", type=str)
parser.add_argument('--root', dest='root', help='root path',
                    default="/mnt/data/dataset_stuff/openimages/", type=str)


pargs = parser.parse_args()



classes={}

classes['car']          ='/m/0k4j'
classes['bus']          ='/m/01bjv'
classes['van']          ='/m/0h2r6'
classes['limousine']    ='/m/01lcw4'
classes['taxi']         ='/m/0pg52'
classes['ambulance']    ='/m/012n7d'




counter={}

counter[classes['car']         ]=0
#counter[classes['vehicle']     ]=0
#counter[classes['land vehicle']]=0
counter[classes['bus']         ]=0
counter[classes['van']         ]=0
counter[classes['limousine']   ]=0
counter[classes['taxi']        ]=0

image_counter={}

image_counter[classes['car']         ]=0
#image_counter[classes['vehicle']     ]=0
#image_counter[classes['land vehicle']]=0
image_counter[classes['bus']         ]=0
image_counter[classes['van']         ]=0
image_counter[classes['limousine']   ]=0
image_counter[classes['taxi']        ]=0


labels={}

def print_stats():
    print  'number of bounding boxes for car: ' + str(counter[classes['car']])
    #print  'number of bounding boxes for vehicle: ' + str(counter[classes['vehicle']])
    #print  'number of bounding boxes for land vehicle: ' + str(counter[classes['land vehicle']])
    print  'number of bounding boxes for bus: ' + str(counter[classes['bus']])
    print  'number of bounding boxes for van: ' + str(counter[classes['van']])
    print  'number of bounding boxes for limousine: ' + str(counter[classes['limousine']])
    print  'number of bounding boxes for taxi: ' + str(counter[classes['taxi']])

    print '_________________________________________________________________________________'

    print  'number of images that contain car: ' + str(image_counter[classes['car']])
    #print  'number of images that contain vehicle: ' + str(image_counter[classes['vehicle']])
    #print  'number of images that contain land vehicle: ' + str(image_counter[classes['land vehicle']])
    print  'number of images that contain bus: ' + str(image_counter[classes['bus']])
    print  'number of images that contain van: ' + str(image_counter[classes['van']])
    print  'number of images that contain limousine: ' + str(image_counter[classes['limousine']])
    print  'number of images that contain taxi: ' + str(image_counter[classes['taxi']])

prev=''

anno_path=pargs.root+'anno/2017_11/'+pargs.set+'/annotations-human-bbox.csv'
img_path=pargs.root+'images/2017_11/'+pargs.set+'/images.csv'

with open(anno_path, 'r') as f:
    reader=csv.reader(f,delimiter=',')
    for row in reader:
        if row[2] in counter:

            counter[row[2]]+=1  #counting bounding boxes
            if(row[0]!=prev):
                image_counter[row[2]] += 1
                prev=row[0] #counting number of images

            bbox=(float(row[4]),float(row[6]),float(row[5]),float(row[7]))

            if row[0] in labels:    #adding the bounding box to the image
                labels[row[0]].append(bbox)
            else:
                labels[row[0]]=[bbox]

i=0
lst_data=""
save_path=pargs.root+'images/2017_11/'+pargs.set+'/'
with open(img_path, 'r') as f:
    reader=csv.reader(f,delimiter=',')
    existing_images=os.listdir(save_path)
    for row in reader:
        if row[0] in labels:
            try:
                path = save_path + row[0] + '.jpg'
                if row[0]+'.jpg' not in existing_images:
                    r = requests.get(row[10],allow_redirects=False)

                    if r.status_code != 200:
                        continue
                    open(path, 'wb').write(r.content)

                line = str(i) + "\t" + str(2) + "\t" + str(6) + "\t"
                for bbox in labels[row[0]]:
                    label = "0\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t0\t".format(bbox[0], bbox[1], bbox[2], bbox[3])
                    line += label
                line += path
                lst_data += line + "\n"
                i+=1
            except:
                print 'error getting image: '+ row[0] + ' '+ row[10]

with open('openImages_'+pargs.set+'.lst', "w") as text_file:
    text_file.write(lst_data)
#for val,key in labels.iteritems():
#    print ("{} : {} boxes".format(key,len(val)))

print_stats()

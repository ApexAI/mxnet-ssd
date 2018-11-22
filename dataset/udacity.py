import os
import csv
from PIL import Image
import random
import argparse


parser = argparse.ArgumentParser(description='filter openImages v4 dataset into an AD dataset')

parser.add_argument('--object_classes', dest='object_classes', help='comma separated list of classes', default="pedestrian", type=str)
parser.add_argument('--root', dest='root', help='root dir of the dataset', default="/mnt/data/dataset_stuff/udacity", type=str)
parser.add_argument('--out', dest='out', help='output lst name', default="udacity.lst", type=str)

pargs = parser.parse_args()

root=pargs.root


def populate_train(object_classes):
    labels={}

    a_dir=os.path.join(root,'object-dataset')
    a_labelfile=os.path.join(a_dir,"labels.csv")

    with open(a_labelfile,'r') as f:
        lines = f.readlines()

        for line in lines:
            row=line.split(' ')
            obj_cls=row[6].split("\"")[1].lower()
            if obj_cls in object_classes:
                image_path=os.path.join(a_dir,row[0])

                #with Image.open(image_path) as img:
                #    width, height = img.size
                label=(int(row[1]), int(row[2]),
                       int(row[3]), int(row[4]), obj_cls)    # xmin, ymin, xmax, ymax, class

                if image_path in labels:
                    labels[image_path].append(label)
                else:
                    labels[image_path]=[label]


    return labels



def populate_val(object_classes):
    labels={}


    #crowdai dataset
    c_dir=os.path.join(root,'object-detection-crowdai')
    c_labelfile=os.path.join(c_dir,"labels.csv")



    with open(c_labelfile,'r') as f:
        c_reader=csv.reader(f,delimiter=',')

        for row in c_reader:
            obj_cls=row[5].lower()
            if obj_cls in object_classes:
                image_path=os.path.join(c_dir,row[4])

                #with Image.open(image_path) as img:
                #    width, height = img.size
                label=(int(row[0]), int(row[1]),int(row[2]), int(row[3]), obj_cls)    # xmin, ymin, xmax, ymax, class

                if image_path in labels:
                    labels[image_path].append(label)
                else:
                    labels[image_path]=[label]

    return labels

def populate(object_classes=["car"]):
    labels={}


    #crowdai dataset
    c_dir=os.path.join(root,'object-detection-crowdai')
    c_labelfile=os.path.join(c_dir,"labels.csv")



    with open(c_labelfile,'r') as f:
        c_reader=csv.reader(f,delimiter=',')

        for row in c_reader:
            obj_cls=row[5].lower()
            if obj_cls in object_classes:
                image_path=os.path.join(c_dir,row[4])

                #with Image.open(image_path) as img:
                #    width, height = img.size
                label=(int(row[0]), int(row[1]),int(row[2]), int(row[3]), obj_cls)    # xmin, ymin, xmax, ymax, class

                if image_path in labels:
                    labels[image_path].append(label)
                else:
                    labels[image_path]=[label]


    #autti dataset
    a_dir=os.path.join(root,'object-dataset')
    a_labelfile=os.path.join(a_dir,"labels.csv")

    with open(a_labelfile,'r') as f:
        lines = f.readlines()

        for line in lines:
            row=line.split(' ')
            obj_cls=row[6].split("\"")[1].lower()
            if obj_cls in object_classes:
                image_path=os.path.join(a_dir,row[0])

                #with Image.open(image_path) as img:
                #    width, height = img.size
                label=(int(row[1]), int(row[2]),
                       int(row[3]), int(row[4]), obj_cls)    # xmin, ymin, xmax, ymax, class

                if image_path in labels:
                    labels[image_path].append(label)
                else:
                    labels[image_path]=[label]


    return labels




def pedestrian_classifier(add_noise=True):
    data_dict=populate(["car","truck","pedestrian"])

    lst_data=''
    label_map={'car':0,'truck':0,'pedestrian':1}

    lst_idx=0


    for key,val in data_dict.iteritems():
        image_file = key
        image = Image.open(image_file)
        im_w, im_h = image.size

        for box in val:
            obj_cls=box[4]

            #x0 =box[0]
            #y0 =box[1]
            #x1 =box[2]
            #y1 =box[3]



            #x0 = random.randrange(x0 - 2 * width, x0 + width // 4)
            #x1 = random.randrange(x1 - width // 4, x1 + 2 * width)
            #y0 = random.randrange(y0 - 2 * height, y0 + height // 4)
            #y1 = random.randrange(y1 - height // 4, y1 + 2 * height)

            x0 = max(box[0], 0)
            y0 = max(box[1], 0)
            x1 = min(box[2], im_w - 1)
            y1 = min(box[3], im_h - 1)
            width = x1 - x0
            height = y1 - y0

            if(width*height>1000):
                res = image.crop((x0, y0, x1, y1))
                save_path=os.path.join(root,"cropped_images",'{}.jpg'.format(lst_idx))
                res.save(save_path)


                lst_data+="{}\t{}\t{}\n".format(lst_idx,label_map[obj_cls], save_path)

                lst_idx+=1



    with open("udacity_test.lst",'w') as f:
        f.write(lst_data)



def object_detection_lst(fn_out, object_classes=["pedestrian"]):
    data_dict=populate(object_classes)
    lst_data=''
    label_map={'car':0,'truck':0,'pedestrian':1}

    lst_idx=0

    for key,val in data_dict.iteritems():
        image_file = key
        image = Image.open(image_file)
        im_w, im_h = image.size

        lst_line = str(lst_idx) + "\t" + str(2) + "\t" + str(6) + "\t"

        for box in val:
            obj_cls=label_map[box[4]]

            x0 = float(max(box[0], 0))
            y0 = float(max(box[1], 0))
            x1 = float(min(box[2], im_w - 1))
            y1 = float(min(box[3], im_h - 1))


            label = "{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t0\t".format(obj_cls, x0/im_w, y0/im_h, x1/im_w, y1/im_h)
            lst_line += label

        lst_line += image_file + "\n"
        lst_data += lst_line
        lst_idx += 1

    with open(fn_out,'w') as f:
        f.write(lst_data)


if __name__ == "__main__":
    object_detection_lst(pargs.out, pargs.object_classes)

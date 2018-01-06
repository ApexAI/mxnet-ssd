import argparse
import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

parser.add_argument('--root_path', dest='root_path', help='nexar annotations folder',
                    default="/media/tapir/Data/Thesis/Datasets/detrac", type=str)
parser.add_argument('--paint', dest='paint', help='If set to true, only the last 10k images will be used',
                    default=False, type=bool)


pargs = parser.parse_args()
width = 960
height = 540
import time

def get_rectangle_corners(box,relative=True):
    box_width = float(box['width'])
    box_height = float(box['height'])

    xmin = min(float(box['left']),960)
    ymin = min(float(box['top']),540)
    xmax = min((xmin + box_width),960)
    ymax = min((ymin + box_height),540)

    return (xmin / width, ymin / height, xmax / width, ymax / height) if relative \
            else (xmin, ymin, xmax, ymax)

def create_lst():
    i = 0
    root_path = pargs.root_path
    image_folder = os.path.join(root_path, "Insight-MVT_Annotation_Train")
    label_folder = os.path.join(root_path, "DETRAC-Train-Annotations-XML")
    sequences = os.listdir(image_folder)
    lst_data = ""
    for seq in sequences:
        label_file = os.path.join(label_folder, seq + ".xml")
        tree = ET.parse(label_file)
        root = tree.getroot()

        regions_to_paint=[]
        ignored_region=root.find('ignored_region')
        for ignored_box in ignored_region.iter('box'):
            regions_to_paint.append(get_rectangle_corners(ignored_box.attrib,False))

        for idx, frame in enumerate(root.iter('frame')):
            line = str(idx) + "\t" + str(2) + "\t" + str(6) + "\t"
            image_file=image_folder + "/" + seq + "/img" + frame.attrib['num'].zfill(5) + ".jpg"
            image=Image.open(image_file)
            draw = ImageDraw.Draw(image)

            for target in frame.iter("target"):
                box = target.find('box').attrib

                (xmin, ymin, xmax, ymax) =  get_rectangle_corners(box)

                label = "0\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t0\t".format(xmin, ymin, xmax, ymax)
                #print label
                line += label


            image_file = root_path + "/modified_images/" + seq + "_" + str(idx) + ".jpg"
            if pargs.paint:
                for rectangle in regions_to_paint:
                    draw.rectangle(rectangle,fill=0)
                image.save(image_file)

            line += image_file
            lst_data += line + "\n"

    #mode = "val" if pargs.validation else "train"
    list_file = "detrac_full.lst"
    with open(list_file, "w") as text_file:
        text_file.write(lst_data)


create_lst()
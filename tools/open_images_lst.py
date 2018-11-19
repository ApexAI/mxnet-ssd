import csv
import requests
import os
import argparse


parser = argparse.ArgumentParser(description='filter openImages v4 dataset into an AD dataset')

parser.add_argument('--prefix', dest='prefix', help='lst prefix', default="openimages_people", type=str)
parser.add_argument('--root', dest='root', help='root dir of the dataset', default="/mnt/data/dataset_stuff/openimages_people", type=str)

pargs = parser.parse_args()

root = pargs.root
classes_fn  =  "class-descriptions-boxable.csv"
bboxes_fn   =  "{}-annotations-bbox.csv"
images_fn   =  "{}-images{}-with-rotation.csv"

subsets = ['validation']

traffic_categories = {}

traffic_categories['pedestrian'] = ['Person', 'Man', 'Woman', 'Boy', 'Girl']
traffic_categories['cyclist'] = ['Bicycle', 'Motorcycle', 'Unicycle']
traffic_categories['car'] = ['Land Vehicle', 'Car', 'Bus', 'Van', 'Limousine', 'Taxi', 'Truck']
traffic_categories['light'] = ['Traffic light']
traffic_categories['sign'] = ['Traffic sign', 'Stop sign']

traffic_categories['outdoor_traffic'] = ['Street light', 'Tree']
traffic_categories['outdoor_traffic'] += traffic_categories['cyclist'] + traffic_categories['car']  + traffic_categories['light'] + traffic_categories['sign']

traffic_category_ids = {}

used_classes = ['pedestrian', 'cyclist']
used_class_cat_ids= {}      #class_name: [category_ids]


used_class_ids = [id for id in range(len(used_classes))]
classes = []

labels={}   #imageid: {class_id: [bboxes]}

#### get class ids ####

with open(os.path.join(root,classes_fn),'r') as f:
    reader = eader=csv.reader(f,delimiter=',')
    for row in reader:
        for cls, categories in traffic_categories.iteritems():
            if row[1] in categories:
                used_class_cat_ids.setdefault(cls,[]).append(row[0])    #create a new list if key doesn't exist

open_source_license = "https://creativecommons.org/licenses/by/2.0/"

images_failed = 0

relevant_ims = set()


def get_intersection(bbox1, bbox2):
    xmin = max(bbox1[0],bbox2[0])
    ymin = max(bbox1[1],bbox2[1])
    xmax = min(bbox1[2],bbox2[2])
    ymax = min(bbox1[3],bbox2[3])
    return (xmin,ymin, xmax, ymax)

def merge_boxes(bbox1, bbox2):
    xmin = min(bbox1[0],bbox2[0])
    ymin = min(bbox1[1],bbox2[1])
    xmax = max(bbox1[2],bbox2[2])
    ymax = max(bbox1[3],bbox2[3])
    return (xmin,ymin, xmax, ymax)

def get_area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])


for subset in subsets:
    with open(os.path.join(root, bboxes_fn.format(subset)), 'r') as f:
        reader = eader = csv.reader(f, delimiter=',')
        for row in reader:
            for cidx, cls in enumerate(used_classes):
                if row[2] in used_class_cat_ids[cls]:    #check if the bbox belongs to the classes of interest
                    bbox = (float(row[4]), float(row[6]), float(row[5]), float(row[7]))   #(class_id,xmin, ymin,xmax,ymax)
                    if cls == "pedestrian" and ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]) > 0.2 or (bbox[2]-bbox[0]) > (bbox[3]-bbox[1])*1.2):
                        continue
                    labels.setdefault(row[0],{}).setdefault(cidx,[]).append(bbox)


    # merging bikes with people to get cyclists
    if 'cyclist' in used_classes:
        ped_id=used_classes.index("pedestrian")
        cyc_id=used_classes.index("cyclist")
        for im_id, dets in labels.iteritems():
            if cyc_id not in dets or ped_id not in dets or len(dets[cyc_id])==0:
                continue
            cyclists = []
            for bike_idx, bike_bbox in enumerate(dets[cyc_id]):
                max_iou=0
                matched_cycler_idx=0
                match=False
                for p_idx, ped_bbox in enumerate(dets[ped_id]):
                    ped_area =get_area(ped_bbox)
                    bike_area = get_area(bike_bbox)
                    intersection_area = get_area(get_intersection(ped_bbox,bike_bbox))
                    union_area = ped_area+bike_area-intersection_area
                    iou = float(intersection_area)/union_area
                    if iou > max_iou:
                        max_iou=iou
                        matched_cycler_idx = p_idx

                if max_iou>0.15:
                    cyclist_bbox=merge_boxes(bike_bbox,dets[ped_id][matched_cycler_idx])
                    cyclists.append(cyclist_bbox)
                    dets[ped_id].pop(matched_cycler_idx)
            dets[cyc_id]= cyclists


    lst_str=""
    lst_idx=0
    save_dir = os.path.join(root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(root, images_fn.format(subset, "-boxable" if subset == 'train' else "")), 'r') as f:
        reader = eader = csv.reader(f, delimiter=',')
        for row in reader:
            im_url = row[10]
            save_path=os.path.join(save_dir,row[0]+".jpg")

            if row[4] != open_source_license or row[0] not in labels:   #license and relevance check
                continue

            #if used_classes.index("cyclist") not in labels[row[0]] or len(labels[row[0]][used_classes.index("cyclist")]) < 1 :
            #    continue

            if not os.path.exists(save_path):
                try:    #write image
                    r = requests.get(im_url, allow_redirects=False)
                    if r.status_code != 200:
                        continue
                    open(save_path, 'wb').write(r.content)

                except:
                    images_failed+=1
                    continue

            line = "{}\t2\t6\t".format(str(lst_idx))

            for cls, bboxes in labels[row[0]].iteritems():
                for bbox in bboxes:
                    line += "{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t0\t".format(cls, *bbox)

            line += os.path.join(subset,row[0]+".jpg")

            lst_str += line + "\n"
            lst_idx+=1
    with open(pargs.prefix+"_{}.lst".format(subset), 'w') as f:
        f.write(lst_str)



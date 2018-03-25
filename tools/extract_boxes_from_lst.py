import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser(description='order and save detections by confidence')

parser.add_argument('--input', dest='input', help='trained model prefix',
                    default="resnet18",
                    type=str)

parser.add_argument('--root', dest='root', help='root dir',
                    default="/mnt/data/dataset_stuff/coco",
                    type=str)

parser.add_argument('--out-dir', dest='out_dir', help='output dir',
                    default="/mnt/data/dataset_stuff/coco/cropped_images/train",
                    type=str)

parser.add_argument('--out-file', dest='out_file', help='output file',
                    default="coco_pedestrians_train.lst",
                    type=str)
pargs = parser.parse_args()

classes=[0.0,1.0]
lst_idx=0
lst_data=""

with open(pargs.input) as f:
    lines=f.readlines()

    for line in lines:
        elements  = line.split('\t')
        image_name= os.path.join(pargs.root, elements[len(elements)-1])
        image_name=image_name[0:len(image_name)-1]

        if(not os.path.isfile(image_name)):
            continue
        all_boxes = elements[3:len(elements)-1]
        all_boxes = [all_boxes[x:x + 6] for x in range(0, len(all_boxes), 6)]

        image=None


        for box in all_boxes:
            if float(box[0]) in classes:
                if(not image):
                    image=Image.open(image_name)
                    im_w, im_h = image.size

                save_path = os.path.join(pargs.out_dir, str(lst_idx) + ".jpg")

                if not os.path.isfile(save_path):
                    x0 = max(float(box[1])*im_w, 0)
                    y0 = max(float(box[2])*im_h, 0)
                    x1 = min(float(box[3])*im_w, im_w - 1)
                    y1 = min(float(box[4])*im_h, im_h - 1)

                    if((x1-x0)*(y1-y0)<100):
                        continue

                    res = image.crop((x0, y0, x1, y1))
                    res.save(save_path)

                lst_data+="{}\t{}\t{}\n".format(lst_idx, box[0], save_path)
                lst_idx+=1



with open(pargs.out_file,'w') as f:
    f.write(lst_data)








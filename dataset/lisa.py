
import os
from PIL import Image
root="/mnt/data/dataset_stuff/lisa"


subsets=['Dense','Sunny','Urban']

lst_data=""
idx=0

for subset in subsets:
    dir=os.path.join(root,subset)
    with open(os.path.join(dir,'pos_annot.dat')) as f:
        lines = f.readlines()[::4]  #downsampling frames

        for line in lines:

            lst_line = str(idx) + "\t" + str(2) + "\t" + str(6) + "\t"

            els=line.split('\t')

            im_path=os.path.join(dir,'images',str(els[0])+'.jpg')
            with Image.open(im_path) as img:
                width, height = img.size

            for i in range(int(els[1])):
                bbox=[float(n) for n in els[2+i].split(' ')]
                label = "0\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t0\t".format(bbox[0]/width, bbox[1]/height, (bbox[0]+bbox[2])/width, (bbox[1]+bbox[3])/height)
                lst_line += label

            lst_line += im_path+"\n"
            lst_data += lst_line
            idx+=1



with open('lisa.lst', "w") as text_file:
    text_file.write(lst_data)




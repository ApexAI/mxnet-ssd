from __future__ import absolute_import
import os
import numpy as np
from .imdb import Imdb
from .pycocotools.coco import COCO


class Coco(Imdb):
    """
    Implementation of Imdb for MSCOCO dataset: https://http://mscoco.org

    Parameters:
    ----------
    anno_file : str
        annotation file for coco, a json file
    image_dir : str
        image directory for coco images
    shuffle : bool
        whether initially shuffle image list

    """
    def __init__(self, anno_file, image_dir, image_set, shuffle=True, names='mscoco.txt'):
        assert os.path.isfile(anno_file), "Invalid annotation file: " + anno_file
        basename = os.path.splitext(os.path.basename(anno_file))[0]
        super(Coco, self).__init__('coco_' + basename)
        self.image_dir = image_dir

        self.classes = self._load_class_names(names,
            os.path.join(os.path.dirname(__file__), 'names'))
        self.set=image_set
        self.num_classes = len(self.classes)
        self._load_all(anno_file, shuffle)
        self.num_images = len(self.image_set_index)


    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.image_dir, 'images', name)
        assert os.path.isfile(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _load_all(self, anno_file, shuffle):
        """
        initialize all entries given annotation json file

        Parameters:
        ----------
        anno_file: str
            annotation json file
        shuffle: bool
            whether to shuffle image list
        """
        image_set_index = []
        labels = []
        coco = COCO(anno_file)
        img_ids = coco.getImgIds()
        #print(img_ids)
        cars=[3,6,8]
        pedestrians=[1]
        cyclists=[2,4]
        lights=[10]
        signs=[13]

        apex_categories=cars+pedestrians+cyclists+lights+signs
        cnt=0
        humanonly=0
        human_count=0

        for img_id in img_ids:
            relevant=False
            # filename
            image_info = coco.loadImgs(img_id)[0]
            filename = image_info["file_name"]
            #print(filename)
            #subdir = filename.split('_')[1]
            height = image_info["height"]
            width = image_info["width"]
            # label
            anno_ids = coco.getAnnIds(imgIds=img_id)
            annos = coco.loadAnns(anno_ids)
            label = []

            #print("listing categories for filename: "+filename)

            hashumans=False
            for anno in annos:
                cat_id = int(anno["category_id"])
                if(cat_id in apex_categories):
                    cat_reduced= 0 if (cat_id in cars) else 1 if(cat_id in pedestrians) else 2 if(cat_id in cyclists) else 3 if(cat_id in lights) else 4
                    bbox = anno["bbox"]
                    assert len(bbox) == 4
                    xmin = float(bbox[0]) / width
                    ymin = float(bbox[1]) / height
                    xmax = xmin + float(bbox[2]) / width
                    ymax = ymin + float(bbox[3]) / height
                    label.append([cat_reduced, xmin, ymin, xmax, ymax, 0])
                    #print("category: %d"%cat_reduced)
                    if (cat_id in pedestrians):
                        hashumans=True
                    if(cat_id not in pedestrians):   #at least one non-person object is necessary
                        relevant=True

            if(label and not relevant):
                humanonly+=1
            if label and relevant:
                if(hashumans):
                    human_count+=1
                #print("adding "+filename)
                labels.append(np.array(label))
                image_set_index.append(os.path.join(self.set, filename))
                cnt+=1
        print("added %d images"%cnt)
        print("%d images has only humans"%humanonly)
        print("%d registered images has humans"%human_count)

        if shuffle:
            import random
            indices = range(len(image_set_index))
            random.shuffle(indices)
            image_set_index = [image_set_index[i] for i in indices]
            labels = [labels[i] for i in indices]
        # store the results
        self.image_set_index = image_set_index
        self.labels = labels

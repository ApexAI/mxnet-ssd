from __future__ import absolute_import
import os
import numpy as np
from .imdb import Imdb
import yaml

class Bosch(Imdb):
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
    def __init__(self, anno_file, image_dir, image_set, shuffle=True, names='apex_coco.names'):
        assert os.path.isfile(anno_file), "Invalid annotation file: " + anno_file
        basename = os.path.splitext(os.path.basename(anno_file))[0]
        super(Bosch, self).__init__('bosch_' + basename)
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
        image_file = os.path.join(self.image_dir, name)
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
        width=1280
        height=720
        labels=[]
        image_set_index=[]

        with open(anno_file) as f:
            data = yaml.safe_load(f)
            print(data[3]['boxes'][0]['y_min'])
            num_frames=len(data)


            for fidx in range(num_frames):
                frame=data[fidx]
                filename=frame['path']
                label=[]

                num_boxes=len(frame['boxes'])

                for bixd in range(num_boxes):
                    box=frame['boxes'][bixd]

                    xmin=box['x_min'] / width
                    ymin=box['y_min'] / height
                    xmax=box['x_max'] / width
                    ymax=box['y_max'] / height

                    label.append([3, xmin, ymin, xmax, ymax, 0])

                if(label):
                    labels.append(np.array(label))
                    image_set_index.append(filename)

        if shuffle:
            import random
            indices = range(len(image_set_index))
            random.shuffle(indices)
            image_set_index = [image_set_index[i] for i in indices]
            labels = [labels[i] for i in indices]
        # store the results
        self.image_set_index = image_set_index
        self.labels = labels
from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter

class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
        print([('data', (batch_size, 3, data_shape, data_shape))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels
        self.sum=0.0

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        result = []
        detections = []
        print("data shape: {}".format(self.data_shape))
        print("detiter shape: {}".format(det_iter.provide_data))
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        for pred, _, _ in self.mod.iter_predict(det_iter):
            detections.append(pred[0].asnumpy())
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
            self.sum+=time_elapsed

        for output in detections:
            for i in range(output.shape[0]):
                det = output[i, :, :]
                res = det[np.where(det[:, 0] >= 0)[0]]
                result.append(res)
        return result

    def binary_detect(self, det_iter, show_timer=False):
        print("loading binaru symbol......")
        load_symbol, args, auxs = mx.model.load_checkpoint("model/binarized_resnet18_quantized", 2)
        model = mx.mod.Module(load_symbol, label_names=None, context=self.ctx)
        model.bind(data_shapes=det_iter.provide_data,label_shapes=det_iter.provide_label)
        model.set_params(args, auxs)


        num_images = det_iter._size
        result = []
        detections = []
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        for pred, _, _ in model.iter_predict(det_iter):
            detections.append(pred[0].asnumpy())
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
            self.sum+=time_elapsed

        for output in detections:
            for i in range(output.shape[0]):
                det = output[i, :, :]
                res = det[np.where(det[:, 0] >= 0)[0]]
                result.append(res)
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2,
                                    '{:s} {:.3f}'.format(class_name, score),
                                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=12, color='white')
        plt.show()

    def detect_and_visualize(self, im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        import cv2
        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            img = cv2.imread(im_list[k])
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection(img, det, classes, thresh)

    def visualize_stream(self, im_list,  out, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False, benchmark=False):


        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import random
        colors = dict()
        num_cls = len(classes)

        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

        colors = {0: (29, 56, 247), 1: (255, 32, 120), 2: (98, 176, 168), 3: (143, 155, 39), 4: (244, 214, 64)}
        if len(classes) > 5:
            for cid in range(num_cls):
                colors[cid] = (float(cid+1)/(2*num_cls)*255, float(cid+1)/(2*num_cls)*255, (2*num_cls-2*cid)*(255)/(2*num_cls))

        for idx, im in enumerate(im_list):
            dets=self.im_detect([im], root_dir, extension, show_timer=show_timer)
            if not benchmark:
                for k, det in enumerate(dets):
                    img = cv2.imread(im)
                    img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]

                    height = img.shape[0]
                    width = img.shape[1]
                    for i in range(det.shape[0]):
                        cls_id = int(det[i, 0])
                        if cls_id >= 0:
                            score = det[i, 1]
                            if score > thresh:
                                xmin = int(det[i, 2] * width)
                                ymin = int(det[i, 3] * height)
                                xmax = int(det[i, 4] * width)
                                ymax = int(det[i, 5] * height)
                                class_name = str(cls_id)
                                if classes and len(classes) > cls_id:
                                    class_name = classes[cls_id]
                                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[cls_id], 4)
                                cv2.putText(img, "{}: {}".format(class_name, score), (xmin, ymin - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls_id], 2, cv2.LINE_AA)


                    im_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    cv2.imwrite(out+"/{}.jpg".format(idx), im_rgb)
        print("time elapsed total: %f"%self.sum)
        print("average time: %f"% (self.sum/len(im_list)))



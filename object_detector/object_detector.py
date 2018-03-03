import mxnet as mx
import cv2
import numpy as np
import logging
import time
#from collections import namedtuple
#Batch = namedtuple('Batch', ['data'])

class object_detector(object):
    """

    Object detection API for SSD

    :param prefix:          The model file's prefix. Example: 'resnet18' for resnet18-symbol.json
    :param epoch:           The model file's epoch.  Example?  12        for resnet18-0012.params
    :param conf_thresh:     Confidence threshold to filter the detections.
    :param classes:         List of class names. Ignore for single-class detection.
    :param device:          Device to run the network: cpu or gpu
    :param data_shape:      Shape the images will be resized into. Must be fitting to the network.
    :param verbose:         Displays debugging messages if set to True.
    :param batch_size:      The size of each batch. Currently not supported
    """

    def __init__(self, prefix, epoch, conf_thresh, classes=None, device='gpu', data_shape=512, verbose=False, batch_size=1):
        logging.basicConfig()
        self.logger=logging.getLogger("Detection")
        log_level=logging.DEBUG if(verbose) else logging.INFO
        self.logger.setLevel(log_level)
        self.data_shape=data_shape
        self.conf_thresh=conf_thresh

        #Selecting the device
        ctx= mx.gpu() if(device=='gpu') else mx.cpu()
        self.logger.info("Device type used for detection: {} ".format(ctx.device_type))

        #Loading the model
        sym, args, auxs = mx.model.load_checkpoint(prefix, epoch)
        self.mod = mx.mod.Module(sym, label_names=None, context=ctx)
        self.mod.bind(data_shapes=[('data', (batch_size, 3, self.data_shape, self.data_shape))])
        self.mod.set_params(args, auxs)

        self.logger.info("Model loaded successfully")

        self.classes= classes if(classes) else ['base class']



    def process_image(self, image):
        mean_pixels = np.array([128, 128, 128]).reshape((3, 1, 1)) #normalize to mean || TODO: use the mean from data
        cvim        = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change to r,g,b order
        cvim        = cv2.resize(cvim, (self.data_shape, self.data_shape,))    #resize the network's input size
        cvim        = np.swapaxes(cvim, 0, 2)
        cvim        = np.swapaxes(cvim, 1, 2)  # change to (channel, height, width)
        cvim        = cvim - mean_pixels
        cvim        = cvim[np.newaxis, :]
        return cvim


    def detect_objects(self, image):
        """
        Detects the objects in an image and returns a list of bounding boxes
        :param image : OpenCV image input
        :return      : A list of dictionaries containing bounding box corners and the confidence
        """

        detections=[]
        result=[]
        height  = image.shape[0]
        width   = image.shape[1]

        #process the openCV image
        processed_image=self.process_image(image)
        image_input=mx.io.NDArrayIter(data=processed_image)
        self.logger.debug('Image input is ready')


        self.logger.debug("Running the forward pass")
        detection_begin=time.time()
        for pred, _, _ in self.mod.iter_predict(image_input):
            detections.append(pred[0].asnumpy())
        detection_end=time.time()-detection_begin
        self.logger.debug("Time spent for detection: {}".format(detection_end))

        #extract the valid detections from the network output
        for output in detections:
            for i in range(output.shape[0]):
                det = output[i, :, :]
                filtered_detections = det[np.where(((det[:, 0] >= 0)& (det[:, 1] >= self.conf_thresh)))]

        # TODO: experiment using the following instead
        #self.mod.forward(Batch([mx.nd.array(processed_image)]))
        #detections= self.mod.get_outputs()[0][0].asnumpy() #get the detections for the single image
        #filtered_detections=detections[np.where((detections[:,0]>-1)& (detections[:, 1] >= self.conf_thresh))]

        #Iterate over the detections to fill in the returning list
        for fd in filtered_detections:
            detection={}

            if(fd[0]>=len(self.classes)):
                raise UserWarning("The provided number of classes doesn't fit the model")
                detection['class'] = "Unknown"
            else:
                detection['class']      = self.classes[int(fd[0])]

            detection['confidence'] = fd[1]
            detection['xmin']       = int(fd[2] * width)
            detection['ymin']       = int(fd[3] * height)
            detection['xmax']       = int(fd[4] * width)
            detection['ymax']       = int(fd[5] * height)
            result.append(detection)

        return result



    def detect_and_visualize(self,im, output):
        """
        Detects the objects and plots bounding boxes on the image
        :param im: openCV image input
        :param output: output filepath
        :return:
        """
        results=self.detect_objects(im)
        for res in results:
            xmin = res['xmin']
            ymin = res['ymin']
            xmax = res['xmax']
            ymax = res['ymax']
            conf=res['confidence']
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(im, "{}: {}".format(res["class"], str(conf)), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(output,im)
        self.logger.debug('Image is written to: {}'.format(output))


## Object Detection API
### Usage:
Download the model files. It contains a json file defining the network and a .params file containing the weights. Example:`resnet18-symbol.json` and `resnet18-0001.params`

Constructor Parameters:  
**prefix**:          The model file's prefix. Example:`path/resnet18` for `path/resnet18-symbol.json`  
**epoch**:           The model file's epoch.  Example:  `12`        for `path/resnet18-0012.params`  
**conf_thresh**:     Confidence threshold to filter the detections.   
**classes**:         List of class names. You can ignore for single-class detection.  
**device**:          Device to run the network: cpu or gpu.  
**data_shape**:      Shape the images will be resized into. Must be fitting to the network. Ignore if not sure.  
**verbose**:         Displays debugging messages if set to True.  
**batch_size**:      The size of each batch. Currently only single image supported.  

Output is a list of dictionaries that contain the following keys:  
**xmin**        : Bbox coordinate.  
**ymin**        : Bbox coordinate.  
**xmax**        : Bbox coordinate.  
**ymax**        : Bbox coordinate.  
**confidence**  : Confidence value associated with the detection  
**class**       : Predicted class name of the detection. Will only be accurate if the constructor parameter classes is correct.  

### Example:
```python
#symbol_file_path  = 'mypath/resnet18-symbol.json'
#weights_file_path = 'mypath/resnet18-0012.params'
import cv2

prefix = "mypath/resnet18"
epoch  = 12
threshold=0.5

image       =   cv2.imread("image.jpg")
my_detector =   object_detector(prefix, epoch, conf_thresh=threshold, classes=['car'])
detections  =   mydetector.detect_objects(image)

for i, det in enumerate(detections):
    print('\n Detected box {}:'.format(i))
    for key,val in det.iteritems():
        print("{}: {}".format(key,val))
```
Output:
```
Detected box 0:
confidence: 0.774845957756
ymax: 737
xmax: 667
xmin: 537
ymin: 622
class: car

Detected box 1:
confidence: 0.70752632618
ymax: 739
xmax: 1192
xmin: 998
ymin: 643
class: car
```

#### Requirements
  - MxNet
  - OpenCV
  - Numpy

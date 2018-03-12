import mxnet as mx
import os
import tkFileDialog
import Tkinter as tk, Canvas
from traffic_light_classification.weakly_supervise import AssistedSupervisor
from object_detector.object_detector import object_detector
from PIL import Image, ImageTk, ImageDraw
import re
import cv2

class DetectionSupervisor():
    def __init__(self):
        self.root = tk.Tk()

        self.img_display_idx=0
        self.start_x=0
        self.start_y=0
        self.x = self.y = 0
        self.current_rect = None

        self.check_lower_bound  = lambda x:  max(0,x)
        self.check_upper_bound  = lambda x:  min(0,x)
        self.downscale          = lambda x:  int((x * 3) / 4)
        self.upscale            = lambda x:  int((x * 4) / 3)

        self.construct_panels()
        self.set_buttons()
        self.setup_listeners()

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.image_detections=[]

        ## with Windows OS
        #self.image_panel.bind_all("<MouseWheel>", self._on_mousewheel)
        ## with Linux OS
        #self.root.bind("<Button-4>", self._on_mousewheel)
        #self.root.bind("<Button-5>", self._on_mousewheel)

        self.root.bind("<Key>", self._on_key)


    def construct_panels(self):
        self.canvas=Canvas.Canvas(self.root, width=1046, height=842)
        self.canvas.grid(row=0,column=0)

        self.control_panel = tk.Frame(self.root, width=1000, height = 1000, background="#b22222")
        self.control_panel.grid(row=1,column=0)

    def setup_listeners(self):
        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.shape_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.register_rectangle)

    def set_buttons(self):
        self.quit_button=tk.Button(self.control_panel, text="QUIT", fg="red", command=self.control_panel.quit)
        self.quit_button.grid(row=0,column=0)

        self.select_folder_button = tk.Button(self.control_panel, text='Select image folder', command=self.select_directory)
        self.select_folder_button.grid(row=0, column=1)

        self.select_model_button = tk.Button(self.control_panel, text='Select model', command=self.select_model)
        self.select_model_button.grid(row=0, column=2)

        self.classify_button = tk.Button(self.control_panel, state=tk.DISABLED, text='Run detection',command=self.detect_objects)
        self.classify_button.grid(row=0, column=3)

        #self.save_button = tk.Button(self.control_panel, state=tk.DISABLED, text='Save', command=self.save_lst_file)
        #self.save_button.grid(row=0, column=4)


    def select_directory(self):
        self.images_path= tkFileDialog.askdirectory(initialdir=".",title='Choose a directory')
        self.select_folder_button['text']="Select new directory"
        self.get_imagenames()

    def select_model(self):
        model_fn= tkFileDialog.askopenfilename(initialdir=".",title='Choose a \'params\' file (you must have the json file too)')
        if(model_fn!=""):
            self.model_prefix=model_fn[0:len(model_fn)-12]
            self.model_epoch=int(model_fn[len(model_fn) - 11:len(model_fn) - 7])
            print(self.model_prefix)
            print(self.model_epoch)
            print(type(self.model_epoch))

            self.select_model_button['text']="Select a new model"
            self.classify_button.config(state="normal")

    def get_imagenames(self):
        re_prog = re.compile(".*\.(png|jpg|PNG|JPG|JPEG|jpeg|gif)$")  # image check

        def compare_filename(s1, s2):
            return cmp(int(s1[0:len(s1) - 4]), int(s2[0:len(s2) - 4]))

        try:
            self.images = sorted(os.listdir(self.images_path),cmp=compare_filename)
        except:
            self.images=sorted(os.listdir(self.images_path))

        self.images=[os.path.join(self.images_path,im) for im in self.images]
        self.images = [elem for elem in self.images if re_prog.match(elem)]
        self.num_imgs=len(self.images)

    def detect_objects(self):
        self.detector=object_detector(prefix=self.model_prefix, epoch=self.model_epoch, conf_thresh=0.5, device=mx.cpu())

        for idx, imname in enumerate(self.images):
            im=cv2.imread(imname)
            self.image_detections.append(self.detector.detect_objects(im))

        self.display_detections()

    def onRectClick(self, event):
        rect = self.canvas.find_withtag("current")[0]
        event.widget.delete("current")
        self.image_detections[self.img_display_idx].remove(self.current_rectangles[rect])


    def plot_image(self,n):
        image = Image.open(self.images[n])
        self.imw, self.imh=image.size
        image.thumbnail((1026,822), Image.ANTIALIAS)
        self.current_photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.current_photo, anchor = tk.NW)


    def plot_detections(self, n):
        detections = self.image_detections[n]
        self.current_rectangles={}
        for detection in detections:
            rect=self.canvas.create_rectangle(self.downscale(detection['xmin']),self.downscale(detection['ymin']),
                                         self.downscale(detection['xmax']), self.downscale(detection['ymax']), width=3,fill= 'red', stipple='gray12')
            self.canvas.tag_bind(rect, '<ButtonPress-3>', self.onRectClick)
            self.current_rectangles[rect]=detection
            print detection

    def display_detections(self):
        self.plot_image(0)
        self.plot_detections(0)


    def update_display(self):
        self.canvas.delete("all")
        self.plot_image(self.img_display_idx)
        self.plot_detections(self.img_display_idx)


    def _on_key(self,event):
        if(event.keysym=="Right"):
            self.img_display_idx =min(self.img_display_idx+1, self.num_imgs-1)
        if(event.keysym=="Left"):
            self.img_display_idx=max(0,self.img_display_idx-1)
        self.update_display()


    def start_rectangle(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle if not yet exist
        #if not self.rect:
        self.current_rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, width=3,fill= 'red', stipple='gray12')
        self.canvas.tag_bind(self.current_rect, '<ButtonPress-3>', self.onRectClick)

    def shape_rectangle(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, curX, curY)

    def register_rectangle(self, event):
        (xmin,ymin,xmax,ymax) = self.canvas.coords(self.current_rect)
        if((xmax-xmin)*(ymax-ymin)>225):

            xmin=max(self.upscale(xmin),0)
            ymin=max(self.upscale(ymin),0)
            xmax=min(self.upscale(xmax),self.imw)
            ymax=min(self.upscale(ymax),self.imh)

            detection={'class':'car','confidence':1.0,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax}
            self.current_rectangles[self.current_rect] = detection
            self.image_detections[self.img_display_idx].append(detection)
        else:
            print 'selected area too small'
            event.widget.delete(self.current_rect)


    def run(self):
        self.root.mainloop()
        self.root.destroy()



DS= DetectionSupervisor()

DS.run()
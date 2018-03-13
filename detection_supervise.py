import os
import tkFileDialog
import Tkinter as tk, Canvas
from object_detector.object_detector import object_detector
from PIL import Image, ImageTk
import re
import cv2

class DetectionSupervisor():
    """
    Simple GUI to supervise object detection.
    Detects objects, visualizes the results, allows label modifications, exports lst files.
    Author: Yunus Emre Caliskan
    """
    def __init__(self):
        self.root = tk.Tk()

        self.initialize()

        self.downscale = lambda x:  int((x * 3) / 4)
        self.upscale   = lambda x:  int((x * 4) / 3)

        self.construct_panels()
        self.set_buttons()

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def initialize(self):
        self.img_display_idx=0
        self.start_x=0
        self.start_y=0
        self.x = self.y = 0
        self.current_rect = None
        self.rate=tk.IntVar()
        self.ctx = tk.StringVar(self.root)
        self.threshold=tk.DoubleVar()
        self.rate.set(5)
        self.threshold.set(0.35)
        self.ctx = tk.StringVar(self.root)
        self.rate.set(5)
        self.images_path=''
        self.images=[]
        self.image_detections=[]
        self.current_rectangles=[]
        self.classified=False
        self.current_canvas_image = None
        self.current_rectangles=None

    def construct_panels(self):
        self.canvas=Canvas.Canvas(self.root)
        self.canvas.grid(row=0,column=0)

        self.control_panel = tk.Frame(self.root, width=1000, height = 1000)
        self.control_panel.grid(row=1,column=0)

        self.scale_panel=tk.Frame(self.control_panel)
        self.scale_panel.grid(row=0,column=1)

        self.ctx_panel=tk.Frame(self.control_panel)
        self.ctx_panel.grid(row=0,column=4)

        self.threshold_panel=tk.Frame(self.control_panel)
        self.threshold_panel.grid(row=0,column=5)

    def setup_listeners(self):
        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.shape_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.register_rectangle)
        self.root.bind("<Key>", self._on_key)
        # with Linux OS
        self.root.bind("<Button-4>", self._on_mousewheel)
        self.root.bind("<Button-5>", self._on_mousewheel)

    def set_buttons(self):
        self.quit_button=tk.Button(self.control_panel, text="QUIT", fg="red", command=self.control_panel.quit)
        self.quit_button.grid(row=0,column=0)

        self.rate_scale=tk.Scale(self.scale_panel, variable= self.rate, from_=1, to=30,  orient=tk.HORIZONTAL, command=self.rate_change)
        self.rate_scale.grid(row=0, column=0)
        tk.Label(self.scale_panel,text="Use every nth image").grid(row=1, column=0)


        self.select_folder_button = tk.Button(self.control_panel, text='Select image folder', command=self.select_directory)
        self.select_folder_button.grid(row=0, column=2)

        self.select_model_button = tk.Button(self.control_panel, text='Select model', command=self.select_model)
        self.select_model_button.grid(row=0, column=3)

        choices=['cpu','gpu']
        self.popupMenu = tk.OptionMenu(self.ctx_panel, self.ctx, *choices)
        self.popupMenu.grid(row=0, column=0)
        self.ctx.set("gpu")
        #tk.Label(self.ctx_panel, text="Choose a device").grid(row=0, column=0)

        self.threshold_scale=tk.Scale(self.threshold_panel, variable= self.threshold, from_=0.15, to=1.0, resolution=0.05,  orient=tk.HORIZONTAL)
        self.threshold_scale.grid(row=0, column=0)
        tk.Label(self.threshold_panel,text="Set confidence threshold").grid(row=1, column=0)

        self.classify_button = tk.Button(self.control_panel, state=tk.DISABLED, text='Run detection',command=self.detect_objects)
        self.classify_button.grid(row=0, column=6)

        self.save_button = tk.Button(self.control_panel, state=tk.DISABLED, text='Save', command=self.save_lst_file)
        self.save_button.grid(row=0, column=7)

        self.clear_button = tk.Button(self.control_panel, state=tk.DISABLED, text='Clear', command=self.clear)
        self.clear_button.grid(row=0, column=8)


    def select_directory(self):
        self.images_path= tkFileDialog.askdirectory(initialdir=".",title='Choose a directory')
        if (self.images_path != ""):
            self.select_folder_button['text']="Select new directory"
            self.update_image_names()
            self.init_display()
            self.rate_scale.config(state="normal")
            self.threshold_scale.config(state="normal")
            self.setup_listeners()
            self.clear_button.config(state="normal")


    def select_model(self):
        model_fn= tkFileDialog.askopenfilename(initialdir=".",title='Choose a \'params\' file (you must have the json file too)')
        if(model_fn!=""):
            self.model_prefix=model_fn[0:len(model_fn)-12]
            self.model_epoch=int(model_fn[len(model_fn) - 11:len(model_fn) - 7])

            self.select_model_button['text']="Select a new model"
            self.classify_button.config(state="normal")

    def detect_objects(self):
        #display waiting message
        image = Image.open("data/demo/pleasewait.png")
        imw, imh=image.size
        self.current_photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.current_canvas_image, image=self.current_photo)
        self.canvas.config(width=imw, height=imh)
        self.root.update()

        self.detector = object_detector(prefix=self.model_prefix, epoch=self.model_epoch, conf_thresh=self.threshold.get(),
                                        device=self.ctx.get())

        for idx, imname in enumerate(self.images):
            im = cv2.imread(imname)
            self.image_detections.append(self.detector.detect_objects(im))

        self.classified = True
        self.init_display()
        self.save_button.config(state="normal")
        self.rate_scale.config(state=tk.DISABLED)
        self.threshold_scale.config(state=tk.DISABLED)



    def plot_image(self,n):
        image = Image.open(self.images[n])
        self.imw, self.imh=image.size
        image.thumbnail((self.downscale(self.imw),self.downscale(self.imh)), Image.ANTIALIAS)
        self.current_photo = ImageTk.PhotoImage(image)
        if self.current_canvas_image:
            self.canvas.itemconfig(self.current_canvas_image, image=self.current_photo)
        else:
            self.current_canvas_image= self.canvas.create_image(0, 0, image=self.current_photo, anchor = tk.NW)


    def plot_detections(self, n):
        detections = self.image_detections[n]
        self.current_rectangles={}
        for detection in detections:
            rect=self.canvas.create_rectangle(self.downscale(detection['xmin']),self.downscale(detection['ymin']),
                                         self.downscale(detection['xmax']), self.downscale(detection['ymax']), width=3,fill= '#800000', stipple='gray12')
            self.canvas.tag_bind(rect, '<ButtonPress-3>', self.onRectClick)
            self.current_rectangles[rect]=detection

    def init_display(self):
        self.img_display_idx=0
        self.update_display()


    def update_display(self):
        self.plot_image(self.img_display_idx)
        self.canvas.config(width=self.downscale(self.imw), height=self.downscale(self.imh))
        if self.classified:
            if self.current_rectangles:
                for key,val in self.current_rectangles.iteritems():
                    self.canvas.delete(key)
            self.plot_detections(self.img_display_idx)


    def update_image_names(self):
        re_prog = re.compile(".*\.(png|jpg|PNG|JPG|JPEG|jpeg|gif)$")  # image check

        def compare_filename(s1, s2):
            return cmp(int(s1[0:len(s1) - 4]), int(s2[0:len(s2) - 4]))

        try:
            self.images = sorted(os.listdir(self.images_path),cmp=compare_filename)
        except:
            self.images=sorted(os.listdir(self.images_path))

        self.images=self.images[0::self.rate.get()]   #get every nth frame instead of all frames

        self.images=[os.path.join(self.images_path,im) for im in self.images]
        self.images = [elem for elem in self.images if re_prog.match(elem)]
        self.num_imgs=len(self.images)


    def rate_change(self, event):
        if self.images_path:
            self.update_image_names()

    def onRectClick(self, event):
        rect = self.canvas.find_withtag("current")[0]
        event.widget.delete("current")
        self.image_detections[self.img_display_idx].remove(self.current_rectangles[rect])



    def _on_key(self,event):
        if(self.images):
            if(event.keysym=="Right"):
                self.img_display_idx =min(self.img_display_idx+1, self.num_imgs-1)
            if(event.keysym=="Left"):
                self.img_display_idx=max(0,self.img_display_idx-1)
            self.update_display()

    def _on_mousewheel(self, event):
        if (self.images):
            if event.num == 5 or event.delta == -120:
                self.img_display_idx =min(self.img_display_idx+1, self.num_imgs-1)
            if event.num == 4 or event.delta == 120:
                self.img_display_idx=max(0,self.img_display_idx-1)
            self.update_display()

    def start_rectangle(self, event):
        if (self.classified):
            # save mouse drag start position
            self.start_x = event.x
            self.start_y = event.y

            # create rectangle if not yet exist
            #if not self.rect:
            self.current_rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, width=3,fill= '#800000', stipple='gray12')
            self.canvas.tag_bind(self.current_rect, '<ButtonPress-3>', self.onRectClick)

    def shape_rectangle(self, event):
        if (self.classified):
            curX, curY = (event.x, event.y)

            # expand rectangle as you drag the mouse
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, curX, curY)

    def register_rectangle(self, event):
        if (self.classified):
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

    def clear(self):
        self.img_display_idx=0
        self.images_path=''
        self.images=[]
        self.image_detections=[]
        self.current_rectangles=[]
        self.classified=False
        self.canvas.delete("all")
        self.current_canvas_image = None
        self.current_rectangles=None
        self.classify_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.rate_scale.config(state="normal")
        self.threshold_scale.config(state="normal")

    def save_lst_file(self):
        lst_data=""

        lst_idx=0
        for idx, im in enumerate(self.images):
            if self.image_detections[idx]:
                line=str(lst_idx)+ "\t" + str(2) + "\t" + str(6)+ "\t"
                with Image.open(im) as img:
                    width, height= img.size
                for det in self.image_detections[idx]:
                     line += "0.0000\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t0.0000\t".format(float(det['xmin'])/width, float(det['ymin'])/height,
                                                                                           float(det['xmax'])/width, float(det['ymax'])/height)
                lst_data += line + im+ "\n"
                lst_idx+=1

        filename = tkFileDialog.asksaveasfilename(initialdir=".", title="Save the list file")
        if filename != "":
            with open(filename+".lst", "w") as text_file:
                text_file.write(lst_data)



    def run(self):
        self.root.mainloop()
        self.root.destroy()


if __name__ == '__main__':
    DS= DetectionSupervisor()
    DS.run()
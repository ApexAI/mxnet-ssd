import mxnet as mx
import os
import argparse
import tkFileDialog
import Tkinter as tk
from nexar_forward import image_forward
from PIL import Image, ImageTk

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
parser.add_argument('--prefix', dest='prefix', help='train list to use',
                    default="models/resnet_nexar", type=str)
parser.add_argument('--epoch', dest='epoch', help='train list to use',
                    default=0, type=int)
parser.add_argument('--image_path', dest='image_path', help='directory of images to label',
                    default="../data/demo/65.png", type=str)

parser.add_argument('--output_list', dest='output_list', help='list file to output',
                    default="../data/demo/65.png", type=str)

class AssistedSuperviser():
    def __init__(self,parser,master):
        self.pargs = parser.parse_args()
        self.img_display_idx=0
        self.master=master

        self.master.minsize(width=1000, height=1000)
        self.image_panel = tk.LabelFrame(self.master, width=500, height=500, padx=10, pady=10)
        self.image_panel.grid(row=0,column=0)

        self.control_panel = tk.Frame(self.master, width=1000, height = 1000, background="#b22222")
        self.control_panel.grid(row=1,column=0)
        #self.control_panel.place(relx=.5, rely=.9, anchor="c")

        self.quit_button = tk.Button(
            self.control_panel, text="QUIT", fg="red", command=self.control_panel.quit
            )
        self.quit_button.grid(row=0,column=0)

        self.select_folder=tk.Button(self.control_panel, text='Browse', command=self.select_directory)
        self.select_folder.grid(row=0,column=1)
        #self.select_folder.pack(side=tk.LEFT)

        self.classify_button=tk.Button(self.control_panel, text='Classify', command=self.classify_images)
        self.classify_button.grid_forget()


    def svvitch_cb(self):
        image = Image.open(self.images[self.img_display_idx])
        photo = ImageTk.PhotoImage(image)
        self.label.config(image=photo)
        self.label.image = photo
        self.label2.config(text=self.results[self.img_display_idx]['prediction'])

    def show_images(self):

        image = Image.open(self.images[0])
        photo = ImageTk.PhotoImage(image)
        self.label = tk.Label(self.image_panel, image=photo)
        self.label.image = photo  # keep a reference!
        self.label.grid(row=0,column=0)

        self.label_panel = tk.LabelFrame(self.master, width=100, height=500, padx=10, pady=10)

        self.label_panel.grid(row=0,column=1)

        self.label2=tk.Label(self.label_panel,text=self.results[0]['prediction'])
        self.label2.grid(row=0,column=0, columnspan=3)

        tk.Button(self.label_panel,text="green").grid(row=1,column=0)
        tk.Button(self.label_panel,text="yello").grid(row=1,column=1)
        tk.Button(self.label_panel,text="red").grid(row=1,column=2)


        self.image_panel.bind_all("<MouseWheel>", self._on_mousewheel)
        # with Windows OS
        # with Linux OS
        root.bind("<Button-4>", self._on_mousewheel)
        root.bind("<Button-5>", self._on_mousewheel)
        self.image_panel.bind("<Key>", self.key)




        self.image_panel.focus_set()

#        self.image_panel.place(relx=.5, rely=.4, anchor="c")

    def _on_mousewheel(self, event):
        print ("mouse detected ")
        print event.delta
        self.svvitch_cb()

    def key(self,event):
        print "pressed", repr(event.keysym)
        if(event.keysym=="Right"):
            self.img_display_idx =min(self.img_display_idx+1, self.num_imgs-1)
        if(event.keysym=="Left"):
            self.img_display_idx=max(0,self.img_display_idx-1)
        self.svvitch_cb()

    def classify_images(self):
        self.images=[os.path.join(self.images_path,im) for im in os.listdir(self.images_path)]
        self.num_imgs=len(self.images)
        #for image in self.images: print image
        self.results = image_forward(self.images, "trained_models/mobilenet050", 28, batch_size=2)
        self.show_images()



    def select_directory(self):
        self.images_path= tkFileDialog.askdirectory(initialdir="../data/demo",title='Choose a directory')
        self.select_folder['text']="Browse new"
        self.classify_button.grid(row=0,column=2)




if __name__ == '__main__':
    root=tk.Tk()
    AS=AssistedSuperviser(parser,root)

    root.mainloop()
    root.destroy() # optional; see description below
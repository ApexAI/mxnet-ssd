import mxnet as mx
import os
import tkFileDialog
import Tkinter as tk
from nexar_forward import image_forward
from PIL import Image, ImageTk
import platform
import subprocess
import ttk

opsys = platform.system()
dir_path = os.path.dirname(os.path.realpath(__file__))



class AssistedSupervisor():
    def __init__(self,master):
        self.img_display_idx=0
        self.master=master
        self.labels={"green":0,"yellow":1,"red":2,"none":3}

        self.predicted_label_imgs={}
        self.corrected_label_imgs={}

        master.grid_rowconfigure(0, weight=1)
        master.grid_rowconfigure(2, weight=1)
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(2, weight=1)

        images_root="supervisor_static"
        self.predicted_label_imgs['red'] = ImageTk.PhotoImage(Image.open(os.path.join(images_root,"red_predicted.png")))
        self.predicted_label_imgs['yellow'] = ImageTk.PhotoImage(Image.open(os.path.join(images_root,"yellow_predicted.png")))
        self.predicted_label_imgs['green'] = ImageTk.PhotoImage(Image.open(os.path.join(images_root,"green_predicted.png")))
        self.predicted_label_imgs['none'] = ImageTk.PhotoImage(Image.open(os.path.join(images_root,"undefined_predicted.png")))

        self.corrected_label_imgs['red'] = ImageTk.PhotoImage(Image.open(os.path.join(images_root,"red_corrected.png")))
        self.corrected_label_imgs['yellow'] = ImageTk.PhotoImage(Image.open(os.path.join(images_root,"yellow_corrected.png")))
        self.corrected_label_imgs['green'] = ImageTk.PhotoImage(Image.open(os.path.join(images_root,"green_corrected.png")))
        self.corrected_label_imgs['none'] = ImageTk.PhotoImage(Image.open(os.path.join(images_root,"undefined_corrected.png")))

        self.master.minsize(width=1000, height=800)
        self.image_panel = tk.LabelFrame(self.master, width=500, height=500, padx=10, pady=10)
        self.image_panel.grid(row=0,column=0)
        self.image_panel.grid_propagate(0)

        self.image_panel.grid_rowconfigure(0, weight=1)
        self.image_panel.grid_rowconfigure(2, weight=1)
        self.image_panel.grid_columnconfigure(0, weight=1)
        self.image_panel.grid_columnconfigure(2, weight=1)

        self.control_panel = tk.Frame(self.master, width=1000, height = 1000, background="#b22222")
        self.control_panel.grid(row=2,column=0)

        self.quit_button = tk.Button(
            self.control_panel, text="QUIT", fg="red", command=self.control_panel.quit
            )
        self.quit_button.grid(row=0,column=0)

        self.select_folder_button=tk.Button(self.control_panel, text='Select image folder', command=self.select_directory)
        self.select_folder_button.grid(row=0,column=1)

        self.select_model_button=tk.Button(self.control_panel, text='Select model', command=self.select_model)
        self.select_model_button.grid(row=0,column=2)

        self.classify_button=tk.Button(self.control_panel, state=tk.DISABLED, text='Classify', command=self.classify_images)
        self.classify_button.grid(row=0,column=3)

        self.save_button=tk.Button(self.control_panel, state=tk.DISABLED, text='Save', command=self.save_lst_file)
        self.save_button.grid(row=0,column=4)

        self.label_display_panel = tk.LabelFrame(self.master, width=400, height=500, padx=10, pady=10)
        self.label_display_panel.grid(row=0,column=1)


    def correct_prediction(self, correction):
        print(correction)
        self.results[self.img_display_idx]['prediction']=correction
        self.prediction_display.config(image=self.corrected_label_imgs[correction])
        self.prediction_display.image= self.corrected_label_imgs[correction]

    def update_display(self):
        image = Image.open(self.images[self.img_display_idx])
        image.thumbnail((400,400), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_display.config(image=photo)
        self.image_display.image = photo

        self.prediction_display.config(image=self.predicted_label_imgs[self.results[self.img_display_idx]['prediction']])
        self.prediction_display.image= self.predicted_label_imgs[self.results[self.img_display_idx]['prediction']]

        self.image_name_label.config(text=os.path.basename(self.images[self.img_display_idx]))

    def show_images(self):

        image = Image.open(self.images[0])
        image.thumbnail((400,400), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_display = tk.Label(self.image_panel, image=photo)
        self.image_display.image = photo
        self.image_display.place(relx=.5,rely=.5)
        self.image_name_label=tk.Label(self.master, text=os.path.basename(self.images[0]))
        self.image_name_label.grid(row=1,column=0)

        self.prediction_display=tk.Label(self.label_display_panel,image=self.predicted_label_imgs[self.results[0]['prediction']])#self.label_display_panel,text=self.results[0]['prediction'], width=50, height=50)
        self.prediction_display.image=self.predicted_label_imgs[self.results[0]['prediction']]
        self.prediction_display.grid(row=0,column=0, columnspan=4)

        tk.Label(self.label_display_panel,text="Please correct if the prediction is wrong: ").grid(row=1,column=0,columnspan=4)
        tk.Button(self.label_display_panel, height = 3, width = 18, bg="#44fa05", text="green", command=lambda :self.correct_prediction("green")).grid(row=2,column=0)
        tk.Button(self.label_display_panel, height = 3, width = 18, bg="#fff416", text="yellow", command=lambda:self.correct_prediction("yellow")).grid(row=2,column=1)
        tk.Button(self.label_display_panel, height = 3, width = 18, bg="#fb4d4d", text="red", command=lambda:self.correct_prediction("red")).grid(row=2,column=2)
        tk.Button(self.label_display_panel, height = 3, width = 18, background="#5edbf7", text="undefined", command=lambda:self.correct_prediction("none")).grid(row=2,column=3)

        # with Windows OS
        self.image_panel.bind_all("<MouseWheel>", self._on_mousewheel)
        # with Linux OS
        root.bind("<Button-4>", self._on_mousewheel)
        root.bind("<Button-5>", self._on_mousewheel)

        self.image_panel.bind("<Key>", self._on_key)

        self.image_panel.focus_set()


    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta == -120:
            self.img_display_idx =min(self.img_display_idx+1, self.num_imgs-1)
        if event.num == 4 or event.delta == 120:
            self.img_display_idx=max(0,self.img_display_idx-1)
        self.update_display()

    def _on_key(self,event):
        if(event.keysym=="Right"):
            self.img_display_idx =min(self.img_display_idx+1, self.num_imgs-1)
        if(event.keysym=="Left"):
            self.img_display_idx=max(0,self.img_display_idx-1)
        self.update_display()

    def classify_images(self):
        def compare_filename(s1, s2):
            return cmp(int(s1[0:len(s1) - 4]), int(s2[0:len(s2) - 4]))

        try:
            print "try"
            self.images = sorted(os.listdir(self.images_path),cmp=compare_filename)
        except:
            print "except"
            self.images=sorted(os.listdir(self.images_path))

        self.images=[os.path.join(self.images_path,im) for im in self.images]


        self.num_imgs=len(self.images)
        batch_size=128 if(self.num_imgs>512) else self.num_imgs/4

        self.results = image_forward(self.images, self.model_prefix, self.model_epoch, batch_size=batch_size)
        self.show_images()
        self.save_button.config(state="normal")


    def reset(self):
        self.img_display_idx=0
        self.num_imgs=0
        self.results=0
        self.images=[]
        self.images_path=""
        if(hasattr(self,'image_display')):
            self.image_display['image']=""
            self.image_display.image=""

        if(hasattr(self,'image_name_label')):
            self.image_name_label['text']=""

        tk.Tk.update(self.master)


    def select_directory(self):
        self.reset()
        self.images_path= tkFileDialog.askdirectory(initialdir=".",title='Choose a directory')
        self.select_folder_button['text']="Select new directory"

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




    def save_lst_file(self):
        lst_data=""
        for idx, img in enumerate(self.images):
            lst_data+=str(idx)+"\t"+ str(self.labels[self.results[idx]['prediction']])  +"\t"+img+"\n"

        filename= tkFileDialog.asksaveasfilename(initialdir=".", title="Save the list file")
        if filename != "":
            with open(filename+".lst", "w") as text_file:
                text_file.write(lst_data)

            subprocess.check_call(["python",
                                   "im2rec.py",
                                   filename, ".",
                                   "--pack-label", "1",
                                   "--resize","32"])




if __name__ == '__main__':

    root=tk.Tk()
    AS=AssistedSupervisor(root)

    root.mainloop()
    root.destroy() # optional; see description below
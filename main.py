"""Main file with stylizing application class."""
import os
import tkinter
from tkinter.filedialog import askopenfilename
from glob import glob
import torch
import cv2
import numpy as np
from PIL import ImageTk, Image
from utils import get_net, create_circular_mask, preprocess, postprocess

class MainApplication(tkinter.Frame):
    """A class for style app."""

    def __init__(self, parent, img_path, nets, *args, **kwargs):
        """Init app."""
        tkinter.Frame.__init__(self, parent, *args, **kwargs)
        self.root = ROOT
        self.img_path = img_path
        self.options = nets
        self.radius = 15
        self.canvas = tkinter.Canvas(ROOT, bg="white", height=100, width=100)
        self.oval_id = None
        self.img_id = None
        self.lambd = None
        self.height = None
        self.width = None
        self.x_cor, self.y_cor = None, None

        self.inv_btn = tkinter.Button(ROOT, text='Invert', command=self.invert_pic)
        self.err_btn = tkinter.Button(ROOT, text='Erase', command=self.erase)
        self.file_btn = tkinter.Button(ROOT, text='Choose picture', command=self.change_pic)
        self.style_btn = tkinter.Button(ROOT, text='Change style', command=self.change_style)
        self.save_btn = tkinter.Button(ROOT, text='Save result', command=self.save_pic)

        self.weights_path = tkinter.StringVar(ROOT)
        self.weights_path.set(self.options[0])  # default value

        self.opt = tkinter.OptionMenu(ROOT, self.weights_path, *self.options)
        self.intensity = tkinter.DoubleVar()
        self.intensity.set(75)
        self.kernel = create_circular_mask(self.radius)
        self.scale = tkinter.Scale(ROOT, variable=self.intensity)
        self.canvas.config(cursor="plus")
        self.canvas.pack(anchor=tkinter.E, side=tkinter.RIGHT)
        self.file_btn.pack(anchor=tkinter.W, side=tkinter.TOP, padx=10, pady=10)
        self.inv_btn.pack(anchor=tkinter.W, side=tkinter.TOP, padx=10, pady=10)
        self.err_btn.pack(anchor=tkinter.W, side=tkinter.TOP, padx=10, pady=10)
        self.save_btn.pack(anchor=tkinter.W, side=tkinter.TOP, padx=10, pady=10)
        self.style_btn.pack(anchor=tkinter.W, side=tkinter.TOP, padx=10, pady=10)
        self.opt.pack(anchor=tkinter.W, side=tkinter.TOP, padx=10, pady=10)
        self.scale.pack(anchor=tkinter.W, side=tkinter.TOP, padx=10, pady=10)

        self.canvas.bind("<Motion>", self.on_left_mouse_move)
        self.canvas.bind("<B1-Motion>", self.on_left_mouse_down)
        self.canvas.bind("<B2-Motion>", self.on_right_mouse_down)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.get_style()
        self.update()

    def get_style(self):
        """Update image, style image."""
        self.img_orig = cv2.imread(self.img_path)[:, :, ::-1].copy()
        img, h_new, w_new = preprocess(self.img_orig)
        if self.height != h_new or self.width != w_new:
            self.lambd = np.ones((h_new, w_new))
        self.height = h_new
        self.width = w_new
        net = get_net(self.weights_path.get())
        with torch.no_grad():
            res = net(img)
        self.stylized = postprocess(res, self.height, self.width)

    def update(self):
        """Canvas rendering."""
        rad = self.radius
        x_cor, y_cor = self.x_cor, self.y_cor
        lambd = self.lambd[:, :, np.newaxis]
        result = Image.fromarray(
            np.round(self.stylized * lambd + self.img_orig * (1 - lambd)).astype(
                "uint8"))

        self.height, self.width = self.lambd.shape
        self.result = ImageTk.PhotoImage(result)
        self.canvas.config(height=self.height, width=self.width)

        if self.oval_id is not None:
            self.canvas.delete(self.oval_id)
        if self.img_id is None:
            self.img_id = self.canvas.create_image(0, 0, image=self.result, anchor=tkinter.NW)
        else:
            self.canvas.itemconfig(self.img_id, image=self.result)
        if self.x_cor is not None and self.y_cor is not None:
            self.oval_id = self.canvas.create_oval(x_cor - rad,
                                                   y_cor - rad,
                                                   x_cor + rad,
                                                   y_cor + rad)

        self.canvas.update_idletasks()


    def on_left_mouse_down(self, event):
        """Manipulate an image in canvas."""
        rad = self.radius
        self.x_cor, self.y_cor = event.x, event.y
        x_cor, y_cor = self.x_cor, self.y_cor
        height, width = self.height, self.width
        scale = self.intensity.get() / 100
        self.lambd[max(y_cor - rad, 0): min(y_cor + rad, height),
                   max(x_cor - rad, 0): min(x_cor + rad, width)] -= self.kernel[
                       -min(y_cor - rad, 0): 2 * rad + min(height - (y_cor + rad), 0),
                       -min(x_cor-rad, 0): 2 * rad + min(width - (x_cor + rad), 0)] * scale
        self.lambd = self.lambd.clip(0, 1)
        self.update()

    def on_right_mouse_down(self, event):
        """Eraser tool."""
        rad = self.radius
        self.x_cor, self.y_cor = event.x, event.y
        x_cor, y_cor = self.x_cor, self.y_cor
        height, width = self.height, self.width
        self.lambd[max(y_cor - rad, 0): min(y_cor + rad, height),
                   max(x_cor - rad, 0): min(x_cor + rad, width)] += self.kernel[
                       -min(y_cor - rad, 0):2 * rad + min(height - (y_cor + rad), 0),
                       -min(x_cor - rad, 0): 2 * rad + min(width - (x_cor + rad), 0)]
        self.lambd = self.lambd.clip(0, 1)
        self.update()

    def on_left_mouse_move(self, event):
        """Manipulate area figure rendering."""
        self.x_cor, self.y_cor = event.x, event.y
        self.update()

    def invert_pic(self):
        """Invert stylized and original images."""
        self.lambd = 1 - self.lambd
        self.update()

    def erase(self):
        """Undo all the progress."""
        self.lambd = 1 + 0 * self.lambd
        self.update()

    def change_pic(self):
        """Open dialogue to choose new image."""
        self.img_path = askopenfilename()
        self.get_style()
        self.update()

    def change_style(self):
        """Restyle an image by chosen weight from drop-down menu."""
        self.get_style()
        self.update()

    def mouse_wheel(self, event):
        """Change manipulation area size."""
        self.radius += event.delta
        self.radius = min(self.radius, min(self.height, self.width))
        self.radius = max(self.radius, 1)
        self.kernel = create_circular_mask(self.radius)
        self.update()

    def save_pic(self):
        """Open dialogue to save current result."""
        file = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".png")
        if file:
            lambd = self.lambd[:, :, np.newaxis]
            img = Image.fromarray(
                np.round(self.stylized * lambd + self.img_orig * (1 - lambd)).clip(0, 255).astype(
                    "uint8"))

            abs_path = os.path.abspath(file.name)
            img.save(abs_path)


if __name__ == '__main__':
    ROOT = tkinter.Tk()
    MainApplication(ROOT,
                    'images/grape.jpg',
                    sorted(glob('./saved_models/*'))
                   ).pack(side="top", fill="both", expand=True)
    ROOT.mainloop()

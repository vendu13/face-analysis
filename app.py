from tkinter import filedialog
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from analyzer import module_1, module_2, module_3

file_path = -1
cap = cv2.VideoCapture(0)

def load_camera():

    r, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame)
    resized = img_pil.resize((250, 200), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(resized)
    image_view.config(image=img)
    image_view.image = img
    image_view.after(20, load_camera)

def browse_file():
    global file_path
    file_path = filedialog.askopenfilename(initialdir='/',
                                           title='Select The Image',
                                           filetype=(('JPG Files', '*.jpg'),('all files', '*.*')))
    img_pil = Image.open(file_path)
    resized = img_pil.resize((250, 200), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(resized)
    image_view.config(image=img)
    image_view.image = img

def analyze_image():
    analysis_text.config(text='Clicked')
    if file_path == -1:
        analysis_text.config(text='File Not Selected!!')
    else:
        analysis_text.config(text='Analysing...')
        obj1 = module_1()
        result_1 = obj1.analysis(file_path)
        obj2 = module_2()
        result_2 = obj2.detect_and_predict_specs_mask(file_path)
        obj3 = module_3()
        result_3 = obj3.detect_and_predict_hair_beard(file_path)
        analysis_text.config(text=(result_1 + result_2 + result_3))

def click_image():

    if True:
        r, frame = cap.read()
        cv2.imwrite('Images/image.jpg', frame)

        cap.release()

        global file_path
        file_path = 'Images/image.jpg'

        img_pil = Image.open(file_path)
        resized = img_pil.resize((250, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(resized)
        image_view.config(image=img)
        image_view.image = img


root = tk.Tk()
root.geometry('500x500')
root.title('Face Analyzer')
root.pack_propagate(False)
root.resizable(0,0)

image_frame = tk.LabelFrame(root, text='Image View:')
image_frame.place(height=200, width=250)

load_file_frame = tk.LabelFrame(root, text='File Path:')
load_file_frame.place(height=50, width=500, rely=0.4)

analysis_frame = tk.LabelFrame(root, text='Analysis:')
analysis_frame.place(height=200, width=250, relx=0.5)

record_frame = tk.LabelFrame(root, text='Analysis Records:')
record_frame.place(height=220, width=500, rely=0.56)

load_camera_btn = tk.Button(load_file_frame, text='Load Camera', command=load_camera)
load_camera_btn.place(width=80, relx=0.14)

click_photo_btn = tk.Button(load_file_frame, text='Click Image', command=click_image)
click_photo_btn.place(width=70, relx=0.42)

load_photo_btn = tk.Button(load_file_frame, text='Load Photo', command=browse_file)
load_photo_btn.place(width=70, relx=0.7)

analyse_btn = tk.Button(root, text='Analyse Face', command=analyze_image)
analyse_btn.place(height=30, width=500, rely=0.5)

analysis_text = tk.Label(analysis_frame, text='')
analysis_text.place(height=250, width=250)

image_view = tk.Label(image_frame, text='Selected Image Preview')
image_view.place(height=200, width=250)

# button = tkinter.Button(root, text="Exit", width=40, command=root.destroy)
# button.pack()

root.mainloop()
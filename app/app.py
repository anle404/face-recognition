import gradio as gr
import utils as u
from model import get_pt_encoder

encoder = get_pt_encoder()

def register(name, img1, img2, img3):
    if not (name):
        return "Please enter name"

    imgs = [img1, img2, img3]
    msg = u.add_person(name, imgs, encoder)

    if (msg == ""):
        return "Register Successfully"
    else:
        return msg

def checkin(img):
    name = u.get_person_name(img, encoder)
    
    if (name == ""):
        return "Attendance Check Failed"
    else:
        return f"Attendance Check Successfully! Hello, {name}."



#interface 1
app1 =  gr.Interface(fn = register, inputs=["text",gr.Image(type="numpy"),gr.Image(type="numpy"),gr.Image(type="numpy")], outputs="text")
#interface 2

# place your code here
app2 =  gr.Interface(fn = checkin, inputs=gr.Image(type="numpy"), outputs="text", live=False)

demo = gr.TabbedInterface([app1, app2], ["Register Your Attendance", "Check Your Attendance"], title = "Face Recognition System")

demo.launch(debug=True)
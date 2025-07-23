
import gradio as gr
import os
import torch

from model import creat_effnetb1
from timeit import default_timer as timer
from typing import Tuple, Dict

with open(garbage_Classification_class_names, "r") as f:
  class_names = [garbage.strip() for garbage in f.readlines()]

effnetb1 , transform = creat_effnetb1(num_classes = len(class_names),
                                      seed = 42)

effnetb1.load_state_dict(
    torch.load(
        f = "02_efficientnet_b1_pretrained_model.pth",
        map_location= torch.device("cpu")
    )
)

def prdict(image) -> Tuple[Dict, float]:

  start_time = timer()

  image = transform(image).unsqueeze(dime = 1)

  effnetb1.eval()

  with torch.inference_mode():
    pred_prob = torch.softmax(effnetb1(image), dime= 1)

  pred_labels_and_probs = {class_names[i]: float(pred_prob[0][i]) for i in range(len(class_names))}

  pred_time = round(timer() - start_time, 5)

  return pred_labels_and_probs, pred_time


title = "Garbage Classification"
description = "An EfficientNetB1 feature extractor computer vision model to classify images of garbage into [6 different classes]."
article = "Created by Esmail khosravi(ir25) in 2025/7/22."

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(
    fn=prdict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

demo.launch()

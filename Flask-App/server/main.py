from flask import Flask, request
from flask_cors import CORS
from flask import render_template
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from resize import to_square_save
import numpy as np

def predict_on_image(class_names, model_path, img):

  model = torchvision.models.efficientnet_b0().to("cpu")

  model.classifier = torch.nn.Sequential(
      torch.nn.Dropout(p=0.2, inplace=True), 
      torch.nn.Linear(in_features=1280, 
                      out_features=len(class_names), 
                      bias=True))

  model.load_state_dict(torch.load(Path(model_path), map_location=torch.device('cpu')))
  model.eval()

  with torch.inference_mode():
    preds = model(img.unsqueeze(dim=0).to('cpu'))

  target_image_pred_probs = torch.softmax(preds, dim=1)
  class_idx = torch.argmax(target_image_pred_probs, dim=1)
  max_prop = target_image_pred_probs.max().cpu()
  title = f"Pred: {class_names[class_idx.cpu()]} | Prob: {max_prop:.3f}"

  return title



app = Flask(__name__)
cors = CORS(app) #Request will get blocked otherwise on Localhostyz
@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
 
  img_path = request.files['file']
  img = to_square_save(img_path)
  img.save(Path("static/cropped_img.png"))
  rgb_image = img.convert('RGB')
  custom_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])
  transformed_image = custom_transform(rgb_image)
  
  title = predict_on_image(["bad", "good"], "tr_v0_64_1e5.pth", transformed_image)
  # title = title + "\n" + predict_on_image(["Ear", "Nose", "Vocal Folds"], "tr_v5_32_5e6_5e5.pth", transformed_image)
  if "bad" in title:
     return title
  else:
     title = predict_on_image(["Ear", "Nose", "Vocal Folds"], "tr_v5_64_clean.pth", transformed_image)
     
  return title


if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)




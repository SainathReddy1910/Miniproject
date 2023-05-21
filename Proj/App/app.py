import pickle
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
import sys
import time

modell = pickle.load(open('model.pkl','rb'))

from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file=Image.open(file)
        file.resize((400,300))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

'''@app.route("/predict",method=['GET','POST'])
def predict(image_path,model):
    #path = "Model\modelFinal.pth"
    img_path = pat
    global loaded_model
    loaded_model = self.load_checkpoint(model)
    img = self.process_image(img_path)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)
    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
    # Running image through network
        output = loaded_model.forward(img_add_dim)
        #conf, predicted = torch.max(output.data, 1)   
        probs_top = output.topk(topk)[0]
        predicted_top = output.topk(topk)[1]
        # Converting probabilities and outputs to lists
        conf = np.array(probs_top)[0]
        predicted = np.array(predicted_top)[0]
        #return probs_top_list, index_top_list
        return conf, predicted

def load_checkpoint(self,filepath):

        #checkpoint = torch.load(filepath,map_location='cpu') #unka
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        
        #model.load_state_dict(checkpoint['state_dict'])
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model

    def find_classes(dir):

        classes = os.listdir(dir)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    test_dir="Dataset\Dataset\Test"
    global classes
    global c_to_idx
    classes, c_to_idx = find_classes(test_dir)
'''

'''def process_image(self,image):
    
        # Process a PIL image for use in a PyTorch model

        # Converting image to PIL image using image file path
        pil_im = Image.open(f'{image}' )

        # Building image transform
        transform = transforms.Compose([transforms.Resize((244,244)),
                                        #transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])]) 
        
        # Transforming image for use with network
        pil_tfd = transform(pil_im)
        
        # Converting to Numpy array 
        array_im_tfd = np.array(pil_tfd)
        
        return array_im_tfd  '''

'''#ACCURACY
@app.route("/accuracy",method=['POST','GET'])
def acc(self,model):
    correct = 0
    total = 0
    model = model.to('cpu')
    predlist=torch.zeros(0,dtype=torch.long,device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long,device='cpu')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cpu'), labels.to('cpu')
            # Get probabilities
            outputs = model(images)
            # Turn probabilities into predictions
            _, predicted_outcome = torch.max(outputs.data, 1)
            # Total number of images
            total += labels.size(0)
            # Count number of cases in which predictions are correct
            correct += (predicted_outcome == labels).sum().item()
                
        Acc = round(100 * correct / total,3)
        return Acc

 def show_acc(self):
        modelpath="Model\modelFinal.pth"
        model = self.load_checkpoint(modelpath)
        global Accuracy
        Accuracy = self.Cal_Accuracy(model) '''

if __name__== '__main__':
    app.run(debug=True)
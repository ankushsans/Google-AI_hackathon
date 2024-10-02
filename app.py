import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models

# Initialize the Flask app
app = Flask(__name__)

# Path to the saved model
MODEL_PATH = './resnet50_best_model.pth'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Image upload folder
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # Assuming 4 classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Image transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class names (update according to your dataset)
class_names = ['ALL-BENIGN', 'ALL-EARLY', 'ALL-PRE', 'ALL-PRO']

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predict the image class
def predict_image(image_path):
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        return class_names[pred]

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Upload and predict
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict the image class
        predicted_class = predict_image(filepath)

        return render_template('result.html', predicted_class=predicted_class, image_url=filepath)
    
    return redirect(url_for('home'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


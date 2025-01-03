import torch
from flask import Flask, request, render_template
import io
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.serialization

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(32 * 8 * 8, 60)
        self.fc2 = nn.Linear(60, 2)
        self.dropout_fc = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
        
FILE = 'model/model.pth'
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html', prediction_text="")
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return render_template('index.html', prediction_text="No file uploaded!")
    type= request.form.get('model')
    if not type:
        return render_template('index.html',prediction_text='No model chosen')
    
    
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    if request.form['model']=='cnn':
        model = Net()
        model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    else:
        from torchvision.models import resnet18
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)
        device = torch.device("cpu")
        model=resnet18(pretrained=False)
        model.fc=nn.Linear(model.fc.in_features,2)
        model.load_state_dict(torch.load('model/modelres.pth', map_location=device))
        model=model.to(device)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        result = 'Dog' if predicted.item() == 1 else 'Cat'
    return render_template('index.html', prediction_text=f"The uploaded image is classified as: {result}")

if __name__ == '__main__':
    app.run(debug=True)



import cv2
import torch

import torch.nn as nn
import numpy as np
import imutils
import pickle
import os
from torchvision import transforms
import timm
from flask import Flask, render_template, Response,request,redirect,url_for,send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and other resources
# model_name = 'resnet50'
model_name = 'senet154'


le_path = 'label_encoder.pickle'
encodings = 'encoded_faces.pickle'
detector_folder = 'face_detector'
confidence_threshold = 0.5  # Threshold for face detection confidence


# # Load the entire model
# model_name = torch.load(r'C:\Users\prata\OneDrive\Documents\sih\Face-Liveness-Detection-Anti-Spoofing-Web-App\model2.pt')

# # Set the model to evaluation mode
# model_name.eval()


args = {
    'le': le_path,
    'detector': detector_folder,
    'encodings': encodings,
    'confidence': confidence_threshold
}

# Load the encoded faces and names
with open(args['encodings'], 'rb') as file:
    encoded_data = pickle.loads(file.read())

# Load the face detector
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Load the pretrained liveness detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
liveness_model = timm.create_model(model_name, pretrained=True, num_classes=2)
liveness_model.eval()
liveness_model.to(device)  # Move model to the appropriate device

# Load the label encoder
le = pickle.loads(open(args['le'], 'rb').read())

# Define preprocessing transforms for the PyTorch model
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match the model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_frames():
    cap = cv2.VideoCapture(0)  # Initialize the webcam

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frm = imutils.resize(frame, width=800)  # Resize the frame for faster processing
            (h, w) = frm.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            detector_net.setInput(blob)
            detections = detector_net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > args['confidence']:
                    # Calculate bounding box for detected face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')

                    # Adjust bounding box for face extraction
                    startX = max(0, startX - 20)
                    startY = max(0, startY - 20)
                    endX = min(w, endX + 20)
                    endY = min(h, endY + 20)

                    # Extract the face from the frame
                    face = frm[startY:endY, startX:endX]

                    try:
                        face_tensor = preprocess(face).unsqueeze(0).to(device)  # Preprocess and move to device
                    except Exception as e:
                        print(f'[ERROR] {e}')
                        continue

                    with torch.no_grad():
                        preds = liveness_model(face_tensor).cpu().numpy()[0]  # Get predictions from the model
                        
                    # Determine the predicted label
                    j = np.argmax(preds)
                    label_name = le.classes_[j]
                    label = f'{label_name}: {preds[j]:.4f}'
                    print(f'[INFO] {label_name}')

                    # Display results on the frame
                    color = (0, 0, 255) if label_name == 'fake' else (0, 255, 0)  # Red for fake, Green for real
                    cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
                    cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
                    cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frm)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    return redirect(url_for('upload_faces'))

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', title="About")
@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html', title="Contact")

@app.route('/upload_faces', methods=['POST', 'GET'])
def upload_faces():
    global content, graph
    # Display Page
    if request.method == 'GET':
        return render_template('upload_faces.html')
    
@app.route('/start_video')
def start_video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


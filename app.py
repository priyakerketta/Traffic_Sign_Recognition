import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from PIL import Image


from data import classes

# Download Model File
model = load_model("traffic_classifier.h5")


# Create folder to save images temporarily
if not os.path.exists('./static/test'):
    os.makedirs('./static/test')

def predict(test_dir):
    test_img = [f for f in os.listdir(os.path.join(test_dir)) if not f.startswith(".")]
    test_df = pd.DataFrame({'Image': test_img})
    
    pred = []
    for image in test_df['Image'].values:
        data = Image.open(os.path.join(test_dir, image))
        data = data.resize((40, 40))
        a = np.array(data)
        b = []
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                b.append(a[i][j][:3])
        b = np.array(b)
        b = b.reshape(40, 40, 3)
        pred.append(b)
    pred = (np.array(pred)) / 255
    
    pred_y = model.predict(pred)
    test_df['Label'] = np.argmax(pred_y, axis = -1)
    test_df['Label'] = test_df['Label'].add(1)
    test_df['Label'] = test_df['Label'].replace(classes)
    
    prediction_dict = {}
    for value in test_df.to_dict('index').values():
        image_name = value['Image']
        image_prediction = value['Label']
        prediction_dict[image_name] = {}
        prediction_dict[image_name]['prediction'] = image_prediction
    return prediction_dict


# Create an app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # maximum upload size is 50 MB
app.secret_key = "trafficclassifier"
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg'}
folder_num = 0
folders_list = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])

def get_signal():
    global folder_num
    global folders_list
    if request.method == 'POST':
        if folder_num >= 1000000:
            folder_num = 0
        # check if the post request has the file part
        if 'hiddenfiles' not in request.files:
            flash('No files part!')
            return redirect(request.url)
        # Create a new folder for every new file uploaded, 
        # so that concurrency can be maintained
        files = request.files.getlist('hiddenfiles')
        app.config['UPLOAD_FOLDER'] = "./static/test"
        app.config['UPLOAD_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/predict_' + str(folder_num).rjust(6, "0")
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            folders_list.append(app.config['UPLOAD_FOLDER'])
            folder_num += 1
        for file in files:
            if file.filename == '':
                flash('No Files are Selected!')
                return redirect(request.url) 
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                flash("Invalid file type! Only PNG, JPEG/JPG files are supported.")
                return redirect('/')
        try:
            if len(os.listdir(app.config['UPLOAD_FOLDER'])) > 0:
                signals = predict(app.config['UPLOAD_FOLDER'])
                return render_template('show_prediction.html', folder = app.config['UPLOAD_FOLDER'], predictions = signals)
        except:
            return redirect('/')
        
    return render_template('index.html')


    
app.run(debug = True)
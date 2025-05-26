from flask import Flask, render_template, request, redirect, url_for, session, flash , jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
import base64
from io import BytesIO
from tensorflow.keras.models import load_model
from keras.losses import mean_squared_error

# Load the model (include custom loss if required)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

# Load the models and weights
Cmodel = load_model('model/Cmodel.h5', custom_objects={'dice_loss': dice_loss})
Cmodel.load_weights('model/Cbest_model.hdf5')

Nmodel = load_model('model/Nmodel.h5', custom_objects={'dice_loss': dice_loss})
Nmodel.load_weights('model/Nbest_model.hdf5')

app = Flask(__name__)

# Set up the database URI (SQLite in this case)
# SQLite file for local development
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Secret key for session encryption
app.secret_key = 'your_secret_key'

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Ensure the upload folder is inside 'static/uploads'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# User model


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    sketches = db.relationship('Sketch', backref='user', lazy=True)

# Sketch model


class Sketch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


with app.app_context():
    db.create_all()


@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/user_portal')
def user_portal():
    if 'username' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['username']).first()
    if user:
        sketches = Sketch.query.filter_by(user_id=user.id).all()
        return render_template('user_portal.html', user=user, sketches=sketches)
    return redirect(url_for('login'))


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['username']).first()

    if request.method == 'POST':
        new_email = request.form['email']
        new_password = request.form['password']
        user.email = new_email
        user.password = new_password
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('user_portal'))

    return render_template('edit_profile.html', user=user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another one.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please choose another one.', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/contact')
def contact():
    team_members = [
        {"name": "Prashant Acharya", "role": "Frontend/Backend",
            "github": "https://github.com/PrashantAcharya0"},
        {"name": "Pratik Acharya", "role": "Machine Learning",
            "github": "https://github.com/pratik-ac"},
        {"name": "Sakar Khanal", "role": "Research/Documentation",
            "github": "https://www.linkedin.com/in/sakar-khanal-63a108318/"},
        {"name": "Shisir Thapa", "role": "Frontend/Backend",
            "github": "https://github.com/Normi333"},
    ]
    return render_template("contact.html", team=team_members)


@app.route('/upload_sketch', methods=['POST'])
def upload_sketch():
    if 'username' not in session:
        flash('You must be logged in to upload a sketch.', 'danger')
        return redirect(url_for('login'))

    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('home'))

    file = request.files['file']
    selected_model = request.form.get('model')  # Get model selection from form input
    print(f"📌 Selected Model: {selected_model}")  # Debugging


    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('home'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load user
        user = User.query.filter_by(username=session['username']).first()
        if user:
            new_sketch = Sketch(filename=filename, user_id=user.id)
            db.session.add(new_sketch)
            db.session.commit()

        # Load and preprocess the image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (178, 218))  # Resize to match model input size
        img = img.astype(np.float32) / 255.0  # Normalize the image

        # Create a batch of 16 images
        img_batch = np.repeat(img[np.newaxis, ...], 16, axis=0)
        
        # Choose the model based on user selection
        if selected_model == "C":
            y_pred = Cmodel.predict(img_batch)
        else:
            y_pred = Nmodel.predict(img_batch)
            

        # Post-process the prediction
        y_pred = np.clip(y_pred[0], 0, 1)  # Ensure range [0, 1]


        # Convert to uint8 before encoding
        enhanced_color_img = (y_pred * 255).astype(np.uint8)
        enhanced_color_img_bgr = cv2.cvtColor(enhanced_color_img, cv2.COLOR_RGB2BGR)

        # Encode to base64
        _, buffer = cv2.imencode(".jpg", enhanced_color_img_bgr)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"prediction": encoded_image})

    else:
        flash('Invalid file format. Only images are allowed.', 'danger')
        return redirect(url_for('home'))



if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configurations
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plant_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)

# Load disease info JSON
with open('disease_info.json') as f:
    disease_info = json.load(f)

# Load Keras model
model = load_model("plant_disease_model_final_v2.h5", compile=False)
# Save again using SavedModel format

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define Upload model
class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.String(50), default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# Create DB tables
with app.app_context():
    db.create_all()
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Assuming model input 224x224
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    class_idx = np.argmax(preds)
    accuracy = float(np.max(preds) * 100)
    # Get class label
    class_labels = list(disease_info.keys())
    prediction = class_labels[class_idx]
    return prediction, round(accuracy, 2)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash("Email already registered!")
            return redirect(url_for('signup'))
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Signup successful! Login now.")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password!")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!")
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.timestamp.desc()).all()
    total_uploads = len(uploads)
    total_diseases = len([u for u in uploads if 'healthy' not in u.prediction])
    average_accuracy = round(np.mean([u.accuracy for u in uploads]) if uploads else 0, 2)
    recent_uploads = uploads[:6]  # Show 6 recent uploads
    return render_template('dashboard.html', current_user=session, total_uploads=total_uploads,
                           total_diseases=total_diseases, average_accuracy=average_accuracy,
                           recent_uploads=recent_uploads)

@app.route('/upload', methods=['GET','POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['leaf_image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            prediction, accuracy = predict_disease(save_path)
            new_upload = Upload(filename=filename, prediction=prediction, accuracy=accuracy, user_id=session['user_id'])
            db.session.add(new_upload)
            db.session.commit()
            return redirect(url_for('analysis', upload_id=new_upload.id))
        else:
            flash("Invalid file format!")
            return redirect(url_for('upload'))
    return render_template('upload.html')

@app.route('/analysis/<int:upload_id>')
def analysis(upload_id):
    upload_record = Upload.query.get_or_404(upload_id)
    rec = disease_info.get(upload_record.prediction, {})
    recommendation = f"Description: {rec.get('description','')}\nTreatment: {rec.get('treatment','')}\nPrevention: {rec.get('prevention','')}"
    return render_template('analysis.html', filename=upload_record.filename, prediction=upload_record.prediction,
                           accuracy=upload_record.accuracy, recommendation=recommendation)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.timestamp.desc()).all()
    return render_template('history.html', uploads=uploads)

@app.route('/about')
def about():
    return render_template('about.html')

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)
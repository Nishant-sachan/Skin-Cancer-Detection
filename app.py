# app.py - Main Flask application with authentication and database
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import uuid
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)
MODEL_PATH = "model/skin_model.keras"
LABEL_PATH = "model/labels.txt"

model = load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f if line.strip()]

# Figure out what image size the model expects (e.g. 224x224x3)
try:
    _, MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH, MODEL_IMG_CHANNELS = model.input_shape
except Exception:
    MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH, MODEL_IMG_CHANNELS = 224, 224, 3

# Keywords used to flag a predicted class as "high risk" (edit to match your labels.txt)
HIGH_RISK_KEYWORDS = ['malignant', 'melanoma', 'mel', 'bcc', 'akiec', 'carcinoma', 'cancer']

def is_high_risk_label(label):
    label_lower = label.lower()
    return any(keyword in label_lower for keyword in HIGH_RISK_KEYWORDS)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skin_cancer_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100))
    date_of_birth = db.Column(db.Date)
    phone = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    reports = db.relationship('AnalysisReport', backref='patient', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class AnalysisReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Image info
    filename = db.Column(db.String(200), nullable=False)
    image_width = db.Column(db.Integer)
    image_height = db.Column(db.Integer)
    file_size = db.Column(db.Float)
    
    # Analysis results
    asymmetry_score = db.Column(db.Float)
    color_variation = db.Column(db.Float)
    border_irregularity = db.Column(db.Float)
    avg_color_r = db.Column(db.Float)
    avg_color_g = db.Column(db.Float)
    avg_color_b = db.Column(db.Float)
    
    # Risk assessment
    risk_level = db.Column(db.String(20))
    risk_score = db.Column(db.Integer)
    risk_max_score = db.Column(db.Integer, default=7)
    risk_factors = db.Column(db.Text)
    recommendation = db.Column(db.Text)

    # Model prediction
    predicted_label = db.Column(db.String(100))
    model_confidence = db.Column(db.Float)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_skin_condition(image_path):
    """Run the trained Keras model (skin_model.keras) on the uploaded image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

        predictions = model.predict(img_array, verbose=0)[0]

        if len(predictions) == 1:
            # Binary model with a single sigmoid output
            prob = float(predictions[0])
            positive_label = labels[1] if len(labels) > 1 else labels[0]
            negative_label = labels[0]
            predicted_label = positive_label if prob >= 0.5 else negative_label
            confidence = (prob if prob >= 0.5 else 1 - prob) * 100
            all_predictions = [
                {'label': negative_label, 'confidence': round((1 - prob) * 100, 2)},
                {'label': positive_label, 'confidence': round(prob * 100, 2)}
            ]
        else:
            # Multi-class model with softmax output
            predicted_index = int(np.argmax(predictions))
            predicted_label = labels[predicted_index]
            confidence = float(predictions[predicted_index]) * 100
            all_predictions = [
                {'label': labels[i], 'confidence': round(float(predictions[i]) * 100, 2)}
                for i in range(len(predictions))
            ]

        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'success': True,
            'predicted_label': predicted_label,
            'confidence': round(confidence, 2),
            'all_predictions': all_predictions
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

def process_image_analysis(image_path):
    """Analyze skin lesion image using PIL"""
    try:
        original_image = Image.open(image_path)
        
        if original_image.mode != 'RGB':
            rgb_image = original_image.convert('RGB')
        else:
            rgb_image = original_image.copy()
        
        # Image processing
        enhancer = ImageEnhance.Contrast(rgb_image)
        contrast_enhanced = enhancer.enhance(1.2)
        blurred = contrast_enhanced.filter(ImageFilter.GaussianBlur(radius=1))
        img_array = np.array(blurred)
        
        # Color analysis
        avg_color = np.mean(img_array, axis=(0, 1))
        color_std = np.std(img_array, axis=(0, 1))
        
        # Asymmetry detection
        height, width = img_array.shape[:2]
        left_half = img_array[:, :width//2]
        right_half = np.fliplr(img_array[:, width//2:])
        
        if left_half.shape[1] != right_half.shape[1]:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        asymmetry_score = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        color_variation = np.mean(color_std)
        
        # Edge detection
        gray = rgb_image.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)
        border_irregularity = np.std(edge_array)
        
        # Run the actual trained model on the image
        model_prediction = predict_skin_condition(image_path)

        risk_assessment = calculate_risk_score({
            'asymmetry_score': asymmetry_score,
            'color_variation': color_variation,
            'border_irregularity': border_irregularity
        }, model_prediction)
        
        return {
            'success': True,
            'image_size': rgb_image.size,
            'avg_color': avg_color.tolist(),
            'color_variation': float(color_variation),
            'asymmetry_score': float(asymmetry_score),
            'border_irregularity': float(border_irregularity),
            'model_prediction': model_prediction,
            'risk_assessment': risk_assessment,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def calculate_risk_score(analysis_data, model_prediction=None):
    """Calculate risk score using image heuristics + the trained model's prediction"""
    score = 0
    risk_factors = []
    
    if analysis_data['asymmetry_score'] > 50:
        score += 3
        risk_factors.append("High asymmetry detected")
    elif analysis_data['asymmetry_score'] > 30:
        score += 1
        risk_factors.append("Moderate asymmetry")
    
    if analysis_data['color_variation'] > 40:
        score += 2
        risk_factors.append("Significant color variation")
    elif analysis_data['color_variation'] > 25:
        score += 1
        risk_factors.append("Some color variation")
    
    if analysis_data['border_irregularity'] > 30:
        score += 2
        risk_factors.append("Irregular borders detected")
    elif analysis_data['border_irregularity'] > 20:
        score += 1
        risk_factors.append("Some border irregularity")

    # --- Model-driven scoring (this is the main signal, heuristics above are supporting) ---
    max_score = 7
    if model_prediction and model_prediction.get('success'):
        label = model_prediction['predicted_label']
        confidence = model_prediction['confidence']
        max_score = 11

        if is_high_risk_label(label):
            if confidence > 75:
                score += 4
                risk_factors.append(f"AI model detected '{label}' with {confidence:.1f}% confidence")
            elif confidence > 50:
                score += 2
                risk_factors.append(f"AI model suggests possible '{label}' ({confidence:.1f}% confidence)")
            else:
                score += 1
                risk_factors.append(f"AI model weakly leans toward '{label}' ({confidence:.1f}% confidence)")
        else:
            risk_factors.append(f"AI model classification: '{label}' ({confidence:.1f}% confidence)")
    
    if score >= (0.65 * max_score):
        risk_level = "HIGH"
        recommendation = "Strongly recommend immediate dermatologist consultation"
        color_class = "danger"
    elif score >= (0.35 * max_score):
        risk_level = "MODERATE"
        recommendation = "Consider scheduling dermatologist appointment"
        color_class = "warning"
    else:
        risk_level = "LOW"
        recommendation = "Continue regular self-examinations"
        color_class = "success"
    
    return {
        'score': score,
        'max_score': max_score,
        'level': risk_level,
        'factors': risk_factors,
        'recommendation': recommendation,
        'color_class': color_class,
        'model_label': model_prediction.get('predicted_label') if model_prediction and model_prediction.get('success') else None,
        'model_confidence': model_prediction.get('confidence') if model_prediction and model_prediction.get('success') else None
    }

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        
        # Validation
        if not all([username, email, password, confirm_password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        # Create user
        user = User(username=username, email=email, full_name=full_name)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.full_name or user.username}!', 'success')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    reports = AnalysisReport.query.filter_by(user_id=current_user.id).order_by(AnalysisReport.created_at.desc()).all()
    return render_template('dashboard.html', reports=reports)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    current_user.full_name = request.form.get('full_name')
    current_user.phone = request.form.get('phone')
    
    dob = request.form.get('date_of_birth')
    if dob:
        current_user.date_of_birth = datetime.strptime(dob, '%Y-%m-%d').date()
    
    db.session.commit()
    flash('Profile updated successfully', 'success')
    return redirect(url_for('profile'))

@app.route('/analyze')
@login_required
def analyze():
    return render_template('analyze.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'warning')
        return redirect(url_for('analyze'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'warning')
        return redirect(url_for('analyze'))
    
    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            analysis_result = process_image_analysis(filepath)
            
            if analysis_result['success']:
                file_size = os.path.getsize(filepath) / 1024
                
                # Save report to database
                report = AnalysisReport(
                    user_id=current_user.id,
                    filename=filename,
                    image_width=analysis_result['image_size'][0],
                    image_height=analysis_result['image_size'][1],
                    file_size=file_size,
                    asymmetry_score=analysis_result['asymmetry_score'],
                    color_variation=analysis_result['color_variation'],
                    border_irregularity=analysis_result['border_irregularity'],
                    avg_color_r=analysis_result['avg_color'][0],
                    avg_color_g=analysis_result['avg_color'][1],
                    avg_color_b=analysis_result['avg_color'][2],
                    risk_level=analysis_result['risk_assessment']['level'],
                    risk_score=analysis_result['risk_assessment']['score'],
                    risk_max_score=analysis_result['risk_assessment']['max_score'],
                    risk_factors='|'.join(analysis_result['risk_assessment']['factors']),
                    recommendation=analysis_result['risk_assessment']['recommendation'],
                    predicted_label=analysis_result['risk_assessment']['model_label'],
                    model_confidence=analysis_result['risk_assessment']['model_confidence']
                )
                
                db.session.add(report)
                db.session.commit()
                
                flash('Analysis completed successfully', 'success')
                return redirect(url_for('view_report', report_id=report.id))
            else:
                flash(f'Analysis failed: {analysis_result["error"]}', 'danger')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(url_for('analyze'))
                
        except Exception as e:
            flash(f'Upload failed: {str(e)}', 'danger')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('analyze'))
    
    flash('Invalid file type', 'warning')
    return redirect(url_for('analyze'))

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    report = AnalysisReport.query.get_or_404(report_id)
    
    # Check if user owns this report
    if report.user_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    # Reconstruct analysis data
    analysis = {
        'image_size': [report.image_width, report.image_height],
        'asymmetry_score': report.asymmetry_score,
        'color_variation': report.color_variation,
        'border_irregularity': report.border_irregularity,
        'avg_color': [report.avg_color_r, report.avg_color_g, report.avg_color_b],
        'model_prediction': {
            'success': report.predicted_label is not None,
            'predicted_label': report.predicted_label,
            'confidence': report.model_confidence
        },
        'risk_assessment': {
            'level': report.risk_level,
            'score': report.risk_score,
            'max_score': report.risk_max_score or 7,
            'factors': report.risk_factors.split('|') if report.risk_factors else [],
            'recommendation': report.recommendation,
            'color_class': 'danger' if report.risk_level == 'HIGH' else ('warning' if report.risk_level == 'MODERATE' else 'success'),
            'model_label': report.predicted_label,
            'model_confidence': report.model_confidence
        },
        'analysis_date': report.created_at.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('results.html', 
                         filename=report.filename,
                         file_size=report.file_size,
                         analysis=analysis,
                         report=report)

@app.route('/report/<int:report_id>/delete', methods=['POST'])
@login_required
def delete_report(report_id):
    report = AnalysisReport.query.get_or_404(report_id)
    
    if report.user_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    # Delete image file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], report.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Delete from database
    db.session.delete(report)
    db.session.commit()
    
    flash('Report deleted successfully', 'success')
    return redirect(url_for('dashboard'))

@app.route('/report/<int:report_id>/notes', methods=['POST'])
@login_required
def update_notes(report_id):
    report = AnalysisReport.query.get_or_404(report_id)
    
    if report.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    notes = request.form.get('notes')
    report.notes = notes
    db.session.commit()
    
    flash('Notes updated successfully', 'success')
    return redirect(url_for('view_report', report_id=report_id))

@app.route('/about')
def about():
    return render_template('about.html')

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
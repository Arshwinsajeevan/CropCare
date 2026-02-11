import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

# ML + image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage

# notifications
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from twilio.rest import Client

# supabase
from supabase import create_client, Client as SupabaseClient

# load env
load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))

# ========== CONFIG ==========
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key-change-this")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///crop_disease.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

# Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Should be service_role key
supabase: SupabaseClient = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase")
else:
    print("⚠️ Supabase keys missing — cloud sync disabled.")

# ========== MODELS ==========
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    phone = db.Column(db.String(30), nullable=True)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(IST))
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    image_path = db.Column(db.String(300), nullable=False)
    disease = db.Column(db.String(200), nullable=False)
    certainty = db.Column(db.Integer, nullable=False)  # 0-100 integer
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(IST))

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'))
    type = db.Column(db.String(20))  # 'sms' or 'email'
    status = db.Column(db.String(20))  # 'sent', 'failed'
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(IST))

# ========== Load ML model & labels ==========
# ========== TFLITE MODEL ==========
MODEL_PATH = "model.tflite"

interpreter = None
input_details = None
output_details = None

try:
    # Try using tflite_runtime first (lighter), fallback to tf (dev)
    try:
        import tflite_runtime.interpreter as tflite
        print("Using tflite_runtime")
    except ImportError:
        import tensorflow.lite as tflite
        print("Using tensorflow.lite")

    if os.path.exists(MODEL_PATH):
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Loaded TFLite model from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")

except Exception as e:
    print("❌ Could not load TFLite model:", e)

CLASS_JSON = "class_indices.json"
if os.path.exists(CLASS_JSON):
    with open(CLASS_JSON, "r") as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
else:
    class_names = {}

# ========== Helpers ==========
def pretty_label(raw_label: str) -> str:
    return raw_label.replace("_", " ").replace("  ", " ").title()

def create_alert_message(label: str, certainty: int):
    label_lower = label.lower()
    
    if "healthy" in label_lower:
        sms = f"Your plant looks healthy! Keep monitoring regularly. Certainty: {certainty}%."
        email = (
            f"Hello Farmer,\n\n"
            f"🌱 Your plant is healthy! ✅\n\n"
            f"Checked at: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
            f"Keep monitoring your crops regularly."
        )
    else:
        recs = {
            "pepper bell bacterial spot": "Remove infected leaves, use copper-based bactericide.",
            "potato early blight": "Apply fungicide and remove infected leaves.",
            "potato late blight": "Use fungicide promptly and ensure good drainage.",
            "tomato bacterial spot": "Remove infected fruits/leaves, apply copper spray.",
            "tomato early blight": "Apply fungicide, remove affected leaves, rotate crops.",
            "tomato late blight": "Apply fungicide, avoid overhead irrigation, remove infected plants.",
            "tomato leaf mold": "Improve air circulation, remove affected leaves, use fungicide.",
            "tomato septoria leaf spot": "Remove infected leaves, apply fungicide.",
            "tomato spider mites two spotted spider mite": "Spray miticide and maintain humidity.",
            "tomato target spot": "Remove infected areas and use appropriate fungicide.",
            "tomato yellowleaf curl virus": "Control whitefly vector and remove infected plants.",
            "tomato mosaic virus": "Remove infected plants and disinfect tools.",
            "powdery mildew": "Spray neem oil or suitable fungicide. Avoid wetting leaves.",
            "rust": "Remove infected leaves and apply fungicide.",
            "blight": "Ensure proper spacing and apply copper-based fungicide."
        }
        rec = recs.get(label_lower, "Consult your local agriculture expert for guidance.")
        sms = f"Crop Alert: {label} detected. Certainty: {certainty}%. Recommended action: {rec}"
        email = (
            f"Hello Farmer,\n\n"
            f"🌱 Plant Disease Alert ⚠️\n\n"
            f"Disease Detected: {label}\n"
            f"Certainty: {certainty}%\n\n"
            f"Recommended Action:\n{rec}\n\n"
            f"Checked at: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
            f"Stay safe and monitor your crops regularly."
        )
    return sms, email

def predict_image(img_path):
    if interpreter is None:
        return "model-not-loaded", 0
    
    # Preprocess image
    img = kimage.load_img(img_path, target_size=(128, 128))
    arr = kimage.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    idx = int(np.argmax(preds))
    raw_label = class_names.get(idx, f"class_{idx}")
    label = pretty_label(raw_label)
    certainty = int(round(100 * preds[idx]))
    return label, certainty

# ========== EMAIL & SMS ==========
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

def send_email(to_email, subject, body):
    if not EMAIL_USER or not EMAIL_PASS:
        print("Email credentials not set; skipping email.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        # Connect to server
        print(f"📧 Connecting to SMTP...")
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10) # 10s timeout
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, to_email, msg.as_string())
        server.quit()
        print("✅ Email sent to", to_email)
        return True
    except Exception as e:
        print(f"❌ Email failed (Non-critical): {e}")
        return False

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

def send_sms(to_phone, message):
    if not (TWILIO_SID and TWILIO_AUTH and TWILIO_NUMBER):
        print("Twilio credentials missing; skipping SMS.")
        return False
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        msg = client.messages.create(body=message, from_=TWILIO_NUMBER, to=to_phone)
        print("✅ SMS sent, SID:", getattr(msg, "sid", None))
        return True
    except Exception as e:
        print("❌ SMS error:", e)
        return False

# ========== ROUTES ==========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email").lower().strip()
        phone = request.form.get("phone")
        password = request.form.get("password")

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))

        hashed = generate_password_hash(password)
        user = User(name=name, email=email, phone=phone, password=hashed)
        db.session.add(user)
        db.session.commit()
        flash("Account created. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email").lower().strip()
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["user_name"] = user.name
            session["user_email"] = user.email
            session["user_phone"] = user.phone
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

# ========== DASHBOARD ==========
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        flash("Please log in to access dashboard.", "warning")
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("No file uploaded.", "danger")
            return redirect(url_for("dashboard"))

        filename = secure_filename(file.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"

        # ---------- SUPABASE UPLOAD ----------
        image_url = None
        if supabase:
            try:
                file_bytes = file.read()
                supabase.storage.from_("plant-images").upload(unique, file_bytes)
                url_data = supabase.storage.from_("plant-images").get_public_url(unique)
                image_url = url_data.get("public_url")
                print("✅ Uploaded to Supabase:", image_url)
            except Exception as e:
                print("❌ Supabase upload error:", e)

        # ---------- LOCAL FALLBACK ----------
        if not image_url:
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
            file.seek(0)  # reset file pointer
            file.save(save_path)
            # store relative static path for web display
            image_url = f"/static/uploads/{unique}"

        # ---------- PREDICTION ----------
        # If Supabase gave public URL, download image for model prediction
        if image_url.startswith("http"):
            temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"tmp_{unique}")
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            label, certainty = predict_image(temp_path)
            os.remove(temp_path)
        else:
            label, certainty = predict_image(os.path.join("static", "uploads", unique))

        # Save prediction to DB
        p = Prediction(user_id=user.id, image_path=image_url, disease=label, certainty=certainty)
        db.session.add(p)
        db.session.commit()

        # Sync to Supabase table
        if supabase:
            try:
                supabase.table("predictions").insert({
                    "user_email": user.email,
                    "image_path": image_url,
                    "disease": label,
                    "certainty": certainty,
                    "created_at": datetime.now(IST).isoformat()
                }).execute()
                print("✅ Prediction synced to Supabase")
            except Exception as e:
                print("❌ Supabase table insert error:", e)

        sms_text, email_text = create_alert_message(label, certainty)

        # Send notifications only for diseased plants
        if "healthy" not in label.lower():
            if user.email:
                send_email(user.email, "Crop Disease Alert", email_text)
            if user.phone:
                send_sms(user.phone, sms_text)

        return render_template(
            "result.html",
            img_path=image_url,
            label=label,
            certainty=certainty,
            recommendation=email_text,
            user=user
        )

    # ========== FILTERS & STATS ==========
    crop_filter = request.args.get('crop', '').lower()
    disease_filter = request.args.get('disease', '').lower()
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')

    history_query = Prediction.query.filter_by(user_id=user.id)

    if crop_filter:
        history_query = history_query.filter(Prediction.disease.ilike(f'%{crop_filter}%'))
    if disease_filter:
        history_query = history_query.filter(Prediction.disease.ilike(f'%{disease_filter}%'))
    if date_from:
        history_query = history_query.filter(Prediction.created_at >= datetime.fromisoformat(date_from))
    if date_to:
        history_query = history_query.filter(Prediction.created_at <= datetime.fromisoformat(date_to))

    history = history_query.order_by(Prediction.created_at.desc()).limit(20).all()

    healthy_count = Prediction.query.filter_by(user_id=user.id).filter(Prediction.disease.ilike('%healthy%')).count()
    diseased_count = Prediction.query.filter_by(user_id=user.id).filter(~Prediction.disease.ilike('%healthy%')).count()
    recent_alerts = Prediction.query.filter_by(user_id=user.id).filter(~Prediction.disease.ilike('%healthy%')).order_by(Prediction.created_at.desc()).limit(5).all()

    return render_template(
        "dashboard.html",
        user=user,
        history=history,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        recent_alerts=recent_alerts
    )


@app.route("/crops")
def crops():
    return render_template("crops.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ========== RUN ==========
if __name__ == "__main__":
    import os
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

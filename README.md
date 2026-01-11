# 🌱 CropCare – Automated Crop Disease Detection System

CropCare is a web-based **Automated Crop Disease Detection System with Real-Time Alerts** that uses **AI-based image recognition** to identify plant diseases from leaf images.  
The system provides **instant predictions**, **treatment recommendations**, and **real-time alerts via Email & SMS**, helping farmers take timely action to protect their crops.

---

## 🚀 Features

- 🌿 **AI-Powered Disease Detection**
  - Upload plant leaf images
  - Detect diseases using a trained deep learning model
  - Get prediction confidence percentage

- ⚡ **Real-Time Alerts**
  - Automatic **Email & SMS notifications**
  - Alerts sent only when disease is detected

- 📊 **Dashboard & History**
  - View past predictions
  - Track healthy vs diseased crops
  - Recent alerts and activity logs

- ☁️ **Cloud Integration**
  - Predictions synced to **Supabase**
  - Secure and scalable cloud storage

- 🧑‍🌾 **User-Friendly Interface**
  - Clean, modern UI
  - Mobile-responsive design
  - Simple upload and result flow

---

## 🧠 Tech Stack

### Frontend
- HTML5
- CSS3 (Bootstrap 5)
- Jinja2 Templates

### Backend
- Python
- Flask
- SQLAlchemy (SQLite)

### AI / ML
- TensorFlow / Keras
- Convolutional Neural Network (CNN)
- Image preprocessing & classification

### Cloud & Services
- Supabase (cloud database)
- Gmail SMTP (Email alerts)
- Twilio (SMS alerts)

---

## 

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Arshwinsajeevan/CropCare.git
cd cropcare

2️⃣ Create & Activate Virtual Environment
python -m venv venv
Windows:
venv\Scripts\activate
Linux / macOS:
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt
🔐 Environment Variables (.env)
Create a .env file in the root directory:


env
SECRET_KEY=your_secret_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_email_app_password
TWILIO_SID=your_twilio_sid
TWILIO_AUTH=your_twilio_auth_token
TWILIO_NUMBER=your_twilio_phone_number
MODEL_PATH=model.h5

▶️ Running the Application
python app.py
Then open your browser and go to:
http://127.0.0.1:5000


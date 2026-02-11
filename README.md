# 🌱 CropCare – Automated Crop Disease Detection System

CropCare is a web-based **Automated Crop Disease Detection System with Real-Time Alerts** that uses **AI-based image recognition** to identify plant diseases from leaf images.  

### 🔗 Live Demo: [https://cropcare-7v18.onrender.com](https://cropcare-7v18.onrender.com)

The system provides **instant predictions**, **treatment recommendations**, and **real-time alerts via Email & SMS**, helping farmers take timely action to protect their crops.

---

## 🚀 Features

- 🌿 **AI-Powered Disease Detection**
  - Uses an optimized **TensorFlow Lite (`.tflite`)** model for fast, lightweight prediction.
  - Detects diseases in Pepper Bell, Potato, and Tomato plants.
  - Get prediction confidence percentage.

- ⚡ **Real-Time Alerts**
  - Automatic **Email & SMS notifications** via Gmail SMTP and Twilio.
  - Alerts sent only when disease is detected.

- 📊 **Dashboard & History**
  - View past predictions.
  - Track healthy vs diseased crops.
  - Recent alerts and activity logs.

- ☁️ **Cloud Integration**
  - Predictions synced to **Supabase**.
  - Secure and scalable cloud storage.

- 🧑‍🌾 **User-Friendly Interface**
  - Clean, modern UI (Bootstrap 5).
  - Mobile-responsive design.

---

## 🧠 Tech Stack

### Frontend
- HTML5, CSS3 (Bootstrap 5)
- Jinja2 Templates

### Backend
- Python (Flask)
- SQLAlchemy (SQLite)

### AI / ML
- **TensorFlow Lite**: Lightweight inference for low-memory environments (like Render).
- **CNN Model**: Custom trained Convolutional Neural Network.

### Cloud & Services
- **Supabase** (Database)
- **Gmail SMTP** (Email alerts)
- **Twilio** (SMS alerts)
- **Render** (Deployment)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Arshwinsajeevan/CropCare.git
cd CropCare
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables (.env)
Create a `.env` file in the root directory:

```env
SECRET_KEY=your_secret_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_email_app_password
TWILIO_SID=your_twilio_sid
TWILIO_AUTH=your_twilio_auth_token
TWILIO_NUMBER=your_twilio_phone_number
MODEL_PATH=model.tflite  
```

### 5️⃣ Run the Application
```bash
python app.py
```
Open your browser and go to: `http://127.0.0.1:5000`

---

## 🚀 Deployment (Render)

This project is optimized for **Render Free Tier**:
1.  **Repo Structure**: Contains `Procfile` and `runtime.txt`.
2.  **Model**: Uses `model.tflite` (28MB) instead of huge `.h5` files.
3.  **Deploy**:
    - Connect repo to Render.
    - Add Environment Variables.
    - Set `MODEL_PATH` to `model.tflite`.
    
See `DEPLOY.md` for full instructions.

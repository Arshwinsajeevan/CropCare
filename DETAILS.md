# 🌿 CropCare: Automated Crop Disease Detection System

## 🌟 Project Overview
CropCare is a web-based platform designed to assist farmers by identifying crop diseases using AI. It leverages deep learning (CNN/TFLite) to analyze leaf images and provides real-time alerts via Email and SMS, along with treatment recommendations.

## ⚙️ How It Works (Optimized for Render)
1.  **User Registration**: Farmers sign up securely.
2.  **Image Upload**: Users upload an image of a crop leaf through the dashboard.
3.  **AI Analysis (Fast & Light)**:
    -   We use the **TensorFlow Lite Interpreter** for prediction.
    -   This allows the app to run on **Render's Free Tier** (512MB RAM) without crashing.
    -   The model file (`model.tflite`) is only ~28MB (down from ~90MB).
4.  **Instant Feedback**: The system displays the diagnosis and confidence level.
5.  **Smart Alerts (Robust)**: If a disease is detected, an automated email and SMS are sent. (We handle connection errors gracefully).
6.  **Data Sync**: Predictions are stored locally (SQLite) and synced to the cloud (Supabase) for record-keeping.
7.  **History & Analytics**: Users can view past diagnoses and track disease trends.

## 🛠️ Technology Stack
### Frontend
-   **HTML5 & CSS3**: Structured and styled using Bootstrap 5 for responsiveness.
-   **Jinja2**: Templating engine for dynamic content rendering.

### Backend
-   **Python (Flask)**: Core logic handling routes, authentication, and API integration.
-   **SQLAlchemy (SQLite)**: Local database for user and prediction management.

### Machine Learning (AI)
-   **TensorFlow Lite (`tflite-runtime`)**: Memory-efficient inference for production.
-   **Original Model**: Keras CNN (`.h5`) -> Converted to TFLite.
-   **Image Preprocessing**: Resizing (128x128) and normalization.

### Cloud & Third-Party Services
-   **Supabase**: Cloud database for secure and scalable storage of prediction logs.
-   **Twilio API**: Sends instant SMS alerts.
-   **Gmail SMTP**: Delivers detailed email reports (with error handling).

## 📊 Supported Crops & Diseases
The system is currently trained to detect:

### 🌶️ Pepper Bell
-   Bacterial Spot
-   Healthy

### 🥔 Potato
-   Early Blight
-   Late Blight
-   Healthy

### 🍅 Tomato
-   Bacterial Spot
-   Early Blight
-   Late Blight
-   Leaf Mold
-   Septoria Leaf Spot
-   Spider Mites (Two-Spotted Spider Mite)
-   Target Spot
-   Yellow Leaf Curl Virus
-   Mosaic Virus
-   Healthy

## 🚀 Key Features
-   **Render Optimized**: Fully functional on free cloud hosting.
-   **High Accuracy**: Utilizing a deep learning model trained on a comprehensive dataset.
-   **Dual Notifications**: Ensures critical alerts are not missed.
-   **Cloud Backup**: Securely stores data in Supabase.
-   **User-Centric Design**: Simple, intuitive interface for ease of use.

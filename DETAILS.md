# 🌿 CropCare: Automated Crop Disease Detection System

## 🌟 Project Overview
CropCare is a web-based platform designed to assist farmers by identifying crop diseases using AI. It leverages deep learning (CNN) to analyze leaf images and provides real-time alerts via Email and SMS, along with treatment recommendations.

## ⚙️ How It Works
1.  **User Registration**: Farmers sign up securely.
2.  **Image Upload**: Users upload an image of a crop leaf through the dashboard.
3.  **AI Analysis**: A trained TensorFlow/Keras model processes the image to detect diseases.
4.  **Instant Feedback**: The system displays the diagnosis and confidence level.
5.  **Smart Alerts**: If a disease is detected, an automated email and SMS are sent to the farmer with recommended actions.
6.  **Data Sync**: Predictions are stored locally (SQLite) and synced to the cloud (Supabase) for record-keeping.
7.  **History & Analytics**: Users can view past diagnoses and track disease trends on their dashboard.

## 🛠️ Technology Stack
### Frontend
-   **HTML5 & CSS3**: Structured and styled using Bootstrap 5 for responsiveness.
-   **Jinja2**: Templating engine for dynamic content rendering.

### Backend
-   **Python (Flask)**: Core logic handling routes, authentication, and API integration.
-   **SQLAlchemy (SQLite)**: Local database for user and prediction management.

### Machine Learning (AI)
-   **TensorFlow & Keras**: Used to build and train the Convolutional Neural Network (CNN).
-   **Image Preprocessing**: Resizing and normalization of leaf images.

### Cloud & Third-Party Services
-   **Supabase**: Cloud database for secure and scalable storage of prediction logs.
-   **Twilio API**: Sends instant SMS alerts to users.
-   **Gmail SMTP**: Delivers detailed email reports.

## 📊 Supported Crops & Diseases
The system is currently trained to detect the following conditions:

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
-   **High Accuracy**: Utilizing a deep learning model trained on a comprehensive dataset.
-   **Dual Notifications**: Ensures critical alerts are not missed.
-   **Cloud Backup**: Securely stores data in Supabase.
-   **User-Centric Design**: Simple, intuitive interface for ease of use.

BrainTumorUI is a Streamlit-based web application for brain tumor detection and classification using deep learning models. Users can upload MRI brain scan images and receive AI-powered predictions with confidence scores through an interactive web interface.

Features

MRI image upload support
Brain tumor detection using AI models
Multi-class tumor classification
Confidence score predictions
Streamlit web interface
TensorFlow and PyTorch integration
Real-time image preprocessing and prediction pipeline

Technologies Used

Python
Streamlit
TensorFlow / Keras
PyTorch
OpenCV
NumPy
Pandas
Matplotlib

Project Structure

BrainTumorUI/
│
├── app.py
├── pipeline.py
├── utils.py
├── requirements.txt
├── runtime.txt
│
├── models/
│   ├── best_model_full.pth
│   └── multiclass_tumor_model.h5

Installation

Clone the repository:

git clone https://github.com/CameronAskins/BrainTumorUI.git
cd BrainTumorUI

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py



Deployment

This project is deployed using Streamlit Community Cloud.

How It Works
User uploads an MRI image
Image preprocessing is performed
AI model analyzes the scan
Prediction and confidence scores are displayed
Tumor classification results are returned to the user
Future Improvements
Grad-CAM heatmap visualization
Improved UI/UX design
User authentication
Prediction history storage
Mobile responsive interface
Additional tumor classification categories

Authors

Cameron Askins, Axel Espinosa-Chan, Williams Okoye, Tirth Patel

License

This project is for educational and research purposes.

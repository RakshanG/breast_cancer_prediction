Breast Cancer Prediction App

This project is all about using deep learning to help with a real-world problem: predicting whether a breast tumor is Benign or Malignant.

I trained a neural network on the well-known Breast Cancer Wisconsin dataset (available in scikit-learn) and then built a simple Streamlit app so anyone can try it out by entering a few values.
What it does

You enter 5 key medical measurements about a tumor:

1.Mean Radius
2.Mean Texture
3.Mean Smoothness
4.Mean Concavity
5.Mean Symmetry

The app then predicts:
1.Benign (safe)
2.Malignant (needs attention)

It also shows a confidence score so you know how sure the model is.

How I built it

1.Used Python + TensorFlow/Keras to train a neural network
2.Preprocessed the data with StandardScaler
3.Trained with just 5 selected features (instead of all 30) and still got 95% accuracy
4.Deployed the model with Streamlit so it’s interactive and easy to use.

Try it yourself

Clone the repo and install the requirements:

git clone https://github.com/RakshanG/breast_cancer_prediction.git
cd breast_cancer_prediction
pip install -r requirements.txt


Run the app locally:
streamlit run app.py

Open http://localhost:8501
 in your browser to use it.

Example
If you enter:
Mean Radius = 13.54
Mean Texture = 14.36
Mean Smoothness = 0.09779
Mean Concavity = 0.08129
Mean Symmetry = 0.173
The model predicts:
Benign (Confidence: ~98%) 

breast_cancer_prediction/
│── app.py                   
│── breast_cancer_model_5.h5 
│── scaler_5.pkl             
│── requirements.txt         
│── README.md     

I built this to strengthen my deep learning fundamentals and showcase how to go from data → model → app.

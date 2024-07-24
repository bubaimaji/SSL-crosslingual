import pandas as pd
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import soundfile as sf
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import numpy as np
import librosa
# Assume "preprocess" is a module you've defined for preprocessing, and "df" is a DataFrame obtained from there
from preprocess import df
import re

# Assuming df is defined in preprocess.py and contains 'path' and 'emotion' columns

# Load the pre-trained WavLM model and processor
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
original_model = WavLMModel.from_pretrained("microsoft/wavlm-base")

# Move the original model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
original_model.to(device)

def clone_model(model):
    """Clone the WavLM model's configuration and state."""
    model_clone = WavLMModel(model.config).to(device)
    model_clone.load_state_dict(model.state_dict())
    return model_clone

def extract_features(model, audio_paths, target_sampling_rate=16000, batch_size=8):
    """Extract features from audio files using the model."""
    batch_features = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        speeches = []

        for audio_path in batch:
            speech, _ = librosa.load(audio_path, sr=target_sampling_rate)
            if len(speech) == 0:
                raise ValueError(f"Audio file {audio_path} is silent or empty.")
            speeches.append(speech)

        input_values = processor(speeches, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
        with torch.no_grad():
            hidden_states = model(input_values).last_hidden_state
            batch_features.extend([state.mean(dim=0).cpu().numpy() for state in hidden_states])

    return batch_features

audio_files = df['path'].tolist()  # Ensure this matches your DataFrame column names
labels = df['emotion'].tolist()

for N in range(13):  # 0 to 12
    # Clone the model for each N
    model = clone_model(original_model)
    model.encoder.layers = torch.nn.ModuleList(model.encoder.layers[:N])

    try:
        features = extract_features(model, audio_files)
    except ValueError as e:
        print(e)
        continue

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    classifiers = {
        "SVC": SVC()
    }

    print(f"\n--- Results for {N} Transformer Layers ---")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"\n{name} Classifier Report:\n")
        print(classification_report(y_test, y_pred, digits=4))

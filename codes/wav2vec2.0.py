import torch
import numpy as np
import librosa
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from preprocess import df

# Load the pre-trained Wav2Vec 2.0 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
full_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


def extract_features(model, audio_paths, target_sampling_rate=16000, batch_size=8):
    batch_features = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        speeches = []

        for audio_path in batch:
            speech, _ = librosa.load(audio_path, sr=target_sampling_rate)
            if len(speech) == 0:
                raise ValueError(f"Audio file {audio_path} is silent or empty.")
            speeches.append(speech)

        input_values = processor(speeches, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True).input_values
        with torch.no_grad():
            hidden_states = model(input_values).last_hidden_state
            batch_features.extend([np.mean(state.numpy(), axis=0) for state in hidden_states])

    return batch_features

audio_files = df['path'].tolist()  # Replace with your column name for file paths
labels = df['emotion'].tolist()

for N in range(13):  # 0 to 12
    # Create a deep copy of the full model for each N
    model = deepcopy(full_model)
    model.encoder.layers = torch.nn.ModuleList(model.encoder.layers[:N])

    try:
        features = extract_features(model, audio_files)
    except ValueError as e:
        print(e)
        continue

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    classifiers = {
        "SVC": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    print(f"\n--- Results for {N} Transformer Layers ---")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"\n{name} Classifier Report:\n")
        print(classification_report(y_test, y_pred, digits=4))

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# Assuming df is your DataFrame with 'path' and 'emotion' columns
#from preprocess import df
df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/all_segments.csv')
class DownstreamModel(nn.Module):
    def __init__(self, input_size, hidden_size, pooling_method='mean'):
        super(DownstreamModel, self).__init__()
        self.pooling_method = pooling_method
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.batch_norm(x)
        # Pooling is already applied during feature extraction, so it's not needed here
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Initialize the feature extractor and original model
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
original_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
original_model.to(device)

def clone_model(model):
    model_clone = WavLMModel(model.config).to(device)
    model_clone.load_state_dict(model.state_dict())
    return model_clone

def extract_features(model, audio_paths, target_sampling_rate=16000, batch_size=8):
    batch_features = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        speeches = []
        for audio_path in batch:
            speech, _ = librosa.load(audio_path, sr=target_sampling_rate)
            if len(speech) == 0:
                # Skip or handle empty audio files
                continue
            speeches.append(speech)
        input_values = processor(speeches, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values).last_hidden_state
            # Mean pooling applied here as an example; adjust based on your pooling preference
            batch_features.extend(outputs.mean(dim=1).cpu().numpy())
    return batch_features

def process_features_with_downstream(model, features):
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float).to(device)
        processed_features = model(features_tensor).cpu().numpy()
    return processed_features

audio_files = df['path'].tolist()
labels = df['label'].tolist()

for N in range(25):  # Simplified for demonstration; replace with range(13) for full variation
    model = clone_model(original_model)
    model.encoder.layers = torch.nn.ModuleList(model.encoder.layers[:N])

    for pooling_method in ['mean', 'max']:  # Iterate over your pooling preferences here
        downstream_model = DownstreamModel(input_size=1024, hidden_size=256, pooling_method=pooling_method).to(device)

        try:
            features = extract_features(model, audio_files)
            processed_features = process_features_with_downstream(downstream_model, features)
        except ValueError as e:
            print(e)
            continue

        X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.5, random_state=42)

        classifiers = {"SVC": SVC()}
        print(f"\n--- Results for {N} Transformer Layers with {pooling_method} Pooling ---")
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(f"\n{name} Classifier Report:\n")
            print(classification_report(y_test, y_pred, digits=4, zero_division=0))

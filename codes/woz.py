import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, HubertModel

class DownstreamModel(nn.Module):
    def __init__(self, input_size, hidden_size, pooling_method='mean'):
        super(DownstreamModel, self).__init__()
        self.pooling_method = pooling_method
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.batch_norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Load training and validation data from separate CSV files
train_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/train_segments.csv')

# Load validation data
validation_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/validation_segments.csv')


# Assuming both DataFrames have 'path' and 'emotion' columns
train_audio_files = train_df['path'].tolist()
train_labels = train_df['label'].tolist()

validation_audio_files = validation_df['path'].tolist()
validation_labels = validation_df['label'].tolist()

# Initialize the feature extractor and original model
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
original_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
original_model.to(device)

def clone_model(model):
    model_clone = HubertModel(model.config).to(device)
    model_clone.load_state_dict(model.state_dict())
    return model_clone

def extract_features(model, audio_paths, target_sampling_rate=16000, batch_size=64):
    batch_features = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        speeches = []
        for audio_path in batch:
            speech, _ = librosa.load(audio_path, sr=target_sampling_rate)
            if len(speech) == 0:
                continue
            speeches.append(speech)
        input_values = processor(speeches, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values).last_hidden_state
            batch_features.extend(outputs.mean(dim=1).cpu().numpy())
    return batch_features

def process_features_with_downstream(model, features):
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float).to(device)
        processed_features = model(features_tensor).cpu().numpy()
    return processed_features

# Example of processing and classification
for N in range(13):  # Simplified for demonstration
    model = clone_model(original_model)
    model.encoder.layers = torch.nn.ModuleList(model.encoder.layers[:N])

    for pooling_method in ['mean', 'max']:
        downstream_model = DownstreamModel(input_size=1024, hidden_size=512, pooling_method=pooling_method).to(device)

        try:
            # Extract and process features for training set
            train_features = extract_features(model, train_audio_files)
            train_processed_features = process_features_with_downstream(downstream_model, train_features)

            # Extract and process features for validation set
            validation_features = extract_features(model, validation_audio_files)
            validation_processed_features = process_features_with_downstream(downstream_model, validation_features)
        except ValueError as e:
            print(e)
            continue

        classifiers = {"SVC": SVC()}
        print(f"\n--- Results for {N} Transformer Layers with {pooling_method} Pooling ---")
        for name, clf in classifiers.items():
            clf.fit(train_processed_features, train_labels)
            y_pred = clf.predict(validation_processed_features)
            print(f"\n{name} Classifier Report:\n")
            print(classification_report(validation_labels, y_pred, digits=4))

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import librosa
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F
from sklearn.svm import SVC
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.model_selection import KFold

class DownstreamModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DownstreamModel, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.batch_norm(x)
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

def extract_features(model, processor, audio_paths, target_sampling_rate=16000, batch_size=64, device='cpu', pooling_method='mean'):
    batch_features = []
    model.to(device)  # Ensure the model is on the correct device
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        speeches = []
        for audio_path in batch:
            speech, _ = librosa.load(audio_path, sr=target_sampling_rate)
            if len(speech) == 0:
                continue
            speeches.append(speech)
        input_values = processor(speeches, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True).input_values.to(device)  # Ensure tensors are on the correct device
        with torch.no_grad():
            outputs = model(input_values).last_hidden_state
            if pooling_method == 'mean':
                pooled_outputs = outputs.mean(dim=1)
            else:  # If 'max' pooling
                pooled_outputs = outputs.max(dim=1).values
            batch_features.extend(pooled_outputs.cpu().numpy())
    return batch_features

def process_features_with_downstream(downstream_model, features):
    features_tensor = torch.tensor(features, dtype=torch.float).to(device)
    with torch.no_grad():
        processed_features = downstream_model(features_tensor).cpu().detach().numpy()
    return processed_features

# Load datasets
train_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/train_segments.csv')
validation_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/validation_segments.csv')

# Prepare data
train_audio_files = train_df['path'].tolist()
train_labels = train_df['label'].tolist()
validation_audio_files = validation_df['path'].tolist()
validation_labels = validation_df['label'].tolist()

# 5-Fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for N in range(1, 25):  # Adjust based on your needs
    model = clone_model(original_model)
    
    for pooling_method in ['mean', 'max']:
        downstream_model = DownstreamModel(input_size=1024, hidden_size=512).to(device)  # Corrected line
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_audio_files)):
            cv_train_files = [train_audio_files[i] for i in train_idx]
            cv_train_labels = [train_labels[i] for i in train_idx]
            cv_val_files = [train_audio_files[i] for i in val_idx]
            cv_val_labels = [train_labels[i] for i in val_idx]

            train_features = extract_features(model, processor, cv_train_files, target_sampling_rate=16000, batch_size=64, device=device, pooling_method=pooling_method)
            train_processed = process_features_with_downstream(downstream_model, train_features)
            
            val_features = extract_features(model, processor, cv_val_files, target_sampling_rate=16000, batch_size=64, device=device, pooling_method=pooling_method)
            val_processed = process_features_with_downstream(downstream_model, val_features)
            
            classifier = SVC()
            classifier.fit(train_processed, cv_train_labels)
            y_pred = classifier.predict(val_processed)

            report = classification_report(cv_val_labels, y_pred, digits=4, zero_division=0)
            print(f"Fold {fold+1}, {N} Layers, {pooling_method} Pooling:\n{report}")
            cv_results.append(report)

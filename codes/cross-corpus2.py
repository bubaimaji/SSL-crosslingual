import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# Assuming df is already loaded
train_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/train_segments.csv')
validation_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/validation_segments.csv')


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

def clone_model(model, N, device):
    model_clone = WavLMModel(model.config).to(device)
    model_clone.load_state_dict(model.state_dict())
    # Adjust the encoder layers based on the desired configuration
    if N is not None:
        model_clone.encoder.layers = torch.nn.ModuleList(model_clone.encoder.layers[:N])
    return model_clone


def extract_features(model, processor, audio_paths, target_sampling_rate=16000, batch_size=64, device='cpu'):
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

def process_features_with_downstream(downstream_model, features, device='cpu'):
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float).to(device)
        processed_features = downstream_model(features_tensor).cpu().numpy()
    return processed_features

processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Perform 5-Fold Cross-Validation on Training Data
         

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for N in range(1, 25):  # Adjust based on your needs
    for pooling_method in ['mean', 'max']:
        fold_results = []

        for fold, (train_index, test_index) in enumerate(kf.split(train_df), start=1):
            print(f"Fold {fold}, Testing with {N} Transformer Layers and {pooling_method} pooling")

            # Correctly split data for the current fold
            train_audio_files = train_df.iloc[train_index]['path'].tolist()
            test_audio_files = train_df.iloc[test_index]['path'].tolist()
            train_labels = train_df.iloc[train_index]['label'].tolist()
            test_labels = train_df.iloc[test_index]['label'].tolist()

            # Initialize and prepare models
            original_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
            model = clone_model(original_model, N, device)
            downstream_model = DownstreamModel(input_size=1024, hidden_size=512, pooling_method=pooling_method).to(device)

            # Feature extraction and processing
           # Feature extraction and processing
            train_features = extract_features(model, processor, train_audio_files, device=device)
            test_features = extract_features(model, processor, test_audio_files, device=device)
            train_processed_features = process_features_with_downstream(downstream_model, train_features, device)
            test_processed_features = process_features_with_downstream(downstream_model, test_features, device)

            # Train and evaluate classifier
            classifier = SVC()
            classifier.fit(train_processed_features, train_labels)
            y_pred = classifier.predict(test_processed_features)
            report = classification_report(test_labels, y_pred, digits=4, zero_division=0)
            print(report)
            fold_results.append(report)

        # Optionally, aggregate and print fold results for this configuration

        
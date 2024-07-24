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

# Load and combine training and validation data
train_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/train_segments.csv')
validation_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/validation_segments.csv')
combined_df = pd.concat([train_df, validation_df]).sample(frac=1).reset_index(drop=True)

audio_files = combined_df['path'].tolist()
labels = combined_df['label'].tolist()

# Initialize the feature extractor
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Original model loading moved inside the loop to allow modification per fold

# Perform 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for N in range(1, 4):  # Example: Test with first 1, 2, and 3 layers. Adjust as needed.
    for pooling_method in ['mean', 'max']:  # Iterate over desired pooling methods
        print(f"Testing with {N} Transformer Layers and {pooling_method} pooling")

        for fold, (train_index, test_index) in enumerate(kf.split(audio_files), start=1):
            print(f"Fold {fold}")

            # Split data for the current fold
            train_audio_files, test_audio_files = [audio_files[i] for i in train_index], [audio_files[i] for i in test_index]
            train_labels, test_labels = [labels[i] for i in train_index], [labels[i] for i in test_index]

            # Initialize models for the current configuration
            original_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
            model = clone_model(original_model, N, device)
            downstream_model = DownstreamModel(input_size=1024, hidden_size=512, pooling_method=pooling_method).to(device)

            # Extract and process features for the current fold
            train_features = extract_features(model, processor, train_audio_files, device=device)
            test_features = extract_features(model, processor, test_audio_files, device=device)
            
            train_processed_features = process_features_with_downstream(downstream_model, train_features, device)
            test_processed_features = process_features_with_downstream(downstream_model, test_features, device)

            # Train classifier
            classifier = SVC()
            classifier.fit(train_processed_features, train_labels)
            y_pred = classifier.predict(test_processed_features)

            # Evaluate and print results
            report = classification_report(test_labels, y_pred, digits=4)
            print(f"Results for {N} layers with {pooling_method} pooling in Fold {fold}:\n{report}")
            results.append((N, pooling_method, fold, report))

# Results contain performance metrics for each configuration and fold.

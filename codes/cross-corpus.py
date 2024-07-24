import pandas as pd
import os
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Your Data Preprocessing
folder_path = '/home/suhita/Documents/SSL-deep/Bng/New folder/'  # Update this path accordingly
#folder_path = '/home/suhita/Documents/SSL-deep/Bng/bn-depress/'  
audio_files = os.listdir(folder_path)
emotions = []
file_paths = []
for f in audio_files:
    part = f.split('.')[0].split('-')
    emotions.append(int(part[1]))  # Adjust indexing if necessary
    file_paths.append(os.path.join(folder_path, f))

audio_df = pd.DataFrame({'label': emotions, 'path': file_paths})
audio_df['label'] = audio_df['label'].replace({1: 1, 2: 0})

# Split your dataset
train_df, test_df = train_test_split(audio_df, test_size=0.2, random_state=42)
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

# Load and prepare external datasets
external_train_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/train_segments.csv')
external_test_df = pd.read_csv('/home/suhita/Documents/SSL-deep/Speech-model/output/validation_segments.csv')
external_train_df['label'] = external_train_df['label'].astype(int)
external_test_df['label'] = external_test_df['label'].astype(int)

# Initialize model and processor
processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
original_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
original_model.to(device)

def clone_model(model, num_layers=None):
    """Clone the WavLM model's configuration and state, keeping only a specified number of transformer layers."""
    model_clone = WavLMModel(model.config).to(device)
    model_clone.load_state_dict(model.state_dict())
    if num_layers is not None and num_layers > 0:
        model_clone.encoder.layers = torch.nn.ModuleList(model_clone.encoder.layers[:num_layers])
    return model_clone

def extract_features(model, audio_paths, target_sampling_rate=16000, batch_size=4, pooling_method='mean'):
    """Extract features from audio files using the model, with specified pooling method."""
    batch_features = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        speeches = []
        for audio_path in batch:
            speech, _ = librosa.load(audio_path, sr=target_sampling_rate)
            if len(speech) == 0:
                continue  # Skip empty files
            speeches.append(speech)
        input_values = processor(speeches, return_tensors="pt", padding=True, sampling_rate=target_sampling_rate).input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state
            
            if pooling_method == 'mean':
                pooled_features = hidden_states.mean(dim=1)
            elif pooling_method == 'max':
                pooled_features, _ = hidden_states.max(dim=1)
            else:
                raise ValueError(f"Unsupported pooling method: {pooling_method}")
            
            batch_features.extend(pooled_features.cpu().numpy())
    return batch_features

def process_data(model, df, pooling_method='mean'):
    audio_files = df['path'].tolist()
    labels = df['label'].tolist()
    return extract_features(model, audio_files, pooling_method=pooling_method), labels

def run_cross_corpus_experiment(train_features, train_labels, test_features, test_labels, experiment_name=""):
    classifiers = {
        "SVC": SVC()
    }
    print(f"\n--- Running Experiment: {experiment_name} ---")
    for name, clf in classifiers.items():
        clf.fit(train_features, train_labels)
        y_pred = clf.predict(test_features)
        print(f"\n{name} Classifier Report:\n")
        print(classification_report(test_labels, y_pred, digits=4, labels=np.unique(test_labels).tolist(), zero_division=0))

# Varying transformer layers and pooling methods in experiments
for N in range(1, 25):  # Iterate through 1 to 12 layers
    for pooling_method in ['mean', 'max']:  # Add more pooling methods as needed
        cloned_model = clone_model(original_model, N)
        
        # Prepare and process your data with varying layers
        my_train_features, my_train_labels = process_data(cloned_model, train_df, pooling_method=pooling_method)
        my_test_features, my_test_labels = process_data(cloned_model, test_df, pooling_method=pooling_method)
         # Prepare and process external data
        external_train_features, external_train_labels = process_data(cloned_model, external_train_df,pooling_method=pooling_method)
        external_test_features, external_test_labels = process_data(cloned_model, external_test_df,pooling_method=pooling_method)
        
        # Cross-corpus experiments
        print(f"\nTraining on My Data, Testing on External Test Data with {N} layers and {pooling_method} pooling:")
        run_cross_corpus_experiment(my_train_features, my_train_labels, external_test_features, external_test_labels, f"My Data -> External Test Data with {N} layers and {pooling_method} pooling")
        
        print(f"\nTraining on External Train Data, Testing on My Data with {N} layers and {pooling_method} pooling:")
        run_cross_corpus_experiment(external_train_features, external_train_labels, my_test_features, my_test_labels, f"External Train Data -> My Test Data with {N} layers and {pooling_method} pooling")

        print("Experiment within My Data (Train vs. Test):")
        run_cross_corpus_experiment(my_train_features, my_train_labels, my_test_features, my_test_labels, f"My Train Data and Test Data with {N} layers and {pooling_method} pooling")
        
        print("Experiment within External Data (Train vs. Test):")
        run_cross_corpus_experiment(external_train_features, external_train_labels, external_test_features, external_test_labels, f"External Train Data and Test Data with {N} layers and {pooling_method} pooling")

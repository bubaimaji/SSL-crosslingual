import pandas as pd
from tqdm import tqdm
import os
from pydub import AudioSegment

def load_mapping(file_path):
    df = pd.read_csv(file_path)
    return df.set_index('Participant_ID')[['PHQ8_Binary', 'PHQ8_Score']].to_dict('index')

# Paths to the train, validation, and test CSV files
train_csv_path = "/home/suhita/Documents/SSL-deep/daic/label/train_split_Depression_AVEC2017.csv"
validation_csv_path = "/home/suhita/Documents/SSL-deep/daic/label/dev_split_Depression_AVEC2017.csv"
test_csv_path = "/home/suhita/Documents/SSL-deep/daic/label/full_test_split - Copy.csv"

# Load mappings
train_mapping = load_mapping(train_csv_path)
test_mapping = load_mapping(test_csv_path)
validation_mapping = load_mapping(validation_csv_path)

audio_directory = '/home/suhita/Documents/SSL-deep/daic/audio/' 
output_directory = '/home/suhita/Documents/SSL-deep/Speech-model/output/'
segment_duration = 4000  # 4 seconds in milliseconds
overlap = 0.5  # 50% overlap

# Define directories for each set
train_output_directory = os.path.join(output_directory, 'train')
validation_output_directory = os.path.join(output_directory, 'validation')
test_output_directory = os.path.join(output_directory, 'test')

# Create these directories if they don't exist
for directory in [train_output_directory, validation_output_directory, test_output_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def process_audio(file_id, category):
    audio_path = os.path.join(audio_directory, f'{file_id}_final.wav')
    audio = AudioSegment.from_file(audio_path)
    segments_info = []

    for i in range(0, len(audio) - segment_duration + 1, int(segment_duration * (1 - overlap))):
        segment = audio[i:i + segment_duration]
        #segment_file = os.path.join(output_directory, f'{file_id}_segment_{i // 1000}.wav')
        segment_file = f'{file_id}_segment_{i // 1000}.wav'
        segment_path = os.path.join(output_directory, category, segment_file)
        segment.export(segment_path, format='wav')

        label = train_mapping[file_id]['PHQ8_Binary'] if category == 'train' else \
                test_mapping[file_id]['PHQ8_Binary'] if category == 'test' else \
                validation_mapping[file_id]['PHQ8_Binary'] 

        score = train_mapping[file_id]['PHQ8_Score'] if category == 'train' else \
                test_mapping[file_id]['PHQ8_Score'] if category == 'test' else \
                validation_mapping[file_id]['PHQ8_Score'] 
        
        segments_info.append([segment_path, file_id, label, score])

    return segments_info

# Process each audio file and collect segment info
train_segments = []
test_segments = []
validation_segments = []

for file_id in range(300, 493):
    if file_id in train_mapping:
        train_segments += process_audio(file_id, 'train')
    elif file_id in test_mapping:
        test_segments += process_audio(file_id, 'test')
    elif file_id in validation_mapping:
        validation_segments += process_audio(file_id, 'validation')


pd.DataFrame(train_segments, columns=['SegmentFileName', 'FileID', 'Label','Score']).to_csv(os.path.join(output_directory, 'train_segments.csv'), index=False)
pd.DataFrame(test_segments, columns=['SegmentFileName', 'FileID', 'Label','Score']).to_csv(os.path.join(output_directory, 'test_segments.csv'), index=False)
pd.DataFrame(validation_segments, columns=['SegmentFileName', 'FileID', 'Label','Score']).to_csv(os.path.join(output_directory, 'validation_segments.csv'), index=False)


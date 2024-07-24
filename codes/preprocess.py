# Example: Listing contents of a specific folder in your Drive
import pandas as pd
import os

folder_path = '/home/suhita/Documents/SSL-deep/Bng/New folder/'  # Update this path
audio = os.listdir(folder_path)

# CREATE FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
emotion = []

file_path = []
#filename = os.listdir(audio + i) #iterate over Actor folders
for f in audio: # go through files in Actor folder
    part = f.split('.')[0].split('-')
    emotion.append(int(part[1]))

    file_path.append(folder_path  + '/' + f)

# PUT EXTRACTED LABELS WITH FILEPATH INTO DATAFRAME
audio_df = pd.DataFrame(emotion)
audio_df = audio_df.replace({1:'0', 2:'1'})
#audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
audio_df.columns = ['label']
audio_df = pd.concat([pd.DataFrame(file_path, columns = ['path']), audio_df],axis=1)
#audio_df.head()

label2id = {'depressed': 0, 'normal': 1}
id2label = {0: 'depressed', 1: 'normal'}

# Replace 'label' column with integer values
audio_df['label'] = audio_df['label'].replace({'0': 0, '1': 1})
# Create 'emotion' column based on 'labels' column
audio_df['emotion'] = audio_df['label'].map(id2label)
df=audio_df
#df.head()
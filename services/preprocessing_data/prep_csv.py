import os
import pandas as pd
import argparse
import random


def main(input_csv, output_dir):
    """
    This script prepares a metadata_train.csv and metadata_eval.csv and metadata_test.csv file for Whisper fine-tuning.
    Args:
        input_csv (str): Path to the input CSV file containing the audio file paths and their corresponding text.
        output_dir (str): Directory where the output metadata_train.csv and metadata_eval.csv files will be saved.
    """

    # Read the input CSV or TSV file
    if input_csv.endswith('.tsv'):
        df = pd.read_csv(input_csv, sep='\t', header=None)
        # TSV have a different column structure, so we need to set the column names accordingly
        # This is the header: iso_code	user_ID	source_text	target_text	filename	recording_date	recording_duration	comment	comment_date
        df.columns = ['iso_code', 'user_ID', 'source_text', 'target_text', 'filename', 'recording_date', 'recording_duration', 'comment', 'comment_date']

        # We need to extract the relevant columns into the same csv format as the original script
        processed_data = []
        for index, row in df.iterrows():
            lang = row['iso_code'].strip()
            if lang == 'efi':
                lang = 'efik'
            elif lang == 'ibo':
                lang = 'igbo'
            elif lang == 'xho':
                lang = 'xhosa'
            # Use the correct base path for your audio files
            audio_file = f"/home/vacl2/multimodal_translation/services/data/languages/{lang}/wav_audio/{str(row['filename']).strip()}"
            # Replace the .webm extension with .wav
            audio_file = audio_file.replace('.webm', '.wav')
            text = row['target_text'].strip()
            # Do a little additional cleaning by removing all \""
            text = text.replace('"', '').replace("'", "")
            speaker_name = str(row['user_ID']).strip() if 'user_ID' in row else "unknown"
            processed_data.append([audio_file, text, speaker_name])
    else:
        # Assuming the input CSV is pipe-separated as per the original context
        # If it's not, you can change the separator accordingly
        df = pd.read_csv(input_csv, sep='|', header=None)
        df.columns = ['audio_file', 'text', 'normalized_text', 'speaker_name', 'lang']

        # The current CSV has too much information, we only need "audio_file|text|speaker_name",
        # currently we have "audio_file|Text|Normalized Text|speaker_name|lang"
        # Extract the relevant columns
        processed_data = []
        for index, row in df.iterrows():
            audio_file = row['audio_file'].strip()
            # Replace the .webm extension with .wav
            audio_file = audio_file.replace('.webm', '.wav')
            text = row['text'].strip()
            speaker_name = str(row['speaker_name']).strip() if 'speaker_name' in row else "unknown"
            processed_data.append([audio_file, text, speaker_name])

    # Shuffle the rows to ensure randomness
    random.shuffle(processed_data)

    # Separate data into training and evaluation lists
    train_data = []
    eval_data = []
    test_data = []
    for i, row in enumerate(processed_data):
        # Use a 8:1:1 split for train, eval, and test (80% train, 10% eval, 10% test)
        if i % 10 == 8:
            eval_data.append(row)
        elif i % 10 == 9:
            test_data.append(row)
        else: # Catches 8 out of 10 cases (0-7)
            train_data.append(row)

    # Define the column headers
    columns = ['audio_file', 'text', 'speaker_name']

    # Create DataFrames from your lists
    train_df = pd.DataFrame(train_data, columns=columns)
    eval_df = pd.DataFrame(eval_data, columns=columns)
    test_df = pd.DataFrame(test_data, columns=columns)

    # Write the DataFrames to CSV files with '|' as the delimiter
    train_df.to_csv(
        os.path.join(output_dir, "metadata_train.csv"),
        sep='|',          # Use '|' as the separator/delimiter
        index=False,      # Do not write the DataFrame index
        encoding='utf-8'
    )

    eval_df.to_csv(
        os.path.join(output_dir, "metadata_eval.csv"),
        sep='|',          # Use '|' as the separator/delimiter
        index=False,      # Do not write the DataFrame index
        encoding='utf-8'
    )

    test_df.to_csv(
        os.path.join(output_dir, "metadata_test.csv"),
        sep='|',          # Use '|' as the separator/delimiter
        index=False,      # Do not write the DataFrame index
        encoding='utf-8'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare metadata CSV files for Whisper fine-tuning.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV files.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args.input_csv, args.output_dir)  # Call the main function to execute the script logic
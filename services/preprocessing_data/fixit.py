import re
import os



def remove_broken_filepaths(input_tsv_path, output_tsv_path, broken_path_regex):
    """
    Removes lines from a TSV file that contain a broken filepath pattern.

    Args:
        input_tsv_path (str): Path to the input TSV file.
        output_tsv_path (str): Path to the output TSV file where cleaned data will be saved.
        broken_path_regex (str): Regular expression pattern to identify broken filepaths.
    """
    removed_count = 0
    total_lines = 0
    with open(input_tsv_path, 'r', encoding='utf-8') as infile, \
         open(output_tsv_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            total_lines += 1
            if re.search(broken_path_regex, line):
                removed_count += 1
            else:
                outfile.write(line)
    print(f"Cleaned TSV file saved to: {output_tsv_path}")
    print(f"Total lines processed: {total_lines}")
    print(f"Lines with broken filepaths removed: {removed_count}")

if __name__ == "__main__":
    # Define the input and output TSV file paths
    input_tsv = "/grphome/grp_mtlab/projects/project-speech/african_tts/data/metadata.csv"
    output_tsv = "/grphome/grp_mtlab/projects/project-speech/african_tts/data/metadata_cleaned.csv"

    # Define the regex pattern for the broken filepath
    # The | needs to be escaped if it's meant to be a literal character in the path
    broken_filepath_pattern = r"/grphome/grp_mtlab/projects/project-speech/african_tts/data/wavs/\|"

    # Run the cleaning function
    remove_broken_filepaths(input_tsv, output_tsv, broken_filepath_pattern)
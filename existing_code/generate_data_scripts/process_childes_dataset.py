import argparse
import os
import re

from datasets import load_dataset
from huggingface_hub import login


def save_dataset_to_txt(dataset, output_file):
    """
    Save the CHILDES dataset to a text file, filtering out speaker tags and
    annotations, and condensing the result into a single paragraph.
    
    Args:
        dataset: Hugging Face dataset object
        output_file (str): Path to the output text file
    """
    all_utterances = []
    # Iterate through each split in the dataset (train, validation, test)
    for split in dataset:
        # Get all examples in the current split
        for example in dataset[split]:
            # Get the utterance line, preferring 'utterance' over 'text'
            line = example.get('utterance') or example.get('text')
            if line:
                # 1. Remove speaker tag (e.g., *MOT:, *FAT:) using regex
                cleaned_line = re.sub(r'^\*[A-Z]{3}:\s*', '', line)
                # 2. Remove content in square brackets (e.g., [+ I], [= crying])
                cleaned_line = re.sub(r'\[.*?\]', '', cleaned_line)
                # 3. Strip leading/trailing whitespace and add if not empty
                cleaned_line = cleaned_line.strip()
                if cleaned_line:
                    all_utterances.append(cleaned_line)
    # Join all cleaned utterances into a single paragraph separated by spaces
    condensed_paragraph = ' '.join(all_utterances)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(condensed_paragraph)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download CHILDES dataset from Hugging Face')
    parser.add_argument('--token', type=str, help='Hugging Face access token')
    args = parser.parse_args()
    
    # Load the CHILDES dataset
    print("Loading CHILDES dataset...")
    try:
        # Try to load the dataset without authentication first
        try:
            dataset = load_dataset("wonderwind271/CHILDES-raw")
            print("Dataset loaded successfully without authentication!")
        except Exception as e:
            if not args.token:
                print("Authentication required. Please provide your Hugging Face access token.")
                print("Get your token from: https://huggingface.co/settings/tokens")
                print("\nRun the script like this:")
                print("python process_childes_dataset.py --token YOUR_ACCESS_TOKEN")
                return
                
            # Login to Hugging Face with provided token
            login(token=args.token)
            
            # Try loading the dataset with authentication
            dataset = load_dataset("wonderwind271/CHILDES-raw")
            # See the columns in the train split
            print(dataset["train"].features)
            print("Dataset loaded successfully with authentication!")

        # Pattern that starts with *xxx where x is 3 letters except for CHI
        pattern = re.compile(r'^\*(?!CHI)[A-Z]{3}')
        # could write as a lambda function
        def is_valid(example):
            text = example.get('text')
            return bool(pattern.match(text))
        
        filtered_dataset = dataset.filter(is_valid)
        print(filtered_dataset)

        # Define output file path inside the 'data' directory
        project_root = os.path.dirname(__file__)
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_file = os.path.join(data_dir, 'childes_filtered.txt')
        
        # Save the dataset to a text file
        print(f"Saving dataset to {output_file}...")
        save_dataset_to_txt(filtered_dataset, output_file)
        print(f"Dataset saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

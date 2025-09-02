from datasets import load_dataset
from huggingface_hub import login
import os
import argparse

def save_dataset_to_txt(dataset, output_file):
    """
    Save the CHILDES dataset to a text file.
    
    Args:
        dataset: Hugging Face dataset object
        output_file (str): Path to the output text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Iterate through each split in the dataset (train, validation, test)
        for split in dataset:
            # Get all examples in the current split
            for example in dataset[split]:
                # Write each utterance to the file
                if 'utterance' in example:
                    f.write(example['utterance'] + '\n')
                # Some datasets might use 'text' instead of 'utterance'
                elif 'text' in example:
                    f.write(example['text'] + '\n')

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
            print("Dataset loaded successfully with authentication!")
        
        # Define output file path
        output_file = os.path.join(os.path.dirname(__file__), 'childes_data.txt')
        
        # Save the dataset to a text file
        print(f"Saving dataset to {output_file}...")
        save_dataset_to_txt(dataset, output_file)
        print(f"Dataset saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

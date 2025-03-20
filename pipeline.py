import logging
import os
import random
import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import tqdm
from sentence_perturb_create_ds import WordReplacer 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
languages = {
    "en": {"name": "english", "verb": "verb", "adj": "adjective"},
    "es": {"name": "spanish", "verb": "verbo", "adj": "adjetivo"},
    "de": {"name": "german", "verb": "Verb", "adj": "Adjektiv"},
    "fr": {"name": "french", "verb": "verbe", "adj": "adjectif"},
    "ja": {"name": "japanese", "verb": "動詞 (dōshi)", "adj": "形容詞 (keiyōshi)"},
    "ko": {"name": "korean", "verb": "동사 (dongsa)", "adj": "형용사 (hyeongyongsa)"},
    "zh": {"name": "chinese", "verb": "动词 (dòngcí)", "adj": "形容词 (xíngróngcí)"},
}

def run_pipeline(dataset_name: str, lang: str) -> None:
  
    # # Set seeds for reproducibility
    # np.random.seed(42)
    # random.seed(42)
    # torch.manual_seed(42)
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # logger.info(f"Using device: {device}")
    GPT_Word_Replacer = WordReplacer(language=languages[lang])
    # Load the dataset
    logger.info(f"Loading dataset '{dataset_name}' for language '{lang}'")
    dataset = load_dataset(dataset_name, lang)
    logger.info(f"Dataset loaded: {dataset}")
    
    train_dataset = dataset["train"]
    logger.info(f"Training split extracted with {len(train_dataset)} samples")
    
    # Get the text column (adjust according to your dataset structure)
    columns_to_keep = ["id", "sentence1", "label"]
    # Create a list of dictionaries, each containing only the selected columns
    selected_data = []
    for i in range(len(train_dataset)):
        sample = {col: train_dataset[i][col] for col in columns_to_keep}
        selected_data.append(sample)
    # Create batches

    batch_size = 32  
    batches = [selected_data[i:i+batch_size] for i in range(0, len(selected_data), batch_size)]
    logger.info(f"Created {len(batches)} batches of size {batch_size}")
    
    # Process each batch with the API
    all_responses = []
    for i, batch in enumerate(tqdm.tqdm(batches, desc="Processing batches")):
        logger.info(f"Processing batch {i+1}/{len(batches)}")
        # make a list from the batch take sentence1
        batch_sentece = [item["sentence1"] for item in batch]
        # make a copy of batch sentences for modification
        batch_responses_synonyms = process_batch_with_openai(batch_sentece, "synonyms", GPT_Word_Replacer) 
        batch_responses_antonyms = process_batch_with_openai(batch_sentece, "antonyms", GPT_Word_Replacer) 
        logger.info(f"Batch {i+1} processed for synonyms with {len(batch_responses_synonyms)} responses")
        logger.info(f"Batch {i+1} processed for antonyms with {len(batch_responses_antonyms)} responses")
        # Add id and label to the responses
        processed_items = []
        for j in range(len(batch)):
            processed_item = {
                "id": batch[j]["id"],
                "sentence1": batch[j]["sentence1"],
                 "perturbed_synonyms": batch_responses_synonyms[j],
                "perturbed_antonyms": batch_responses_antonyms[j],
                "label": batch[j]["label"],
            }
            processed_items.append(processed_item)
        
        all_responses.extend(processed_items)
        break # for debug
        
        # Optional: Save responses periodically to avoid losing progress
        # if (i+1) % 10 == 0:
        #     logger.info(f"Saving responses after batch {i+1}")
        #     # Add your saving logic here (e.g., to a file or database)
    
    logger.info(f"Processing complete. Processed {len(all_responses)} samples.")
    # Save all responses to a file
    output_file = f"{dataset_name}_perturbed_{lang}.csv"
    save_results(all_responses, output_file)
    
    # Here you can add code to save or further process the responses
    
    return all_responses

def process_batch_with_openai(sentences: List[str], perturbation_type, word_replacer) -> List[str]:
    """
    Processes a batch of data with the OpenAI API.
    
    Args:
        batch: A list of dictionaries, each containing the data to be processed.
        
    Returns:
        A list of responses from the OpenAI API.
    """
    
    perturbed_sentences = word_replacer.sentence_replacement(sentences=sentences, n=1, types=perturbation_type)
     
    return perturbed_sentences

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Saves the processed results to a file.
    
    Args:
        results: A list of dictionaries containing the processed data.
        output_file: The path to the output file.
    """
    df = pd.DataFrame(results)
    # Ensure the directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, output_file)
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    
    logger.info(f"Results saved to {output_file}")
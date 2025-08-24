import pickle
import GPUtil
from datasets import load_dataset
import torch
import pathlib
import numpy as np

# Create data batches
def batchify(lst, batch_size):
    """Yield successive batches from list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def load_datasets():
    # Load datasets
    malicious_text = []
    
    ds = load_dataset("declare-lab/HarmfulQA")
    malicious_text += ds['train']['question']

    ds = load_dataset("LLM-LAT/harmful-dataset")
    malicious_text += ds['train']['prompt']

    # Load JailBreakV-28k dataset
    jailbreakv_28k_ds = load_dataset("JailbreakV-28K/JailBreakV-28k", 'JailBreakV_28K')["JailBreakV_28K"]
    jailbreakv_28k_ds = jailbreakv_28k_ds.filter(lambda ex: ex["format"] == "Template")
    malicious_text += jailbreakv_28k_ds['redteam_query']
    
    ds = load_dataset("sentence-transformers/natural-questions")
    begin_text = ds['train']['query'][:len(malicious_text)]
    
    print(f'Number of benign promts: {len(begin_text)}')
    print(f'Number of malicious promts: {len(malicious_text)}')
    
    return begin_text + malicious_text, np.array([0]*len(begin_text) + [1]*len(malicious_text), dtype=np.int8)


def save_question_and_lable(prompts, labels, dir):
    """
    Save prompts and labels to a file in a specified directory.
    The file will be named 'data.pkl' and will contain a dictionary with keys 'prompts' and 'labels'.
    """
    data = {
        'prompts': prompts,
        'labels': labels
    }
    with open(dir, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_question_and_lable(dir):
    """
    Load prompts and labels from a file in a specified directory.
    The file should be named 'data.pkl' and should contain a dictionary with keys 'prompts' and 'labels'.
    """
    with open(dir, 'rb') as fp:
        data = pickle.load(fp)
    return data['prompts'], data['labels']

def shuffle(inputs, labels):
    # Generate a random permutation of indices based on the number of samples
    indices = torch.randperm(inputs.size(0))

    # Shuffle data and labels using the generated indices
    shuffled_data = inputs[indices]
    shuffled_labels = labels[indices]
    return shuffled_data, shuffled_labels


def save_dict(data, dir):
    with open(dir, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_dict(dir):
    with open(dir, 'rb') as fp:
        data = pickle.load(fp)
        # data = pickle.load(fp, fix_imports=True, encoding='bytes', errors='strict')
    return data

def create_dir(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True) 

def get_response_lengths(responses):
    """
    Given a list of responses, return the lengths of each response.
    """
    response_lengths = [len(response) for response in responses]
    return response_lengths

def get_free_gpu():
    # Get a list of all GPUs and their status
    gpus = GPUtil.getGPUs()
    # If there are no GPUs available, raise an error
    if not gpus:
        raise RuntimeError("No GPU available.")
    # Sort GPUs by available memory (descending order)
    gpus_sorted_by_memory = sorted(gpus, key=lambda gpu: gpu.memoryFree, reverse=True)
    # Select the GPU with the most free memory
    selected_gpu = gpus_sorted_by_memory[0]
    print(f"Selected GPU ID: {selected_gpu.id}")
    return selected_gpu.id
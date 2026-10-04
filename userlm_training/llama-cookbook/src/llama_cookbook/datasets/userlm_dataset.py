import os

from datasets import load_dataset, Features, Sequence, Value

MAX_LEN = 4096

_features = Features({
    "input_ids": Sequence(Value("int64")),
    "labels": Sequence(Value("int64")),
})

def get_preprocessed_flipped_dialogue(dataset_config, tokenizer, split, override_datapath, model_name_string):
    """
    Factory function compatible with llama-cookbook's dataset loading
    This function signature matches the pattern used in llama-cookbook

    Args:
        dataset_config: Configuration with data_path
        tokenizer: HuggingFace tokenizer
        split: 'train', 'validation', or 'test'

    Returns:
        FlippedDialogueDataset instance
    """
    # Map llama-cookbook split names to our split names
    split_mapping = {
        'train': 'train',
        'validation': 'val',
        'test': 'test'
    }
    partition = split_mapping.get(split, split)
    # Each distributed rank preprocesses after loading the model. Avoid forking
    # one worker per host CPU, which can exhaust host memory on large machines.
    num_proc = int(os.environ.get("USERLM_DATASET_NUM_PROC", "1"))
    if num_proc < 1:
        raise ValueError("USERLM_DATASET_NUM_PROC must be at least 1")
    pool_size = num_proc if num_proc > 1 else None
    print(f"Dataset preprocessing processes per rank: {num_proc}")
    load_path = os.path.join(
        override_datapath,
        f"{partition}_{model_name_string}_samples.jsonl"
    )
    ds = load_dataset("json",data_files=load_path,features=_features,split="train")
    print(f"Loaded {partition} dataset with {len(ds)} samples from {load_path}")
    ds = ds.filter(
        lambda ex: len(ex["input_ids"]) <= MAX_LEN,
        num_proc=pool_size
    )
    print(f"Filtered to {len(ds)} samples with max length {MAX_LEN}")

    def _add_attn_mask(ex):
        n = len(ex["input_ids"])
        ex["attention_mask"] = [1] * n
        return ex
    
    ds = ds.map(_add_attn_mask, num_proc=pool_size)
    return ds

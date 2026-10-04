"""Supervised fine-tuning of a user model (App. D) with the reduced llama-cookbook FSDP trainer.

Run from the repository root in the training environment (requirements-training.txt), e.g. on 2 GPUs:

    SRC=wildchat_1m; BASE=Qwen/Qwen2.5-7B-Instruct; TAG=Qwen--Qwen2.5-7B-Instruct
    torchrun --standalone --nproc_per_node 2 userlm_training/train.py \
        --model_name $BASE --tokenizer_name artifacts/models/tokenizers/$TAG \
        --dataset userlm_dataset --dataset_path artifacts/userlm_data/$SRC/sft \
        --enable_fsdp True --fsdp_config.pure_bf16 True --fsdp_config.fsdp_activation_checkpointing True \
        --use_fast_kernels True --batching_strategy padding \
        --batch_size_training 4 --val_batch_size 4 --gradient_accumulation_steps 8 \
        --num_epochs 1 --lr 2e-5 --weight_decay 0.0 --warmup_ratio 0.1 --min_lr 2e-6 \
        --run_validation True --run_validation_before_train True --eval_steps 100 \
        --dist_checkpoint_root_folder artifacts/models/userlm_${SRC}_${TAG} \
        --output_dir artifacts/models/userlm_${SRC}_${TAG}

Samples longer than 4,096 tokens are excluded. The embedding of <|endconversation|> is initialized from
the pretrained EOS embedding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "llama-cookbook" / "src"))
import fire
from llama_cookbook.finetuning import main

if __name__ == "__main__":
    fire.Fire(main)

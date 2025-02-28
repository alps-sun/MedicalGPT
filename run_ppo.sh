CUDA_VISIBLE_DEVICES=0,1 python ppo_training.py \
    --sft_model_path Qwen/Qwen2.5-0.5B-Instruct \
    --reward_model_path Qwen/Qwen2.5-0.5B-Instruct \
    --template_name qwen \
    --torch_dtype bfloat16 \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --max_source_length 1024 \
    --response_length 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --do_train \
    --total_episodes 30000 \
    --output_dir outputs-ppo-qwen-v1 \
    --missing_eos_penalty 1.0 \
    --eval_strategy steps \
    --eval_steps 100 \
    --num_train_epochs 3 \
    --report_to tensorboard
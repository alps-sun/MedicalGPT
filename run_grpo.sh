CUDA_VISIBLE_DEVICES=0 python grpo_training.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name openai/gsm8k \
    --per_device_train_batch_size 1 \
    --max_steps 500 \
    --save_steps 50 \
    --save_strategy steps \
    --max_prompt_length 256 \
    --max_completion_length 512 \
    --num_generations 2 \
    --output_dir outputs-grpo-qwen-v1 \
    --torch_dtype float16 \
    --fp16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03
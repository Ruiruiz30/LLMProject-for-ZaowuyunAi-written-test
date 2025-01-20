@echo off
set CUDA_VISIBLE_DEVICES=0

call python  dpo.py ^
    --model_type Qwen ^
    --model_name_or_path D:\LLMProject\Qwen2.5-0.5B ^
    --template_name qwen ^
    --train_file_dir D:\LLMProject\dpo_dataset ^
    --validation_file_dir D:\LLMProject\dpo_dataset ^
    --per_device_train_batch_size 4 ^
    --per_device_eval_batch_size 4 ^
    --do_train ^
    --do_eval ^
    --use_peft True ^
    --max_train_samples 1000 ^
    --max_eval_samples 10 ^
    --max_steps 100 ^
    --eval_steps 20 ^
    --save_steps 50 ^
    --max_source_length 1024 ^
    --max_target_length 512 ^
    --output_dir dpo_qwen ^
    --target_modules all ^
    --lora_rank 8 ^
    --lora_alpha 16 ^
    --lora_dropout 0.05 ^
    --torch_dtype float16 ^
    --fp16 True ^
    --device_map auto ^
    --report_to tensorboard ^
    --remove_unused_columns False ^
    --gradient_checkpointing True ^
    --cache_dir ./cache


pause

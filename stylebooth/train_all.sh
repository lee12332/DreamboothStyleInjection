#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

BASE_CONTENT_DIR="/root/autodl-tmp/output"
BASE_STYLE_DIR="/root/autodl-tmp/B"
OUTPUT_DIR="../result"
COMMAND="python run_styleid_diffusers.py \
                --cnt_fn \"$BASE_CONTENT_DIR\" \
                --sty_fn \"$BASE_STYLE_DIR\" \
                --save_dir \"$OUTPUT_DIR\" \
                --gamma 0.7"
            echo "运行命令："$COMMAND
            eval $COMMAND


export MODEL=./Mistral-7B-Instruct-v0.2 # Path to the model
export DEVICES=0        # GPU device ID

# you can change the following parameters as needed
CUDA_VISIBLE_DEVICES=$DEVICES python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size 1 \
    --seed 42 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.99 \
    --swap-space 16 \
    --disable-log-requests \
    --dtype bfloat16 \
    --host 0.0.0.0 --port 12345     # make sure to expose the host IP address and port number!
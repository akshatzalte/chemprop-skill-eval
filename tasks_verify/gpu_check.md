This is a check of this environment, not a chemistry task. Run this exact command once with the Bash tool and reply with its full output verbatim:

`echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; ls /dev/nvidia* 2>&1 | head; nvidia-smi --query-gpu=index,name,memory.used --format=csv 2>&1 | head -4; python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"`

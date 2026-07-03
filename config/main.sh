export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python hoi_multi_agent_system.py \
        --test_json ./data/hico_20160224_det/annotations/test_hico.json \
        --image_dir ./data/hico_20160224_det/images/test2015 \
        --output ./result.json

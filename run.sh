python3 train.py    --data Hotel \
                    --device cuda \
                    --gpus 0 \
                    --batch_size 2 \
                    --num_epochs 10 \
                    --learning_rate 2e-3 \
                    --num_workers 4 \
                    --accumulation_step 100 \
                    --experiment_path outputs
#!/bin/bash
# Example training script for DPO on BioMPNN

echo "=========================================="
echo "Example DPO Training"
echo "=========================================="

# Check if base models are set up
if [ ! -f "data/input/BioMPNN/soluble_model_weights/v_48_030.pt" ]; then
    echo "Error: Soluble model weights not found!"
    echo "Please run: ./scripts/setup_base_models.sh"
    exit 1
fi

if [ ! -f "data/input/BioMPNN/base_hparams/v_48_020.yaml" ]; then
    echo "Error: Base model config not found!"
    echo "Please run: ./scripts/setup_base_models.sh"
    exit 1
fi

# Check if training data exists
if [ ! -f "data/input/test_data_real.json" ]; then
    echo "Warning: No training data found at data/training_data.json"
    echo "Using example data instead..."
    
    if [ ! -f "examples/example_training_data.json" ]; then
        echo "Error: Example training data not found!"
        exit 1
    fi
    
    TRAIN_DATA="examples/example_training_data.json"
else
    TRAIN_DATA="data/input/test_data_real.json"
fi

echo ""
echo "Training Configuration:"
echo "  Training data: $TRAIN_DATA"
echo "  Base model: v_48_020"
echo "  Output: outputs/example_dpo_model"
echo "  Epochs: 10"
echo "  Learning rate: 3e-7"
echo "  Beta: 0.1"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting training..."
echo ""

python -m src.dpo_inv.run_dpo \
    --train_data "$TRAIN_DATA" \
    --output_dir outputs/dpo_training \
    --base_model_name v_48_020 \
    --base_model_dir data/input/BioMPNN/soluble_model_weights \
    --base_model_config_dir data/input/BioMPNN/base_hparams \
    --num_epochs 10 \
    --batch_size 1 \
    --lr 3e-6 \
    --weight_decay 0.0 \
    --beta 0.1 \
    --temperature 0.1 \
    --max_length 500 #\
    #--save_every 2

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: outputs/example_dpo_model/"
echo ""
echo "To run inference:"
echo "  python -m src.dpo_inv.inference \\"
echo "    --model_path outputs/example_dpo_model/best_model.pt \\"
echo "    --pdb_file your_protein.pdb \\"
echo "    --output_dir outputs/inference"
echo ""


#!/bin/bash
# Setup script for downloading ProteinMPNN soluble model weights

set -e

echo "=========================================="
echo "BioMPNN Soluble Model Setup"
echo "=========================================="

# Create directories
echo "Creating directories..."
mkdir -p data/input/BioMPNN/soluble_model_weights
mkdir -p data/input/BioMPNN/base_hparams
mkdir -p data/pdbs
mkdir -p outputs

echo "Directories created successfully!"

# Check if models already exist
if [ -f "data/input/BioMPNN/soluble_model_weights/v_48_020.pt" ]; then
    echo ""
    echo "Soluble model weights already exist!"
    echo "Location: data/input/BioMPNN/soluble_model_weights/"
    echo ""
    read -p "Do you want to re-download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

echo ""
echo "=========================================="
echo "Downloading ProteinMPNN Soluble Models"
echo "=========================================="
echo ""
echo "Soluble models are optimized for soluble proteins."
echo "Source: https://github.com/dauparas/ProteinMPNN"
echo ""
echo "Options:"
echo "1. Download directly using wget/curl (recommended)"
echo "2. Clone ProteinMPNN repository and copy weights"
echo "3. Use existing weights from another location"
echo ""
read -p "Choose option (1/2/3): " option

case $option in
    1)
        echo ""
        echo "Downloading soluble model weights..."

        # Base URL for ProteinMPNN models
        BASE_URL="https://files.ipd.uw.edu/pub/training_sets/soluble_model_weights"

        # Common soluble model files
        MODELS=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt" "v_48_030.pt")

        cd data/input/BioMPNN/soluble_model_weights

        for model in "${MODELS[@]}"; do
            echo "Downloading $model..."
            if command -v wget &> /dev/null; then
                wget -q --show-progress "${BASE_URL}/${model}" -O "${model}" || echo "Failed to download $model (may not exist)"
            elif command -v curl &> /dev/null; then
                curl -L -o "${model}" "${BASE_URL}/${model}" || echo "Failed to download $model (may not exist)"
            else
                echo "Error: Neither wget nor curl is available. Please install one of them."
                cd - > /dev/null
                exit 1
            fi
        done

        cd - > /dev/null
        echo "Download complete!"
        ;;
    2)
        echo ""
        echo "Cloning ProteinMPNN repository..."
        if [ ! -d "temp_proteinmpnn" ]; then
            git clone https://github.com/dauparas/ProteinMPNN.git temp_proteinmpnn
        fi

        echo "Copying soluble model weights..."
        if [ -d "temp_proteinmpnn/training/model_weights" ]; then
            # Try to find soluble weights in the repo
            find temp_proteinmpnn -name "*.pt" -path "*/soluble*" -exec cp {} data/input/BioMPNN/soluble_model_weights/ \; 2>/dev/null || \
            echo "Note: Copying all available .pt files..."
            find temp_proteinmpnn -name "*.pt" -exec cp {} data/input/BioMPNN/soluble_model_weights/ \; 2>/dev/null || \
            echo "Warning: No .pt files found in repository"
        fi

        echo "Cleaning up..."
        # Optionally remove the cloned repo
        read -p "Remove temporary ProteinMPNN clone? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf temp_proteinmpnn
        fi
        ;;
    3)
        echo ""
        read -p "Enter path to existing soluble model weights directory: " weights_path
        if [ -d "$weights_path" ]; then
            echo "Creating symbolic link..."
            ln -sf "$(realpath $weights_path)" data/input/BioMPNN/soluble_model_weights
            echo "Linked successfully!"
        else
            echo "Error: Directory not found: $weights_path"
            exit 1
        fi
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Creating Model Configuration Files"
echo "=========================================="

# Create default config for v_48_020
cat > data/input/BioMPNN/base_hparams/v_48_020.yaml << 'EOF'
num_letters: 21
node_features: 128
edge_features: 128
hidden_dim: 128
num_encoder_layers: 3
num_decoder_layers: 3
vocab: 21
k_neighbors: 48
dropout: 0.1
EOF

echo "Created: data/input/BioMPNN/base_hparams/v_48_020.yaml"

# Create configs for other common models
for model in v_48_002 v_48_010 v_48_030; do
    if [ -f "data/input/BioMPNN/vanilla_model_weights/${model}.pt" ]; then
        # Extract k_neighbors from model name (48 in this case)
        k_neighbors=$(echo $model | grep -oP 'v_\K\d+')
        
        cat > data/input/BioMPNN/base_hparams/${model}.yaml << EOF
num_letters: 21
node_features: 128
edge_features: 128
hidden_dim: 128
num_encoder_layers: 3
num_decoder_layers: 3
vocab: 21
k_neighbors: ${k_neighbors}
dropout: 0.1
EOF
        echo "Created: data/input/BioMPNN/base_hparams/${model}.yaml"
    fi
done

echo ""
echo "=========================================="
echo "Verifying Setup"
echo "=========================================="

# Check for model files
model_count=$(ls data/input/BioMPNN/soluble_model_weights/*.pt 2>/dev/null | wc -l)
config_count=$(ls data/input/BioMPNN/base_hparams/*.yaml 2>/dev/null | wc -l)

echo "Soluble model weights found: $model_count"
echo "Config files found: $config_count"

if [ $model_count -gt 0 ] && [ $config_count -gt 0 ]; then
    echo ""
    echo "✓ Setup completed successfully!"
    echo ""
    echo "Available soluble models:"
    ls data/input/BioMPNN/soluble_model_weights/*.pt 2>/dev/null | xargs -n 1 basename
    echo ""
    echo "Next steps:"
    echo "1. Prepare your training data (see examples/example_training_data.json)"
    echo "2. Run training with soluble models:"
    echo "   python -m src.dpo_inv.run_dpo \\"
    echo "     --train_data <your_data.json> \\"
    echo "     --base_model_dir data/input/BioMPNN/soluble_model_weights \\"
    echo "     --base_model_name v_48_020"
    echo ""
    echo "For inference:"
    echo "   python -m src.dpo_inv.inference \\"
    echo "     --model_path outputs/model/best_model.pt \\"
    echo "     --pdb_file your_protein.pdb \\"
    echo "     --base_model_dir data/input/BioMPNN/soluble_model_weights"
else
    echo ""
    echo "⚠ Warning: Setup may be incomplete"
    echo "Please ensure soluble model weights (.pt files) are in:"
    echo "  data/input/BioMPNN/soluble_model_weights/"
    echo ""
    echo "You can download them manually from:"
    echo "  https://files.ipd.uw.edu/pub/training_sets/soluble_model_weights/"
fi

echo ""
echo "=========================================="


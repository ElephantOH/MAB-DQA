#!/bin/bash
# Auto-select fastest source to download Mdocagent dataset

# Configuration
DATASET="Mdocagent-dataset"
HF_ORIGIN="https://huggingface.co/datasets/Lillianwei/$DATASET.git"
HF_MIRROR="https://hf-mirror.com/datasets/Lillianwei/$DATASET.git"
TARGET_DIR="datasets"

# Test connection speed (return time cost in seconds)
test_speed() {
    local url=$1
    echo "Testing: $url"
    local time_log=$( (time git ls-remote --heads "$url" >/dev/null 2>&1) 2>&1 )
    local cost=$(echo "$time_log" | grep real | awk '{print $2}' | sed 's/[ms]//g' | awk -F. '{print $1*60 + $2/100}')
    echo "Cost: $cost s"
    echo "$cost"
}

# Check git installation
if ! command -v git &> /dev/null; then
    echo "Error: Git is required"
    exit 1
fi

# Create target directory
echo -e "\n=== Prepare directory ==="
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1
echo "Current path: $(pwd)"

# Speed test
echo -e "\n=== Speed test ==="
t_origin=$(test_speed "$HF_ORIGIN")
t_mirror=$(test_speed "$HF_MIRROR")

# Choose faster source
echo -e "\n=== Select source ==="
if (( $(echo "$t_origin < $t_mirror" | bc -l) )); then
    DOWNLOAD_URL="$HF_ORIGIN"
    echo " Use official source"
else
    DOWNLOAD_URL="$HF_MIRROR"
    echo " Use mirror source"
fi

# Download dataset
echo -e "\n=== Start downloading ==="
git clone "$DOWNLOAD_URL" .

# Complete
echo -e "\n=== Done ==="
echo "Dataset path: $(pwd)"
ls -lh

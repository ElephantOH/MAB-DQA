#!/bin/bash
# Strict mode for robust scripting
set -euo pipefail
IFS=$'\n\t'

# ==============================================================================
# Configuration Section (High Maintainability: Modify ONLY this section)
# ==============================================================================
# Basic environment settings
ENV_NAME="mab_dqa"
PYTHON_VERSION="3.12.11"

# Core main packages (fixed versions, dependencies auto-installed)
CORE_PACKAGES=(
    "torch==2.8.0"
    "torchaudio==2.8.0"
    "torchvision==0.23.0"
    "transformers==4.56.0"
    "scikit-learn==1.7.2"
    "accelerate==1.10.1"
    "datasets==4.1.0"
    "sentence-transformers==5.1.0"
    "peft"
    "pandas"
    "numpy"
    "matplotlib"
    "openai"
    "llama-index"
    "fastapi"
    "uvicorn"
    "tqdm"
    "pyyaml"
    "python-dotenv"
    "hydra-core"
    "seaborn"
    "prettytable"
    "jsonlines"
    "tokenizers"
    "loguru"
    "requests"
    "setuptools"
    "bitsandbytes"
    "safetensors"
    "gpustat"
    "icecream"
    "pdf2image"
    "editdistance"
    "einops"
    "fire"
    "timm"
    "sentencepiece"
    "easyocr"
    "qwen-vl-utils"
    "faiss-cpu"
    "word2number"
    "pymupdf"
)

# Pip mirrors (auto speed test + select fastest)
PIP_MIRRORS=(
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://pypi.org/simple"
)

# Speed test timeout (seconds)
TEST_TIMEOUT=5

# ==============================================================================
# Core Utility Functions
# ==============================================================================
# Log to stderr (NO pollution to function return value)
log_info() {
    echo -e "\033[32m[INFO] $1\033[0m" >&2
}

log_warn() {
    echo -e "\033[33m[WARN] $1\033[0m" >&2
}

log_error() {
    echo -e "\033[31m[ERROR] $1\033[0m" >&2
}

# Check conda installation
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "Conda is not installed! Please install Anaconda/Miniconda first."
        exit 1
    fi
    log_info "Conda is detected successfully"
}

# Measure fastest pip mirror
measure_fastest_mirror() {
    local mirrors=("$@")
    local fastest_url=""
    local min_time=999999

    log_info "Testing speed for ${#mirrors[@]} mirrors, timeout: ${TEST_TIMEOUT}s"
    
    for url in "${mirrors[@]}"; do
        local response_time=$(curl -s -o /dev/null -w "%{time_total}" --max-time ${TEST_TIMEOUT} "${url}" || echo "999999")
        if [[ $(echo "$response_time < $min_time" | bc -l) -eq 1 ]]; then
            min_time=$response_time
            fastest_url=$url
        fi
        log_info "Mirror: ${url} | Response Time: ${response_time}s"
    done

    if [[ -z $fastest_url ]]; then
        log_error "All mirrors are unreachable!"
        exit 1
    fi

    log_info "Fastest mirror selected: ${fastest_url}"
    # ONLY output pure URL to stdout (for function return)
    echo "$fastest_url"
}

# ==============================================================================
# Installation Functions
# ==============================================================================
# Create conda environment
create_conda_env() {
    log_info "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
    if conda env list | grep -q "^$ENV_NAME "; then
        log_info "Environment $ENV_NAME already exists, skipping creation"
        return
    fi
    # Clean conda command (no extra parameters, fix Unicode error)
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
}

# Install pip packages with fastest mirror
pip_install_with_mirror() {
    local package=$1
    local fastest_pip=$2

    local host=$(echo "$fastest_pip" | cut -d '/' -f3)
    log_info "Installing $package from fastest mirror: $fastest_pip"
    
    if pip install "$package" -i "$fastest_pip" --trusted-host "$host"; then
        log_info "Successfully installed $package"
        return 0
    fi

    log_error "Primary mirror failed, trying backup mirrors..."
    for mirror in "${PIP_MIRRORS[@]}"; do
        local h=$(echo "$mirror" | cut -d '/' -f3)
        if pip install "$package" -i "$mirror" --trusted-host "$h"; then
            log_info "Successfully installed $package from backup mirror"
            return 0
        fi
    done

    log_error "All mirrors failed for package: $package"
    exit 1
}

# ==============================================================================
# Main Workflow
# ==============================================================================
main() {
    log_info "=== Starting Automated Setup for $ENV_NAME Environment ==="
    
    # Step 1: Check conda
    check_conda

    # Step 2: Initialize conda
    log_info "Initializing conda shell..."
    source "$(conda info --base)/etc/profile.d/conda.sh"

    # Step 3: Create & activate environment
    create_conda_env
    conda activate "$ENV_NAME"
    log_info "Activated conda environment: $ENV_NAME"

    # Step 4: Auto select FASTEST pip mirror
    local fastest_pip=$(measure_fastest_mirror "${PIP_MIRRORS[@]}")

    # Step 5: Upgrade pip
    log_info "Upgrading pip to latest version"
    local host=$(echo "$fastest_pip" | cut -d '/' -f3)
    pip install --upgrade pip -i "$fastest_pip" --trusted-host "$host"

    # Step 6: Install core packages
    log_info "=== Installing Core Main Packages ==="
    for pkg in "${CORE_PACKAGES[@]}"; do
        pip_install_with_mirror "$pkg" "$fastest_pip"
    done

    # Step 7: Verify installation
    log_info "=== Verifying Environment Installation ==="
    conda list | grep -E "torch|transformers|scikit-learn|pandas|numpy"
    log_info "All core packages installed successfully!"

    log_info "=== Setup Completed Successfully ==="
    log_info "Activate environment: conda activate $ENV_NAME"
}

# Execute main function
main

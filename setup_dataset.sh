#!/bin/bash
# Auto-select fastest source to download Mdocagent dataset
# Usage: ./setup_dataset.sh [DATASET_NAME ...]   e.g. ./setup_dataset.sh MMLongBench
# With no arguments, all datasets (FetaTab, LongDocURL, MMLongBench, PaperTab, PaperText) are downloaded.
set -euo pipefail

# Configuration
REPO_ID="Lillianwei/Mdocagent-dataset"
HF_ORIGIN="https://huggingface.co"
HF_MIRROR="https://hf-mirror.com"
TEST_TIMEOUT=10
MAX_WORKERS=4
MAX_ATTEMPTS=3
RETRY_DELAY=30
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$PROJECT_DIR/datasets"

# Test connection speed (prints time cost in seconds to stdout, logs to stderr).
# An endpoint only counts as usable if it serves file metadata itself
# (hf-mirror.com redirects clients outside mainland China, which breaks the hf CLI).
test_speed() {
    local endpoint=$1
    echo "Testing: $endpoint" >&2
    local headers_file
    headers_file=$(mktemp)
    local cost
    cost=$(curl -s -I -o "$headers_file" -w "%{time_total}" --max-time "$TEST_TIMEOUT" \
        "$endpoint/datasets/$REPO_ID/resolve/main/README.md" 2>/dev/null) || cost=999999
    if ! grep -qi "^x-repo-commit:" "$headers_file"; then
        echo "Unusable (no Hub metadata returned)" >&2
        cost=999999
    fi
    rm -f "$headers_file"
    echo "Cost: ${cost}s" >&2
    echo "$cost"
}

# Check uv installation (the hf CLI comes from the project environment)
if ! command -v uv &> /dev/null; then
    echo "Error: uv is required. Run ./setup_env.sh first."
    exit 1
fi

# Create target directory
echo -e "\n=== Prepare directory ==="
mkdir -p "$TARGET_DIR"
echo "Target path: $TARGET_DIR"

# Speed test (skip if HF_ENDPOINT is already set)
if [[ -z "${HF_ENDPOINT:-}" ]]; then
    echo -e "\n=== Speed test ==="
    t_origin=$(test_speed "$HF_ORIGIN")
    t_mirror=$(test_speed "$HF_MIRROR")

    echo -e "\n=== Select source ==="
    if (( $(echo "$t_origin <= $t_mirror" | bc -l) )); then
        export HF_ENDPOINT="$HF_ORIGIN"
        echo "Use official source"
    else
        export HF_ENDPOINT="$HF_MIRROR"
        echo "Use mirror source"
    fi
fi
echo "Endpoint: $HF_ENDPOINT"

# Restrict to requested datasets, if any
INCLUDE_ARGS=()
for name in "$@"; do
    INCLUDE_ARGS+=(--include "$name/*")
done

# Load HF_TOKEN from .env if present (anonymous downloads get rate limited)
ENV_ARGS=()
if [[ -f "$PROJECT_DIR/.env" ]]; then
    ENV_ARGS=(--env-file "$PROJECT_DIR/.env")
    echo "Loading environment from $PROJECT_DIR/.env"
fi

# Download dataset (resumable; files already present and up to date are skipped)
echo -e "\n=== Start downloading ==="
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if uv run ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} hf download "$REPO_ID" \
        --repo-type dataset \
        --local-dir "$TARGET_DIR" \
        --max-workers "$MAX_WORKERS" \
        --exclude ".gitattributes" ".gitignore" "README.md" \
        ${INCLUDE_ARGS[@]+"${INCLUDE_ARGS[@]}"}; then
        break
    fi
    if (( attempt == MAX_ATTEMPTS )); then
        echo "Error: download failed after $MAX_ATTEMPTS attempts"
        exit 1
    fi
    echo "Download failed (attempt $attempt/$MAX_ATTEMPTS), retrying in ${RETRY_DELAY}s..."
    sleep "$RETRY_DELAY"
done

# Upstream PaperText/documents is a symlink to an absolute path on the uploader's machine;
# it is meant to share PaperTab's documents.
PAPERTEXT_DOCS="$TARGET_DIR/PaperText/documents"
if [[ -e "$PAPERTEXT_DOCS" || -L "$PAPERTEXT_DOCS" ]] && [[ ! -d "$PAPERTEXT_DOCS" ]]; then
    echo "Linking PaperText/documents -> ../PaperTab/documents"
    rm -f "$PAPERTEXT_DOCS"
    ln -s ../PaperTab/documents "$PAPERTEXT_DOCS"
fi

# Complete
echo -e "\n=== Done ==="
echo "Dataset path: $TARGET_DIR"
ls -lh "$TARGET_DIR"

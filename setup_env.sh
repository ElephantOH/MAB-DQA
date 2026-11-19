#!/bin/bash
# Strict mode for robust scripting
set -euo pipefail
IFS=$'\n\t'

# ==============================================================================
# Configuration Section (High Maintainability: Modify ONLY this section)
# Dependencies are declared in pyproject.toml and pinned in uv.lock.
# The Python version is pinned in .python-version.
# ==============================================================================
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

# Check uv installation
check_uv() {
    if ! command -v uv &> /dev/null; then
        log_error "uv is not installed! Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    log_info "uv is detected successfully ($(uv --version))"
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
# Main Workflow
# ==============================================================================
main() {
    cd "$(dirname "$0")"
    log_info "=== Starting Automated Setup with uv ==="

    # Step 1: Check uv
    check_uv

    # Step 2: Auto select FASTEST pip mirror (skip if UV_DEFAULT_INDEX is already set)
    if [[ -z "${UV_DEFAULT_INDEX:-}" ]]; then
        export UV_DEFAULT_INDEX=$(measure_fastest_mirror "${PIP_MIRRORS[@]}")
    fi
    log_info "Using package index: $UV_DEFAULT_INDEX"

    # Step 3: Install Python (from .python-version) and sync dependencies into .venv
    log_info "=== Installing Python and Project Dependencies ==="
    uv python install
    uv sync

    # Step 4: Verify installation
    log_info "=== Verifying Environment Installation ==="
    uv pip list | grep -E "torch|transformers|scikit-learn|pandas|numpy"
    log_info "All core packages installed successfully!"

    log_info "=== Setup Completed Successfully ==="
    log_info "Activate environment: source .venv/bin/activate (or prefix commands with 'uv run')"
}

# Execute main function
main

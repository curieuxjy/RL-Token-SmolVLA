#!/bin/bash
# install.sh — Install RLT for SmolVLA into a LeRobot v0.4.4 checkout
#
# Usage:
#   bash install.sh /path/to/lerobot
#
# This script:
#   1. Copies smolvla_rlt/ policy module into LeRobot
#   2. Applies patches to factory.py and modeling_smolvla.py
#   3. Copies training/inference scripts
#   4. Copies tests

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: bash install.sh /path/to/lerobot${NC}"
    exit 1
fi

LEROBOT_DIR="$(cd "$1" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Validate LeRobot directory
if [ ! -f "$LEROBOT_DIR/src/lerobot/policies/factory.py" ]; then
    echo -e "${RED}Error: $LEROBOT_DIR doesn't look like a LeRobot checkout.${NC}"
    echo "Expected to find src/lerobot/policies/factory.py"
    exit 1
fi

echo -e "${GREEN}Installing RLT for SmolVLA into: $LEROBOT_DIR${NC}"
echo ""

# ── Step 1: Copy smolvla_rlt policy module ──────────────────────────────
POLICY_DIR="$LEROBOT_DIR/src/lerobot/policies/smolvla_rlt"
if [ -d "$POLICY_DIR" ]; then
    echo -e "${YELLOW}smolvla_rlt/ already exists — backing up to smolvla_rlt.bak/${NC}"
    rm -rf "$POLICY_DIR.bak"
    mv "$POLICY_DIR" "$POLICY_DIR.bak"
fi

echo "  Copying smolvla_rlt/ → $POLICY_DIR"
cp -r "$SCRIPT_DIR/smolvla_rlt" "$POLICY_DIR"

# ── Step 2: Apply patches ───────────────────────────────────────────────
echo ""
echo "  Applying patches to LeRobot..."

# Patch modeling_smolvla.py (adds extract_embeddings method)
MODELING_FILE="$LEROBOT_DIR/src/lerobot/policies/smolvla/modeling_smolvla.py"
if grep -q "extract_embeddings" "$MODELING_FILE" 2>/dev/null; then
    echo -e "  ${YELLOW}modeling_smolvla.py already patched (extract_embeddings found), skipping${NC}"
else
    echo "  Patching modeling_smolvla.py..."
    cd "$LEROBOT_DIR"
    if git apply "$SCRIPT_DIR/lerobot_patches/modeling_smolvla.patch" 2>/dev/null; then
        echo -e "  ${GREEN}modeling_smolvla.py patched successfully${NC}"
    else
        echo -e "  ${YELLOW}Patch failed (may need manual merge). Trying with --3way...${NC}"
        git apply --3way "$SCRIPT_DIR/lerobot_patches/modeling_smolvla.patch" 2>&1 || true
    fi
fi

# Patch factory.py (registers smolvla_rlt)
FACTORY_FILE="$LEROBOT_DIR/src/lerobot/policies/factory.py"
if grep -q "smolvla_rlt" "$FACTORY_FILE" 2>/dev/null; then
    echo -e "  ${YELLOW}factory.py already patched (smolvla_rlt found), skipping${NC}"
else
    echo "  Patching factory.py..."
    cd "$LEROBOT_DIR"
    if git apply "$SCRIPT_DIR/lerobot_patches/factory.patch" 2>/dev/null; then
        echo -e "  ${GREEN}factory.py patched successfully${NC}"
    else
        echo -e "  ${YELLOW}Patch failed (may need manual merge). Trying with --3way...${NC}"
        git apply --3way "$SCRIPT_DIR/lerobot_patches/factory.patch" 2>&1 || true
    fi
fi

# ── Step 3: Copy scripts ───────────────────────────────────────────────
SCRIPTS_DIR="$LEROBOT_DIR/../scripts"
mkdir -p "$SCRIPTS_DIR"
echo ""
echo "  Copying training scripts → $SCRIPTS_DIR"
cp "$SCRIPT_DIR/scripts/train_rlt_stage1.py" "$SCRIPTS_DIR/"
cp "$SCRIPT_DIR/scripts/train_rlt_stage2.py" "$SCRIPTS_DIR/"
cp "$SCRIPT_DIR/scripts/run_rlt_inference.py" "$SCRIPTS_DIR/"

# ── Step 4: Copy tests ─────────────────────────────────────────────────
TESTS_DIR="$LEROBOT_DIR/../tests/policies/smolvla_rlt"
mkdir -p "$TESTS_DIR"
echo "  Copying tests → $TESTS_DIR"
cp "$SCRIPT_DIR/tests/"*.py "$TESTS_DIR/"
# Ensure __init__.py exists
touch "$TESTS_DIR/__init__.py"

# ── Verify ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Installed files:"
echo "  $POLICY_DIR/"
echo "    ├── __init__.py"
echo "    ├── configuration_smolvla_rlt.py"
echo "    ├── modeling_smolvla_rlt.py"
echo "    └── processor_smolvla_rlt.py"
echo "  $SCRIPTS_DIR/"
echo "    ├── train_rlt_stage1.py"
echo "    ├── train_rlt_stage2.py"
echo "    └── run_rlt_inference.py"
echo "  $TESTS_DIR/"
echo "    └── test_smolvla_rlt.py"
echo ""
echo "Patched files:"
echo "  $MODELING_FILE"
echo "  $FACTORY_FILE"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Activate your LeRobot venv: source $LEROBOT_DIR/.venv/bin/activate"
echo "  2. Run tests: python -m pytest tests/policies/smolvla_rlt/ -v"
echo "  3. See RLT-Setup-Guide.md for the full training workflow"

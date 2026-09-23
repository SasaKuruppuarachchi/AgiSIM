#!/usr/bin/env bash
set -euo pipefail

# ANSI Color Codes
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_GREEN="\033[0;32m"
C_YELLOW="\033[0;33m"
C_CYAN="\033[0;36m"
C_RED="\033[0;31m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGISIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_ISAACSIM_PATH="${HOME}/isaacsim"

# ---------------------------------------------------------
# Usage / Help
# ---------------------------------------------------------
show_help() {
    echo -e "${C_BOLD}Usage:${C_RESET} $0 [ISAACSIM_PATH] [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --path PATH    Path to Isaac Sim installation (default: ${DEFAULT_ISAACSIM_PATH})"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ISAACSIM_PATH      Path to Isaac Sim installation"
    echo "  ISAACSIM_ASSET_ROOT  Local asset root (default: ~/isaac_sim_assets/Assets/Isaac/6.0)"
}

# ---------------------------------------------------------
# Parse Arguments
# ---------------------------------------------------------
INPUT_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--path)
            INPUT_PATH="$2"
            shift 2
            ;;
        *)
            if [ -z "$INPUT_PATH" ]; then
                INPUT_PATH="$1"
                shift
            else
                echo -e "${C_RED}[ERROR] Unknown option: $1${C_RESET}"
                show_help
                exit 1
            fi
            ;;
    esac
done

# ---------------------------------------------------------
# Step 0: Resolve Isaac Sim Installation Path
# ---------------------------------------------------------
if [ -n "$INPUT_PATH" ]; then
    ISAACSIM_PATH="$INPUT_PATH"
elif [ -z "${ISAACSIM_PATH:-}" ]; then
    echo -e "${C_YELLOW}[!] ISAACSIM_PATH is not specified.${C_RESET}"
    read -rp "$(echo -e "${C_CYAN}Enter Isaac Sim installation path [default: ${DEFAULT_ISAACSIM_PATH}]: ${C_RESET}")" USER_INPUT
    ISAACSIM_PATH="${USER_INPUT:-$DEFAULT_ISAACSIM_PATH}"
fi

# Expand tilde if present
ISAACSIM_PATH="${ISAACSIM_PATH/#\~/$HOME}"
export ISAACSIM_PATH

echo -e "\n${C_BOLD}${C_GREEN}=============================================${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}       Isaac Sim Customization Setup         ${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}=============================================${C_RESET}"
echo -e "${C_BOLD}Isaac Sim Path :${C_RESET} ${ISAACSIM_PATH}"
echo -e "${C_BOLD}AgiSIM Root    :${C_RESET} ${AGISIM_DIR}\n"

# Resolve the asset root in this process: exports in the child setup script
# cannot update our environment. Pass the same root to both setup steps.
if [ -z "${ISAACSIM_ASSET_ROOT:-}" ]; then
    ISAACSIM_ASSET_ROOT="${HOME}/isaac_sim_assets/Assets/Isaac/6.0"
    if [ ! -d "$ISAACSIM_ASSET_ROOT" ] && [ -d "${HOME}/isaac_sim_assets/assets/Isaac/6.0" ]; then
        ISAACSIM_ASSET_ROOT="${HOME}/isaac_sim_assets/assets/Isaac/6.0"
    fi
fi
ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT/#\~/$HOME}"
if [[ "$ISAACSIM_ASSET_ROOT" != /* ]]; then
    echo -e "${C_RED}[ERROR] ISAACSIM_ASSET_ROOT must be an absolute local directory.${C_RESET}"
    exit 1
fi
export ISAACSIM_ASSET_ROOT

# ---------------------------------------------------------
# Step 1: Run get_isaac_assets.sh interactively
# ---------------------------------------------------------
GET_ASSETS_SCRIPT="${SCRIPT_DIR}/get_isaac_assets.sh"

echo -e "${C_CYAN}${C_BOLD}[Step 1/4] Running Isaac Sim assets setup...${C_RESET}"
if [ -f "$GET_ASSETS_SCRIPT" ]; then
    chmod +x "$GET_ASSETS_SCRIPT"
    bash "$GET_ASSETS_SCRIPT"
else
    echo -e "${C_RED}[ERROR] ${GET_ASSETS_SCRIPT} not found! Skipping asset setup.${C_RESET}"
    exit 1
fi

# ---------------------------------------------------------
# Step 2: Copy Mid_360.usda LiDAR Sensor Asset
# ---------------------------------------------------------
echo -e "\n${C_CYAN}${C_BOLD}[Step 2/4] Installing Mid-360 LiDAR asset...${C_RESET}"

SRC_MID360="${AGISIM_DIR}/extensions/pegasus.simulator/pegasus/simulator/assets/sensors/lidar/Mid_360.usda"
DEST_DIR="${ISAACSIM_ASSET_ROOT}/Isaac/Sensors/NVIDIA"
DEST_MID360="${DEST_DIR}/Mid_360.usda"

if [ ! -f "$SRC_MID360" ]; then
    echo -e "${C_RED}[ERROR] Source asset not found at: ${SRC_MID360}${C_RESET}"
    exit 1
fi

mkdir -p "$DEST_DIR"
cp -v "$SRC_MID360" "$DEST_MID360"
echo -e "${C_GREEN}[OK] Successfully copied Mid_360.usda to:${C_RESET} ${DEST_MID360}"

# ---------------------------------------------------------
# Step 3: Register Mid_360.usda in RTX LiDAR configs
# ---------------------------------------------------------
echo -e "\n${C_CYAN}${C_BOLD}[Step 3/4] Registering Mid-360 LiDAR in RTX configs...${C_RESET}"

# Isaac Sim 6 keeps the legacy command configuration in extsDeprecated.
# The deprecated API may re-export the experimental registry.
LIDAR_CONFIG_CANDIDATES=(
    "${ISAACSIM_PATH}/exts/isaacsim.sensors.experimental.rtx/isaacsim/sensors/experimental/rtx/impl/rtx_lidar_configs.py"
    "${ISAACSIM_PATH}/exts/isaacsim.sensors.rtx/isaacsim/sensors/rtx/impl/supported_lidar_configs.py"
    "${ISAACSIM_PATH}/extsDeprecated/isaacsim.sensors.rtx/isaacsim/sensors/rtx/impl/supported_lidar_configs.py"
)

CONFIG_FOUND=false
for config_file in "${LIDAR_CONFIG_CANDIDATES[@]}"; do
    if [ ! -f "$config_file" ]; then
        continue
    fi
    registration=$(python3 - "$config_file" << 'PYEOF'
import ast
from pathlib import Path
import shutil
import sys

path = Path(sys.argv[1])
content = path.read_text(encoding="utf-8")
tree = ast.parse(content, filename=str(path))
registry = None
for node in tree.body:
    targets = node.targets if isinstance(node, ast.Assign) else (
        [node.target] if isinstance(node, ast.AnnAssign) else []
    )
    if any(isinstance(target, ast.Name) and target.id == "SUPPORTED_LIDAR_CONFIGS" for target in targets):
        registry = node.value
        break

if registry is None:
    # Isaac Sim 6's deprecated module re-exports the experimental registry.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "isaacsim.sensors.experimental.rtx":
            if any(alias.name == "SUPPORTED_LIDAR_CONFIGS" and alias.asname in (None, "SUPPORTED_LIDAR_CONFIGS") for alias in node.names):
                print("REEXPORT")
                sys.exit(0)
    sys.exit(f"No supported LiDAR registry definition found in {path}")
if not isinstance(registry, ast.Dict):
    sys.exit(f"Unsupported LiDAR registry format in {path}; file left unchanged")

asset = "/Isaac/Sensors/NVIDIA/Mid_360.usda"
if any(isinstance(key, ast.Constant) and key.value == asset for key in registry.keys):
    print("EXISTS")
    sys.exit(0)

# AST column offsets are UTF-8 byte offsets. Insert just inside the dictionary.
lines = content.encode("utf-8").splitlines(keepends=True)
offset = sum(map(len, lines[:registry.lineno - 1])) + registry.col_offset + 1
entry = f'\n    "{asset}": set(),\n'.encode("utf-8")
updated = (content.encode("utf-8")[:offset] + entry + content.encode("utf-8")[offset:]).decode("utf-8")
compile(updated, str(path), "exec")
backup = Path(str(path) + ".bak")
if not backup.exists():
    shutil.copy2(path, backup)
path.write_text(updated, encoding="utf-8")
print("ADDED")
PYEOF
    )
    case "$registration" in
        REEXPORT)
            echo -e "${C_GREEN}[SKIP] Uses the shared experimental registry:${C_RESET} ${config_file}"
            ;;
        EXISTS|ADDED)
            CONFIG_FOUND=true
            echo -e "${C_GREEN}[OK] Mid_360.usda registered in:${C_RESET} ${config_file} (${registration})"
            ;;
        *)
            echo -e "${C_RED}[ERROR] Unexpected registration result: ${registration}${C_RESET}"
            exit 1
            ;;
    esac
done

if [ "$CONFIG_FOUND" = false ]; then
    echo -e "${C_RED}[ERROR] No writable LiDAR registry definition found in ${ISAACSIM_PATH}.${C_RESET}"
    exit 1
fi

# ---------------------------------------------------------
# Step 4: Configure 'agisim' alias in ~/.bashrc
# ---------------------------------------------------------
echo -e "\n${C_CYAN}${C_BOLD}[Step 4/4] Configuring 'agisim' alias in ~/.bashrc...${C_RESET}"

RUN_SIM_SCRIPT="${SCRIPT_DIR}/run_sim.sh"
if [ -f "$RUN_SIM_SCRIPT" ]; then
    chmod +x "$RUN_SIM_SCRIPT"
fi

BASHRC_FILE="${HOME}/.bashrc"
ALIAS_LINE="alias agisim=\"${RUN_SIM_SCRIPT}\""

if [ -f "$BASHRC_FILE" ] && grep -q "alias agisim=" "$BASHRC_FILE"; then
    CURRENT_ALIAS=$(grep "alias agisim=" "$BASHRC_FILE" | tail -n1)
    echo -e "${C_YELLOW}[!] 'agisim' alias is already defined in ${BASHRC_FILE}:${C_RESET}"
    echo -e "    ${C_BOLD}${CURRENT_ALIAS}${C_RESET}"
    sed -i "s|alias agisim=.*|${ALIAS_LINE}|" "$BASHRC_FILE"
    echo -e "${C_GREEN}[OK] Updated alias in ${BASHRC_FILE} to: ${ALIAS_LINE}${C_RESET}"
else
    echo "" >> "$BASHRC_FILE"
    echo "# AgiSIM launcher alias" >> "$BASHRC_FILE"
    echo "${ALIAS_LINE}" >> "$BASHRC_FILE"
    echo -e "${C_GREEN}[OK] Added alias to ${BASHRC_FILE}: ${ALIAS_LINE}${C_RESET}"
fi

# ---------------------------------------------------------
# Future Customization Steps (Placeholder)
# ---------------------------------------------------------
# Add remaining customization steps below as needed

echo -e "\n${C_BOLD}${C_GREEN}✔ Isaac Sim setup completed successfully.${C_RESET}"



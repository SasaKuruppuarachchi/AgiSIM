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
DEST_DIR="${ISAACSIM_PATH}/Assets/Isaac/6.0/Isaac/Sensors/NVIDIA"
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

# Support both experimental (earlier Isaac Sim) and standard (Isaac Sim 4.5/6.0+) config paths
LIDAR_CONFIG_CANDIDATES=(
    "${ISAACSIM_PATH}/exts/isaacsim.sensors.experimental.rtx/isaacsim/sensors/experimental/rtx/impl/rtx_lidar_configs.py"
    "${ISAACSIM_PATH}/exts/isaacsim.sensors.rtx/isaacsim/sensors/rtx/impl/supported_lidar_configs.py"
)

CONFIG_FOUND=false
for config_file in "${LIDAR_CONFIG_CANDIDATES[@]}"; do
    if [ -f "$config_file" ]; then
        CONFIG_FOUND=true
        if grep -Fq '"/Isaac/Sensors/NVIDIA/Mid_360.usda"' "$config_file"; then
            echo -e "${C_GREEN}[SKIP] Mid_360.usda is already registered in:${C_RESET} ${config_file}"
        else
            echo -e "${C_YELLOW}Registering Mid_360.usda in:${C_RESET} ${config_file}..."
            cp "$config_file" "${config_file}.bak"
            python3 - "$config_file" << 'PYEOF'
import sys
import re

file_path = sys.argv[1]
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

entry = '    "/Isaac/Sensors/NVIDIA/Mid_360.usda": set(),\n'
pattern = r'(SUPPORTED_LIDAR_CONFIGS\s*(?::[^=]+)?=\s*\{[^\n]*\n)'

if re.search(pattern, content):
    new_content = re.sub(pattern, r'\1' + entry, content, count=1)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS")
else:
    print("ANCHOR_NOT_FOUND")
    sys.exit(1)
PYEOF
            echo -e "${C_GREEN}[OK] Successfully registered Mid_360.usda in:${C_RESET} ${config_file}"
        fi
    fi
done

if [ "$CONFIG_FOUND" = false ]; then
    echo -e "${C_RED}[WARN] No RTX LiDAR config file found in Isaac Sim installation (${ISAACSIM_PATH}). Skipping registration.${C_RESET}"
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



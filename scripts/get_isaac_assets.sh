#!/usr/bin/env bash
set -euo pipefail

# ANSI Color Codes
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_GREEN="\033[0;32m"
C_YELLOW="\033[0;33m"
C_CYAN="\033[0;36m"
C_RED="\033[0;31m"

DEFAULT_ISAACSIM_PATH="${HOME}/isaacsim"

# ---------------------------------------------------------
# Step 1: Environment & Directory Resolution
# ---------------------------------------------------------
if [ -z "${ISAACSIM_PATH:-}" ]; then
    echo -e "${C_YELLOW}[!] ISAACSIM_PATH is not set in your environment.${C_RESET}"
    read -rp "$(echo -e "${C_CYAN}Enter Isaac Sim installation path [default: ${DEFAULT_ISAACSIM_PATH}]: ${C_RESET}")" USER_INPUT
    ISAACSIM_PATH="${USER_INPUT:-$DEFAULT_ISAACSIM_PATH}"
fi

# Expand tilde if passed manually
ISAACSIM_PATH="${ISAACSIM_PATH/#\~/$HOME}"
mkdir -p "$ISAACSIM_PATH"

TARGET_DIR="${HOME}/isaac_sim_assets"
mkdir -p "$TARGET_DIR"

ASSET_ROOT_PATH="${TARGET_DIR}/Assets/Isaac/6.0"
[ ! -d "$ASSET_ROOT_PATH" ] && [ -d "${TARGET_DIR}/assets/Isaac/6.0" ] && ASSET_ROOT_PATH="${TARGET_DIR}/assets/Isaac/6.0"

echo -e "\n${C_BOLD}${C_GREEN}=== Configuration ===${C_RESET}"
echo -e "${C_BOLD}ISAACSIM_PATH   :${C_RESET} ${ISAACSIM_PATH}"
echo -e "${C_BOLD}TARGET_DIR      :${C_RESET} ${TARGET_DIR}"
echo -e "${C_BOLD}ASSET_ROOT_PATH :${C_RESET} ${ASSET_ROOT_PATH}\n"

# ---------------------------------------------------------
# Precaution Check: Existing Asset Directory Tree
# ---------------------------------------------------------
SKIP_DOWNLOAD_EXTRACTION=false

if [ -d "$ASSET_ROOT_PATH" ] && [ "$(ls -A "$ASSET_ROOT_PATH" 2>/dev/null)" ]; then
    echo -e "${C_YELLOW}[PRECAUTION] Detected populated asset root at:${C_RESET} ${ASSET_ROOT_PATH}"
    read -rp "$(echo -e "${C_CYAN}Do you want to SKIP downloading and extraction? [Y/n]: ${C_RESET}")" skip_prompt
    skip_prompt="${skip_prompt:-Y}"
    if [[ "$skip_prompt" =~ ^[Yy]$ ]]; then
        SKIP_DOWNLOAD_EXTRACTION=true
        echo -e "${C_GREEN}Skipping asset download and extraction steps.${C_RESET}\n"
    fi
fi

# ---------------------------------------------------------
# Steps 2-6: Download, Verify, Merge, and Extract
# ---------------------------------------------------------
if [ "$SKIP_DOWNLOAD_EXTRACTION" = false ]; then
    cd "$TARGET_DIR"
    DOCS_URL="https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/download.html"

    echo -e "${C_CYAN}${C_BOLD}[1/5] Querying download links from Isaac Sim documentation...${C_RESET}"
    ASSET_URLS=$(curl -skSL "$DOCS_URL" | grep -oE 'https://[^"'\'' ]+isaac[_-]sim[_-]assets[^"'\'' ]+' | sort -u)

    if [ -z "$ASSET_URLS" ]; then
        echo -e "${C_YELLOW}Docs link parsing fell back to generic regex pattern...${C_RESET}"
        ASSET_URLS=$(curl -skSL "$DOCS_URL" | grep -oiE 'https://[^\"]+(\.zip|\.part[0-9]+|\.00[1-5]\.zip)' | grep -i "asset" | sort -u)
    fi

    echo -e "\n${C_CYAN}${C_BOLD}[2/5] Downloading archive parts...${C_RESET}"
    for url in $ASSET_URLS; do
        filename=$(basename "$url" | cut -d'?' -f1)
        echo -e "${C_YELLOW}--> Fetching: ${C_RESET}${filename}"
        
        if command -v aria2c >/dev/null 2>&1; then
            aria2c --check-certificate=false -x 16 -s 16 -c "$url" -o "$filename"
        elif command -v wget >/dev/null 2>&1; then
            wget --no-check-certificate -c "$url" -O "$filename"
        else
            curl -k -C - -L "$url" -o "$filename"
        fi
    done

    echo -e "\n${C_CYAN}${C_BOLD}[3/5] Validating MD5 Checksums...${C_RESET}"
    declare -A CHECKSUMS=(
        ["isaac-sim-assets-complete-6.0.1.001.zip"]="92149a1f50a21c0f04cca6507ab00653"
        ["isaac-sim-assets-complete-6.0.1.002.zip"]="9b4b924e2d31bce41712d7637a0d6e42"
        ["isaac-sim-assets-complete-6.0.1.003.zip"]="b1c62924beda91251d3f5318ffec2b00"
        ["isaac-sim-assets-complete-6.0.1.004.zip"]="6bd7aa4d9b6c4161c2302e4c9418ade7"
        ["isaac-sim-assets-complete-6.0.1.005.zip"]="c4a17942014be6b50492ae860496fef7"
    )

    checksum_failed=false
    for file in "${!CHECKSUMS[@]}"; do
        if [ ! -f "$file" ]; then
            echo -e "${C_RED}[MISSING] ${file}${C_RESET}"
            checksum_failed=true
            continue
        fi

        actual_md5=$(md5sum "$file" | awk '{print $1}')
        expected_md5="${CHECKSUMS[$file]}"

        if [ "$actual_md5" = "$expected_md5" ]; then
            echo -e "${C_GREEN}[PASS]${C_RESET} ${file}"
        else
            echo -e "${C_RED}[FAIL]${C_RESET} ${file} (Expected: ${expected_md5}, Received: ${actual_md5})"
            checksum_failed=true
        fi
    done

    if [ "$checksum_failed" = true ]; then
        echo -e "\n${C_RED}${C_BOLD}[ERROR] MD5 verification failed. Aborting before merging.${C_RESET}"
        exit 1
    fi

    echo -e "\n${C_CYAN}${C_BOLD}[4/5] Concatenating parts into complete_assets.zip...${C_RESET}"
    cat isaac-sim-assets-complete-6.0.1.001.zip \
        isaac-sim-assets-complete-6.0.1.002.zip \
        isaac-sim-assets-complete-6.0.1.003.zip \
        isaac-sim-assets-complete-6.0.1.004.zip \
        isaac-sim-assets-complete-6.0.1.005.zip > complete_assets.zip

    echo -e "${C_GREEN}[OK] Consolidated into complete_assets.zip${C_RESET}"
    rm -f isaac-sim-assets-complete-6.0.1.00*.zip
    echo -e "${C_GREEN}[CLEANUP] Deleted split files (.001 to .005).${C_RESET}"

    echo -e "\n${C_CYAN}${C_BOLD}[5/5] Archive Extraction${C_RESET}"
    read -rp "$(echo -e "${C_YELLOW}Do you want to extract complete_assets.zip now inside ${TARGET_DIR}? [y/N]: ${C_RESET}")" do_extract

    if [[ "$do_extract" =~ ^[Yy]$ ]]; then
        echo -e "${C_GREEN}Extracting complete_assets.zip into ${TARGET_DIR}...${C_RESET}"
        unzip -q complete_assets.zip
        echo -e "${C_GREEN}Extraction finished. Assets folder retained in: ${TARGET_DIR}/Assets${C_RESET}"

        read -rp "$(echo -e "\n${C_YELLOW}Do you want to remove complete_assets.zip? [y/N]: ${C_RESET}")" do_clean_archive
        if [[ "$do_clean_archive" =~ ^[Yy]$ ]]; then
            rm -f complete_assets.zip
            echo -e "${C_GREEN}[CLEANUP] complete_assets.zip deleted.${C_RESET}"
        else
            echo -e "${C_YELLOW}Retained archive at: ${TARGET_DIR}/complete_assets.zip${C_RESET}"
        fi
    else
        echo -e "${C_YELLOW}Skipped extraction. Consolidated archive saved at: ${TARGET_DIR}/complete_assets.zip${C_RESET}"
    fi

    # Update path if extracted folder was lowercased
    [ ! -d "$ASSET_ROOT_PATH" ] && [ -d "${TARGET_DIR}/assets/Isaac/6.0" ] && ASSET_ROOT_PATH="${TARGET_DIR}/assets/Isaac/6.0"
fi

# ---------------------------------------------------------
# Step 7a: Configure isaacsim.exp.base.kit
# ---------------------------------------------------------
echo -e "\n${C_CYAN}${C_BOLD}Configuring Isaac Sim Kit Settings...${C_RESET}"
KIT_FILE="${ISAACSIM_PATH}/apps/isaacsim.exp.base.kit"

if [ -f "$KIT_FILE" ]; then
    if grep -Fq "persistent.isaac.asset_root.default" "$KIT_FILE"; then
        echo -e "${C_YELLOW}[!] 'persistent.isaac.asset_root.default' is already set in ${KIT_FILE}.${C_RESET}"
        read -rp "$(echo -e "${C_CYAN}Do you want to skip modifying ${KIT_FILE}? [Y/n]: ${C_RESET}")" skip_kit
        skip_kit="${skip_kit:-Y}"
        if [[ "$skip_kit" =~ ^[Yy]$ ]]; then
            echo -e "${C_GREEN}Skipping ${KIT_FILE} modification.${C_RESET}"
        else
            cp "$KIT_FILE" "${KIT_FILE}.bak"
            cat <<EOF >> "$KIT_FILE"

# Isaac Sim Local Asset Pack Settings
[settings]
persistent.isaac.asset_root.default = "${ASSET_ROOT_PATH}"
exts."isaacsim.gui.content_browser".folders = [
    "${ASSET_ROOT_PATH}/Isaac/Robots",
    "${ASSET_ROOT_PATH}/Isaac/Robots_Multiphysics",
    "${ASSET_ROOT_PATH}/Isaac/People",
    "${ASSET_ROOT_PATH}/Isaac/IsaacLab",
    "${ASSET_ROOT_PATH}/Isaac/Props",
    "${ASSET_ROOT_PATH}/Isaac/Environments",
    "${ASSET_ROOT_PATH}/Isaac/Materials",
    "${ASSET_ROOT_PATH}/Isaac/Samples",
    "${ASSET_ROOT_PATH}/Isaac/Sensors",
]
EOF
            echo -e "${C_GREEN}[OK] Appended asset configurations to ${KIT_FILE}.${C_RESET}"
        fi
    else
        echo -e "${C_YELLOW}Appending local assets configuration to ${KIT_FILE}...${C_RESET}"
        cp "$KIT_FILE" "${KIT_FILE}.bak"
        cat <<EOF >> "$KIT_FILE"

# Isaac Sim Local Asset Pack Settings
[settings]
persistent.isaac.asset_root.default = "${ASSET_ROOT_PATH}"
exts."isaacsim.gui.content_browser".folders = [
    "${ASSET_ROOT_PATH}/Isaac/Robots",
    "${ASSET_ROOT_PATH}/Isaac/Robots_Multiphysics",
    "${ASSET_ROOT_PATH}/Isaac/People",
    "${ASSET_ROOT_PATH}/Isaac/IsaacLab",
    "${ASSET_ROOT_PATH}/Isaac/Props",
    "${ASSET_ROOT_PATH}/Isaac/Environments",
    "${ASSET_ROOT_PATH}/Isaac/Materials",
    "${ASSET_ROOT_PATH}/Isaac/Samples",
    "${ASSET_ROOT_PATH}/Isaac/Sensors",
]
EOF
        echo -e "${C_GREEN}[OK] Successfully updated ${KIT_FILE} (backup saved as ${KIT_FILE}.bak)${C_RESET}"
    fi
else
    echo -e "${C_RED}[WARN] Kit file not found at: ${KIT_FILE}. Skipping.${C_RESET}"
fi

# ---------------------------------------------------------
# Step 7b: Configure ~/.bashrc
# ---------------------------------------------------------
echo -e "\n${C_CYAN}${C_BOLD}Configuring ~/.bashrc...${C_RESET}"
BASHRC_FILE="${HOME}/.bashrc"
BASHRC_EXPORT="export ISAACSIM_ASSET_ROOT=\"${ASSET_ROOT_PATH}\""

if grep -q "ISAACSIM_ASSET_ROOT=" "$BASHRC_FILE"; then
    CURRENT_LINE=$(grep "ISAACSIM_ASSET_ROOT=" "$BASHRC_FILE" | tail -n1)
    echo -e "${C_YELLOW}[!] ISAACSIM_ASSET_ROOT is already defined in ${BASHRC_FILE}:${C_RESET}"
    echo -e "    ${C_BOLD}${CURRENT_LINE}${C_RESET}"
    read -rp "$(echo -e "${C_CYAN}Do you want to skip updating ~/.bashrc? [Y/n]: ${C_RESET}")" skip_bashrc
    skip_bashrc="${skip_bashrc:-Y}"
    if [[ "$skip_bashrc" =~ ^[Yy]$ ]]; then
        echo -e "${C_GREEN}Skipping ~/.bashrc modification.${C_RESET}"
    else
        sed -i "s|export ISAACSIM_ASSET_ROOT=.*|${BASHRC_EXPORT}|" "$BASHRC_FILE"
        echo -e "${C_GREEN}[OK] ~/.bashrc updated with: ${BASHRC_EXPORT}${C_RESET}"
    fi
else
    echo -e "${C_YELLOW}Adding ISAACSIM_ASSET_ROOT to ~/.bashrc...${C_RESET}"
    echo "" >> "$BASHRC_FILE"
    echo "# NVIDIA Isaac Sim Asset Root" >> "$BASHRC_FILE"
    echo "${BASHRC_EXPORT}" >> "$BASHRC_FILE"
    echo -e "${C_GREEN}[OK] ~/.bashrc updated with: ${BASHRC_EXPORT}${C_RESET}"
fi
source ~/.bashrc

echo -e "\n${C_BOLD}${C_GREEN}✔ All checks and configurations completed.${C_RESET}"
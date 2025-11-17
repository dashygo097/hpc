#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$BASE_DIR/build"

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
GRAY='\033[1;90m'
NC='\033[0m' 
BOLD='\033[1m'
DIM='\033[2m'
SPACE='                                                             '

show_header() {
  echo -e "${BLUE}"
  echo "                 ██╗  ██╗██████╗  ██████╗"
  echo "                 ██║  ██║██╔══██╗██╔════╝"
  echo "                 ███████║██████╔╝██║     "
  echo "                 ██╔══██║██╔═══╝ ██║     "
  echo "                 ██║  ██║██║     ╚██████╗"
  echo "                 ╚═╝  ╚═╝╚═╝      ╚═════╝"
  echo -e "${NC}"
  echo -e "${DIM}                HPC Cmake Project Builder${NC}"
  echo -e "${DIM}──────────────────────────────────────────────────────────${NC}"
  echo
}

show_status() {
  local status=$1
  local message=$2
  
  case $status in
    "info") echo -e "${DIM}│  ${message}${NC}" ;;
    "success") echo -e "${GREEN}✔  ${message}${NC}" ;;
    "warning") echo -e "${YELLOW}⚠  ${message}${NC}" ;;
    "error") echo -e "${RED}✖  ${message}${NC}" ;;
    *) echo -e "${DIM}│  ${message}${NC}" ;;
  esac
}

select_cmake_build_type() {
  local build_type=""
  local options=("None" "Release" "Debug" "RelWithDebInfo" "MinSizeRel" "Custom")
  local current_selection=0
    
  echo -e "${DIM}${BLUE}◇ Select CMake Build Type${NC}" >&2
  echo -e "│${DIM} Use arrow keys to navigate, Enter to confirm${NC}" >&2
    
  local menu_lines=0
    
  display_menu() {
    for i in "${!options[@]}"; do
      if [ $i -eq "$current_selection" ]; then
        echo -e "│ ${BOLD}${GREEN}❯ ${options[i]}${NC}${SPACE}" >&2
      else
        echo -e "│ ${DIM}${GRAY}${options[i]}${NC}${SPACE}" >&2
      fi
    done
    menu_lines="${#options[@]}"
  }
    
  display_menu
    
  echo -ne "\033[s" >&2
    
  while true; do
    read -rsn1 key
        
    if [ "$key" = $'\x1b' ]; then
      read -rsn1 -t 1 key2
        if [ "$key2" = '[' ]; then
          read -rsn1 key3
          case "$key3" in
            'A')
              if [ "$current_selection" -gt 0 ]; then
                current_selection="$((current_selection - 1))"
                echo -ne "\033[u" >&2
                echo -ne "\033[${menu_lines}A\033[J" >&2
                display_menu
              fi
              ;;
            'B') 
              if [ "$current_selection" -lt "$((${#options[@]} - 1))" ]; then
                current_selection="$((current_selection + 1))"
                echo -ne "\033[u" >&2
                echo -ne "\033[${menu_lines}A\033[J" >&2
                display_menu
              fi
              ;;
            esac
          fi
        elif [ "$key" = "" ]; then
      break
    fi
  done
    
  echo -ne "\033[u\033[J" >&2 
    
  case "${options[current_selection]}" in
    "None") 
      build_type=""
      ;;
    "Custom")
      echo -n "Please enter a custom build type: " >&2
      read custom_bt
      build_type="${custom_bt:-Release}"
      echo -ne "\033[A\033[2K" >&2
      ;;
    *)
      build_type="${options[current_selection]}"
      ;;
  esac
    
  echo -e "\033[8A\033[2K${DIM}${BLUE}◆ Build Type Selected!${NC}${SPACE}" >&2
  display_menu
  echo -e "${GREEN}✔ Build Type: ${NC}${BOLD}$build_type${NC}" >&2
    
  echo "$build_type"
}

# Build script for a CMake project
bootstrap() {
  show_header
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"
  build_type=$(select_cmake_build_type)
  cmake "$BASE_DIR" -DCMAKE_BUILD_TYPE="$build_type"
  cd "$CURRENT_DIR"
}

# Main execution
bootstrap

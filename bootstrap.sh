#!/bin/bash
set -euo pipefail

CURRENT_DIR=$(pwd)
BUILD_DIR=$CURRENT_DIR/build

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
GRAY='\033[1;37m'
NC='\033[0m' 
BOLD='\033[1m'
DIM='\033[2m'

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

# Build script for a CMake project
bootstrap() {
  show_header
  mkdir -p $BUILD_DIR
  cd $BUILD_DIR
  cmake ..
  cd $CURRENT_DIR
}

# Main execution
bootstrap

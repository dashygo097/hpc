PROJECT_DIR = $(shell pwd)
CMAKE_DIR = $(PROJECT_DIR)/cmake
BUILD_DIR = $(PROJECT_DIR)/build
CUDA_EXEC_DIR = $(PROJECT_DIR)/bin/device
OUTPUT_DIR = $(PROJECT_DIR)/.profile

# Build configuration
BUILD_TYPE ?= Release
GENERATOR ?= Ninja 

# Detect generator command
ifeq ($(GENERATOR),Ninja)
    BUILD_CMD = ninja -C $(BUILD_DIR)
else
    BUILD_CMD = cmake --build $(BUILD_DIR)
endif

.PHONY: all config build clean reconfigure help test profile profile-single

all: build

# Configure CMake project
config:
	@echo "==> Configuring project..."
	@mkdir -p $(BUILD_DIR)
	@if [ ! -f $(BUILD_DIR)/config.cmake ]; then \
		echo "Copying default config.cmake to build directory..."; \
		cp $(CMAKE_DIR)/config.cmake $(BUILD_DIR)/config.cmake; \
	else \
		echo "Using existing config.cmake in build directory"; \
	fi

# Build the project
build: config
	@if [ ! -f $(BUILD_DIR)/CMakeCache.txt ]; then \
		echo "Running CMake with generator: $(GENERATOR)"; \
		cd $(BUILD_DIR) && cmake $(PROJECT_DIR) \
			-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
			-G $(GENERATOR); \
	fi
	@echo "==> Building project with $(GENERATOR)..."
	@$(BUILD_CMD)

# Force reconfigure
reconfigure:
	@echo "==> Forcing reconfiguration..."
	@rm -f $(BUILD_DIR)/CMakeCache.txt
	@$(MAKE) config

# Clean build artifacts
clean:
	@echo "==> Cleaning build directory..."
	@rm -rf $(BUILD_DIR)

# Clean profiling data
clean-profile:
	@echo "==> Cleaning profiling data..."
	@rm -rf $(OUTPUT_DIR)

# Run tests
test: build
	@echo "==> Running tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure -j$(JOBS)

# Profile all CUDA executables
profile: build
	@echo "==> Profiling all CUDA executables..."
	@mkdir -p $(OUTPUT_DIR)
	@if [ ! -d $(CUDA_EXEC_DIR) ]; then \
		echo "Error: CUDA executable directory not found: $(CUDA_EXEC_DIR)"; \
		exit 1; \
	fi
	@found=0; \
	for exec_file in $(CUDA_EXEC_DIR)/*; do \
		if [ -x "$$exec_file" ] && [ ! -d "$$exec_file" ]; then \
			found=1; \
			exec_name=$$(basename "$$exec_file"); \
			echo "Profiling $$exec_name..."; \
			sudo ncu --set full --target-processes all -f -o "$(OUTPUT_DIR)/$$exec_name" "$$exec_file" || echo "Warning: Failed to profile $$exec_name"; \
			echo "Profile data for $$exec_name saved in $(OUTPUT_DIR)/$$exec_name.ncu-rep"; \
		fi; \
	done; \
	if [ $$found -eq 0 ]; then \
		echo "Warning: No executable files found in $(CUDA_EXEC_DIR)"; \
	fi

# Profile a single CUDA executable
# Usage: make profile-single EXEC=executable_name
profile-single: build
	@if [ -z "$(EXEC)" ]; then \
		echo "Error: EXEC variable not set. Usage: make profile-single EXEC=executable_name"; \
		exit 1; \
	fi
	@echo "==> Profiling $(EXEC)..."
	@mkdir -p $(OUTPUT_DIR)
	@if [ ! -f $(CUDA_EXEC_DIR)/$(EXEC) ]; then \
		echo "Error: Executable not found: $(CUDA_EXEC_DIR)/$(EXEC)"; \
		exit 1; \
	fi
	@if [ ! -x $(CUDA_EXEC_DIR)/$(EXEC) ]; then \
		echo "Error: File is not executable: $(CUDA_EXEC_DIR)/$(EXEC)"; \
		exit 1; \
	fi
	@sudo ncu --set full --target-processes all -f -o "$(OUTPUT_DIR)/$(EXEC)" "$(CUDA_EXEC_DIR)/$(EXEC)"
	@echo "Profile data for $(EXEC) saved in $(OUTPUT_DIR)/$(EXEC).ncu-rep"

# List available CUDA executables
list-cuda-execs:
	@echo "==> Available CUDA executables:"
	@if [ -d $(CUDA_EXEC_DIR) ]; then \
		for exec_file in $(CUDA_EXEC_DIR)/*; do \
			if [ -x "$$exec_file" ] && [ ! -d "$$exec_file" ]; then \
				basename "$$exec_file"; \
			fi; \
		done; \
	else \
		echo "Directory not found: $(CUDA_EXEC_DIR)"; \
	fi

# Help message
help:
	@echo "Available targets:"
	@echo "  all              - Configure and build (default)"
	@echo "  config           - Configure CMake project"
	@echo "  build            - Build the project"
	@echo "  reconfigure      - Force CMake reconfiguration"
	@echo "  clean            - Remove build directory"
	@echo "  clean-profile    - Remove profiling data"
	@echo "  test             - Run tests"
	@echo "  profile          - Profile all CUDA executables with Nsight Compute"
	@echo "  profile-single   - Profile a single CUDA executable (EXEC=name)"
	@echo "  list-cuda-execs  - List available CUDA executables"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_TYPE       - Build type (Debug, Release, RelWithDebInfo, MinSizeRel)"
	@echo "                     Current: $(BUILD_TYPE)"
	@echo "  GENERATOR        - CMake generator (Ninja, 'Unix Makefiles', etc.)"
	@echo "                     Current: $(GENERATOR)"
	@echo "  EXEC             - Executable name for profile-single target"
	@echo ""
	@echo "Examples:"
	@echo "  make profile                    - Profile all executables"
	@echo "  make profile-single EXEC=myapp  - Profile specific executable"
	@echo "  make clean-profile              - Clean profiling data"

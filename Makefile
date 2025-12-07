PROJECT_DIR = $(shell pwd)
CMAKE_DIR = $(PROJECT_DIR)/cmake
BUILD_DIR = $(PROJECT_DIR)/build

BUILD_TYPE ?= Debug 

.PHONY: all pre build

all: config build

config:
	@mkdir -p $(BUILD_DIR)
	@if [ ! -f $(BUILD_DIR)/config.cmake ]; then \
		echo "Copying default config.cmake to build directory... "; \
		cp $(CMAKE_DIR)/config.cmake $(BUILD_DIR)/config.cmake; \
	else \
		echo "Using existing config.cmake in build directory"; \
	fi

build: config 
	@$(MAKE) -C $(BUILD_DIR) all || ( \
		cd $(BUILD_DIR) && cmake $(PROJECT_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE); \
		$(MAKE) -C $(BUILD_DIR) all; \
	)

clean:
	@rm -rf $(BUILD_DIR)

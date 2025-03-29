# LLM Orchestrator Kernel Module Makefile

# Module name and object files
MODULE_NAME = llm_orchestrator
obj-m += $(MODULE_NAME).o
$(MODULE_NAME)-objs := orchestrator_main.o llm_context.o llm_scheduler.o llm_json_manager.o \
                       llm_network.o llm_tls.o llm_memory_management.o

# Kernel build directory
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# Compiler flags
ccflags-y := -Wall -Werror -Wextra -Wno-unused-parameter -std=gnu99 -g -DDEBUG

# Check for kernel version to determine TLS support
KERNEL_VERSION_CODE := $(shell grep UTS_RELEASE $(KERNELDIR)/include/generated/utsrelease.h | cut -d '"' -f2 | awk -F. '{ printf("%d%02d%02d", $$1, $$2, $$3); }')
KTLS_MIN_VERSION := 41300  # 4.13.0

ifneq ($(shell test $(KERNEL_VERSION_CODE) -ge $(KTLS_MIN_VERSION); echo $$?),0)
    ccflags-y += -DKTLS_NOT_SUPPORTED
    $(warning WARNING: Kernel version $(KERNEL_VERSION_CODE) does not support kTLS (minimum required is $(KTLS_MIN_VERSION)))
    $(warning WARNING: Building with software TLS fallback only)
endif

# Default target
.PHONY: all
all:
	@echo "Building LLM Orchestrator Kernel Module ($(MODULE_NAME))"
	@echo "Kernel directory: $(KERNELDIR)"
	@echo "Object files: $($(MODULE_NAME)-objs)"
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

# Clean target
.PHONY: clean
clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
	rm -f tests/test_app
	rm -rf doc/html doc/latex

# Install target
.PHONY: install
install:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install
	depmod -a
	@echo "Module installed to /lib/modules/$(shell uname -r)/extra/$(MODULE_NAME).ko"

# Uninstall target
.PHONY: uninstall
uninstall:
	rm -f /lib/modules/$(shell uname -r)/extra/$(MODULE_NAME).ko
	depmod -a
	@echo "Module uninstalled"

# Load module with API keys
.PHONY: load
load:
	@if [ -z "$(OPENAI_API_KEY)" ] || [ -z "$(ANTHROPIC_API_KEY)" ] || [ -z "$(GEMINI_API_KEY)" ]; then \
		echo "Error: You must set OPENAI_KEY, ANTHROPIC_KEY, and GEMINI_KEY environment variables."; \
		echo "Example: OPENAI_KEY=sk-xxx ANTHROPIC_KEY=sk-ant-xxx GEMINI_KEY=xxxx make load"; \
		exit 1; \
	fi
	sudo insmod $(MODULE_NAME).ko openai_api_key="$(OPENAI_API_KEY)" anthropic_api_key="$(ANTHROPIC_API_KEY)" \
		google_gemini_api_key="$(GEMINI_API_KEY)" prune_threshold_mins=60
	@echo "Module loaded with API keys"
	@if [ -e "/proc/modules" ] && grep -q $(MODULE_NAME) /proc/modules; then \
		echo "Success: Module loaded correctly"; \
		sudo dmesg | tail -n 20 | grep -i "$(MODULE_NAME)"; \
	else \
		echo "Error: Module failed to load"; \
		sudo dmesg | tail -n 20; \
	fi

# Unload module
.PHONY: unload
unload:
	@if sudo lsmod | grep -q $(MODULE_NAME); then \
		sudo rmmod $(MODULE_NAME); \
		echo "Module unloaded"; \
	else \
		echo "Module not loaded"; \
	fi

# Test application targets
test_app: tests/test_app.c
	@echo "Building test application"
	gcc -Wall -Wextra -o tests/test_app tests/test_app.c

# Run all tests
.PHONY: test
test: test_app
	@echo "Running LLM Orchestrator tests..."
	tests/test_app

# Debug info target
.PHONY: debug
debug:
	@echo "Module info:"
	@echo "  Name: $(MODULE_NAME)"
	@echo "  Objects: $($(MODULE_NAME)-objs)"
	@echo "  Kernel Version: $(shell uname -r)"
	@echo "  Kernel Version Code: $(KERNEL_VERSION_CODE)"
	@echo "  kTLS Support: $(shell test $(KERNEL_VERSION_CODE) -ge $(KTLS_MIN_VERSION) && echo Yes || echo No)"
	@echo "  Build Directory: $(KERNELDIR)"
	@echo "  Current Directory: $(PWD)"
	@echo "  CFLAGS: $(ccflags-y)"

# Documentation generation target - using Doxygen
.PHONY: doc
doc:
	@if ! command -v doxygen > /dev/null; then \
		echo "Error: Doxygen not found. Please install it to generate documentation."; \
		echo "  Ubuntu: sudo apt-get install doxygen"; \
		echo "  CentOS/RHEL: sudo yum install doxygen"; \
		echo "  macOS: brew install doxygen"; \
		exit 1; \
	fi
	@if [ ! -f Doxyfile ]; then \
		echo "Creating default Doxyfile..."; \
		doxygen -g; \
		sed -i 's/PROJECT_NAME.*=.*/PROJECT_NAME = "LLM Orchestrator"/' Doxyfile; \
		sed -i 's/PROJECT_BRIEF.*=.*/PROJECT_BRIEF = "Linux Kernel LLM API Orchestration System"/' Doxyfile; \
		sed -i 's/OUTPUT_DIRECTORY.*=.*/OUTPUT_DIRECTORY = doc/' Doxyfile; \
		sed -i 's/EXTRACT_ALL.*=.*/EXTRACT_ALL = YES/' Doxyfile; \
		sed -i 's/EXTRACT_PRIVATE.*=.*/EXTRACT_PRIVATE = YES/' Doxyfile; \
		sed -i 's/EXTRACT_STATIC.*=.*/EXTRACT_STATIC = YES/' Doxyfile; \
	fi
	doxygen Doxyfile
	@echo "Documentation generated in doc/ directory"

# Code safety checks
.PHONY: safety
safety:
	@echo "Running static code analysis..."
	@echo "Checking for sparse..."
	@if ! command -v sparse > /dev/null; then \
		echo "Warning: sparse not installed. Skipping sparse analysis."; \
		echo "  Ubuntu: sudo apt-get install sparse"; \
		echo "  CentOS/RHEL: sudo yum install sparse"; \
	else \
		echo "Running sparse on all C files..."; \
		sparse -Werror $(ccflags-y) *.c; \
	fi

	@echo "Checking for cppcheck..."
	@if ! command -v cppcheck > /dev/null; then \
		echo "Warning: cppcheck not installed. Skipping cppcheck analysis."; \
		echo "  Ubuntu: sudo apt-get install cppcheck"; \
		echo "  CentOS/RHEL: sudo yum install cppcheck"; \
		echo "  macOS: brew install cppcheck"; \
	else \
		echo "Running cppcheck on all files..."; \
		cppcheck --enable=all --force --error-exitcode=1 *.c *.h; \
	fi

	@echo "Running kernel checkpatch.pl script if available..."
	@if [ -f "$(KERNELDIR)/scripts/checkpatch.pl" ]; then \
		$(KERNELDIR)/scripts/checkpatch.pl --no-tree -f *.c *.h; \
	else \
		echo "Warning: checkpatch.pl not found. Skipping kernel style check."; \
	fi

# Check kernel log for this module
.PHONY: log
log:
	@if lsmod | grep -q $(MODULE_NAME); then \
		echo "Module is loaded. Showing recent kernel logs:"; \
		dmesg | grep -i "$(MODULE_NAME)" | tail -n 20; \
	else \
		echo "Module is not loaded."; \
	fi

# Check for memory leaks (to be run after insmod)
.PHONY: check-leaks
check-leaks:
	@echo "Checking for potential memory leaks..."
	@if lsmod | grep -q $(MODULE_NAME); then \
		if [ -f "/sys/class/$(MODULE_NAME)/$(MODULE_NAME)/memory_stats" ]; then \
			cat /sys/class/$(MODULE_NAME)/$(MODULE_NAME)/memory_stats; \
		else \
			echo "Memory stats sysfs file not available"; \
			echo "Looking for memory information in dmesg:"; \
			dmesg | grep -i "$(MODULE_NAME)" | grep -i "memory" | tail -n 10; \
		fi; \
	else \
		echo "Module not loaded"; \
	fi

# Get module version information
.PHONY: version
version:
	@echo "LLM Orchestrator Kernel Module Version: $(shell grep 'MODULE_VERSION' orchestrator_main.c | cut -d '"' -f 2)"
	@echo "Compiled for kernel: $(shell uname -r)"

# Check system requirements
.PHONY: check-system
check-system:
	@echo "Checking system requirements..."
	@echo "Kernel version: $(shell uname -r)"
	@if [ ! -d "$(KERNELDIR)" ]; then \
		echo "Error: Kernel build directory not found at $(KERNELDIR)"; \
		echo "Install kernel headers: sudo apt-get install linux-headers-$(shell uname -r)"; \
		exit 1; \
	else \
		echo "Kernel build directory: OK"; \
	fi

	@echo "Checking for required tools..."
	@for tool in gcc make sparse cppcheck doxygen; do \
		if command -v $$tool > /dev/null; then \
			echo "  $$tool: OK ($(shell $$tool --version | head -n 1))"; \
		else \
			echo "  $$tool: NOT FOUND"; \
		fi; \
	done

	@echo "Checking for TLS support..."
	@if [ $(KERNEL_VERSION_CODE) -ge $(KTLS_MIN_VERSION) ]; then \
		echo "  kTLS support: YES (kernel $(shell uname -r))"; \
	else \
		echo "  kTLS support: NO (kernel $(shell uname -r))"; \
		echo "  Will use software TLS fallback"; \
	fi

# Help target
.PHONY: help
help:
	@echo "LLM Orchestrator Kernel Module - Build Targets"
	@echo "---------------------------------------------"
	@echo "all         : Build kernel module"
	@echo "clean       : Clean built files"
	@echo "install     : Install module to kernel modules directory"
	@echo "uninstall   : Remove module from kernel modules directory"
	@echo "load        : Load module with API keys (requires env vars)"
	@echo "unload      : Unload module"
	@echo "test        : Build and run test application"
	@echo "doc         : Generate documentation with Doxygen"
	@echo "safety      : Run static code analyzers"
	@echo "version     : Display module version"
	@echo "debug       : Show build configuration debug info"
	@echo "check-leaks : Check for potential memory leaks"
	@echo "log         : View recent module logs from kernel"
	@echo "check-system: Check system requirements"
	@echo ""
	@echo "Example usage:"
	@echo "  make                     # Build module"
	@echo "  make check-system        # Check if system meets requirements"
	@echo "  OPENAI_KEY=sk-xxx ANTHROPIC_KEY=sk-ant-xxx GEMINI_KEY=xxxx make load  # Load module"
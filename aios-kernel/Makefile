# Default values for parameters (can be overridden at the command line)
# Default values for parameters (can be overridden at the command line)
API_KEY ?= your-api-key-here
PROVIDER ?= openai
MODEL ?= gpt-4

# Module name and object files
obj-m := llm_module.o
llm_module-y := llm_module_main.o

# Kernel source directory
KERNELDIR ?= /lib/modules/$(shell uname -r)/build

# Current directory
PWD := $(shell pwd)

# Compiler flags
ccflags-y := -Wall

.PHONY: all clean install load unload reload test logs help

# Default target: build the module
all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

# Clean built files
clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
	rm -f *.o *~ core .depend .*.cmd *.ko *.mod.c
	rm -f Module.symvers Module.markers modules.order
	rm -rf .tmp_versions

# Install the module
install:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install
	depmod -a

# Load module with specified API_KEY, PROVIDER, and MODEL
load:
	insmod llm_module.ko api_key="$(API_KEY)" provider="$(PROVIDER)" model="$(MODEL)"

# Unload module
unload:
	rmmod llm_module 2>/dev/null || true

# Show last 20 lines of kernel messages
logs:
	dmesg | tail -n 20

# Rebuild and reload the module
reload: unload clean all load logs

# Test: write a prompt to /dev/llm and then read the response
test:
	echo "What is Linux?" > /dev/llm
	cat /dev/llm

# Help target to describe available commands
help:
	@echo "Available targets:"
	@echo "  all       - Build the module"
	@echo "  clean     - Clean build files"
	@echo "  install   - Install the module into the kernel modules directory"
	@echo "  load      - Insert the module with parameters (requires root)."
	@echo "             Example: make load API_KEY=mykey PROVIDER=openai MODEL=gpt-4"
	@echo "  unload    - Remove the module (requires root)"
	@echo "  reload    - Rebuild and reload the module"
	@echo "  logs      - Show the last 20 lines of dmesg logs"
	@echo "  test      - Test by writing to /dev/llm and reading back"
	@echo "  help      - Show this help message"

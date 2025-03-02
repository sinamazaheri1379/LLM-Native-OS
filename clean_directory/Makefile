# Makefile for LLM Orchestrator kernel module
obj-m += llm_orchestrator.o
llm_orchestrator-objs := llm_context.o llm_scheduler.o

# Kernel build directory
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# User space test program
CC = gcc
CFLAGS = -Wall -Wextra -Werror -O2

# Debug flags
ifdef DEBUG
CFLAGS += -DDEBUG
ccflags-y := -DDEBUG
endif

all: module

module:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
	rm -f test

# Rule for test program
test_program: test.c
	$(CC) $(CFLAGS) -o test test.c

install: module
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install
	depmod -a
	@echo "To load the module with API keys, run:"
	@echo "  modprobe llm_orchestrator openai_api_key=<key> anthropic_api_key=<key> google_gemini_api_key=<key>"

uninstall:
	rm -f /lib/modules/$(shell uname -r)/extra/llm_orchestrator.ko
	depmod -a

reload: unload load

load:
	insmod llm_orchestrator.ko $(API_KEYS)

unload:
	rmmod llm_orchestrator || true

# Generate documentation
docs:
	mkdir -p docs
	kerneldoc -html -o docs/ *.c *.h

.PHONY: all module clean test_program install uninstall reload load unload docs
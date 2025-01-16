# Check for OpenAI API Key
ifeq ($(strip $(OPENAI_API_KEY)),)
$(error OPENAI_API_KEY environment variable is not set)
endif

# Modules to build
obj-m += llm-native.o
obj-m += test.o

# Source files for llm-native module
llm-native-objs := \
    src/access_manager.o \
    src/context_manager.o \
    src/llm_main.o \
    src/llm_providers.o \
    src/memory_manager.o \
    src/scheduler.o \
    src/storage_manager.o \
    src/tool_manager.o

# Source files for test module
test-objs := test/src/test.o

# Include directories
ccflags-y := -I$(src)/include
ccflags-y += -I$(src)/test/include

# Get kernel build path
KDIR ?= /lib/modules/$(shell uname -r)/build

# Compiler flags
ccflags-y += -Wall -Werror -g
ccflags-y += -DDEBUG

# Module parameters
MODULE_PARAMS := api_key=$(OPENAI_API_KEY)

all:
	make -C $(KDIR) M=$(PWD) modules

load: all
	sudo insmod llm-native.ko $(MODULE_PARAMS)
	sudo insmod test.ko

unload:
	sudo rmmod test
	sudo rmmod llm-native

clean:
	make -C $(KDIR) M=$(PWD) clean

install:
	make -C $(KDIR) M=$(PWD) modules_install
	depmod -a

debug: ccflags-y += -DDEBUG_EXTRA
debug: all

.PHONY: all clean install debug load unload
//
// Created by sina-mazaheri on 12/17/24.
//

#ifndef LLM_NATIVE_AIOS_TOOL_MANAGER_H
#define LLM_NATIVE_AIOS_TOOL_MANAGER_H

#include <linux/types.h>
#include <linux/atomic.h>
#include <linux/mutex.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/net.h>
#include <linux/in.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
/**
 * struct llm_tool_param - Tool parameter definition
 */
#define MAX_TOOL_NAME          64
#define MAX_TOOL_DESC          256



struct llm_tool_param {
    char name[MAX_TOOL_NAME];
    char description[MAX_TOOL_DESC];
    bool required;
    struct list_head list;
};

/**
 * struct llm_tool - Function calling definition
 */
struct llm_tool {
    char name[MAX_TOOL_NAME];
    char description[MAX_TOOL_DESC];
    struct list_head parameters;
    struct list_head list;
};

/**
 * struct llm_tool_call - Function call result
 */
struct llm_tool_call {
    char id[64];
    char name[MAX_TOOL_NAME];
    char arguments[MAX_FUNCTION_ARGS];
    struct list_head list;
};

#endif //LLM_NATIVE_AIOS_TOOL_MANAGER_H

//
// Created by sina-mazaheri on 1/16/25.
//

#ifndef TEST_H
#define TEST_H

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include "llm_providers.h"

/* Test function declarations */
int test_send_message(const char *content);
int test_receive_response(void);

/* Test results structure */
struct test_result {
    int status;
    char message[256];
};

/* Test configuration */
struct test_config {
    struct llm_config *llm_config;
    bool initialized;
};

#endif /* TEST_H */

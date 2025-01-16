//
// Created by sina-mazaheri on 1/16/25.
//

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include "test.h"
#include "llm_providers.h"

static struct test_config test_cfg = {0};

/* Test sending a message to the LLM provider */
int test_send_message(const char *content)
{
    struct llm_message *msg;
    int ret;

    if (!test_cfg.initialized) {
        pr_err("Test module not initialized\n");
        return -EINVAL;
    }

    msg = llm_message_alloc(ROLE_USER, content);
    if (!msg) {
        pr_err("Failed to allocate message\n");
        return -ENOMEM;
    }

    ret = llm_send_message(test_cfg.llm_config, msg);
    if (ret < 0) {
        pr_err("Failed to send message: %d\n", ret);
        llm_message_free(msg);
        return ret;
    }

    llm_message_free(msg);
    return 0;
}

/* Test receiving a response from the LLM provider */
int test_receive_response(void)
{
    struct llm_response resp = {0};
    int ret;

    if (!test_cfg.initialized) {
        pr_err("Test module not initialized\n");
        return -EINVAL;
    }

    ret = llm_receive_response(test_cfg.llm_config, &resp);
    if (ret < 0) {
        pr_err("Failed to receive response: %d\n", ret);
        return ret;
    }

    if (resp.message && resp.message->content) {
        pr_info("Received response: %s\n", resp.message->content);
    }

    return 0;
}

/* Example test function */
static int run_basic_test(void)
{
    int ret;
    const char *test_message = "Hello, this is a test message.";

    pr_info("Running basic LLM provider test\n");

    ret = test_send_message(test_message);
    if (ret < 0) {
        pr_err("Send message test failed: %d\n", ret);
        return ret;
    }

    ret = test_receive_response();
    if (ret < 0) {
        pr_err("Receive response test failed: %d\n", ret);
        return ret;
    }

    pr_info("Basic test completed successfully\n");
    return 0;
}

static int __init test_init(void)
{
    int ret;

    pr_info("Initializing LLM provider test module\n");

    /* Get global llm config */
    test_cfg.llm_config = get_global_config();
    if (!test_cfg.llm_config) {
        pr_err("Failed to get LLM configuration\n");
        return -ENODEV;
    }

    test_cfg.initialized = true;

    /* Run basic test */
    ret = run_basic_test();
    if (ret < 0) {
        pr_err("Basic test failed: %d\n", ret);
        return ret;
    }

    return 0;
}

static void __exit test_exit(void)
{
    pr_info("Cleaning up LLM provider test module\n");
    test_cfg.initialized = false;
    test_cfg.llm_config = NULL;
}

module_init(test_init);
module_exit(test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Test module for LLM Provider");


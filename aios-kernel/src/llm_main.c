#include <linux/module.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include "llm_providers.h"
#include "llm_rate_limiter.h"
#include "llm_conn_pool.h"
#include "llm_queue.h"

/* Module information */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Sina Mazaheri");
MODULE_DESCRIPTION("LLM Provider Kernel Module");
MODULE_VERSION("1.0");

/* Global variables */
static struct llm_config* current_config;
static struct rate_limiter* global_limiter;
static struct conn_pool* global_pool;
static struct request_queue* global_queue;
static DEFINE_MUTEX(llm_mutex);

/* Module parameters */
static char *api_key = "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";
module_param(api_key, charp, 0660);
MODULE_PARM_DESC(api_key, "API key for LLM provider");

static char *provider = "openai";
module_param(provider, charp, 0660);
MODULE_PARM_DESC(provider, "LLM provider (openai, anthropic, etc.)");

static char *model = "chatgpt-4o-latest";
module_param(model, charp, 0660);
MODULE_PARM_DESC(model, "Model name for the LLM provider");

/* Helper function to determine provider enum from string */
static enum llm_provider get_provider_from_string(const char *provider_str)
{
    if (!provider_str)
        return LLM_OPENAI;  // Default provider

    if (strcmp(provider_str, "openai") == 0)
        return LLM_OPENAI;
    else if (strcmp(provider_str, "anthropic") == 0)
        return LLM_ANTHROPIC;
    else if (strcmp(provider_str, "mistral") == 0)
        return LLM_MISTRAL;
    else if (strcmp(provider_str, "huggingface") == 0)
        return LLM_HUGGINGFACE;
    else if (strcmp(provider_str, "gemini") == 0)
        return LLM_GEMINI;
    return LLM_OPENAI;  // Default if not recognized
}

/* Module initialization */
static int __init llm_init(void)
{
    int ret;

    /* Parameter validation */
    if (!api_key || strcmp(api_key, "default-key") == 0) {
        pr_err("LLM: Valid API key required\n");
        return -EINVAL;
    }

    /* Allocate global config */
    current_config = kmalloc(sizeof(*current_config), GFP_KERNEL);
    if (!current_config) {
        pr_err("LLM: Failed to allocate config\n");
        return -ENOMEM;
    }

    /* Initialize config */
    memset(current_config, 0, sizeof(*current_config));
    strncpy(current_config->api_key, api_key, MAX_API_KEY_LENGTH - 1);
    strncpy(current_config->model, model, MAX_MODEL_NAME - 1);
    current_config->provider = get_provider_from_string(provider);
    current_config->max_tokens = 1000;
    current_config->temperature_X100 = 70;
    current_config->use_ssl = true;
    current_config->timeout_ms = 30000;

    /* Initialize rate limiter */
    global_limiter = kmalloc(sizeof(*global_limiter), GFP_KERNEL);
    if (!global_limiter) {
        ret = -ENOMEM;
        goto err_limiter;
    }

    ret = rate_limiter_init(global_limiter);
    if (ret < 0)
        goto err_limiter_init;

    /* Initialize connection pool */
    global_pool = kmalloc(sizeof(*global_pool), GFP_KERNEL);
    if (!global_pool) {
        ret = -ENOMEM;
        goto err_pool;
    }

    ret = conn_pool_init(global_pool);
    if (ret < 0)
        goto err_pool_init;

    /* Initialize request queue */
    global_queue = kmalloc(sizeof(*global_queue), GFP_KERNEL);
    if (!global_queue) {
        ret = -ENOMEM;
        goto err_queue;
    }

    ret = request_queue_init(global_queue);
    if (ret < 0)
        goto err_queue_init;

    pr_info("LLM: Module initialized with provider %s and model %s\n",
            provider, current_config->model);
    return 0;

    err_queue_init:
    kfree(global_queue);
    err_queue:
    conn_pool_cleanup(global_pool);
    err_pool_init:
    kfree(global_pool);
    err_pool:
    rate_limiter_cleanup(global_limiter);
    err_limiter_init:
    kfree(global_limiter);
    err_limiter:
    kfree(current_config);
    return ret;
}

/* Module cleanup */
static void __exit llm_exit(void)
{
    mutex_lock(&llm_mutex);

    if (global_queue) {
        request_queue_cleanup(global_queue);
        kfree(global_queue);
    }

    if (global_pool) {
        conn_pool_cleanup(global_pool);
        kfree(global_pool);
    }

    if (global_limiter) {
        rate_limiter_cleanup(global_limiter);
        kfree(global_limiter);
    }

    kfree(current_config);

    mutex_unlock(&llm_mutex);
    pr_info("LLM: Module unloaded\n");
}

module_init(llm_init);
module_exit(llm_exit);
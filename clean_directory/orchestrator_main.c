#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/timer.h>
#include <linux/version.h>
#include "orchestrator_main.h"

#define MODULE_NAME "llm_orchestrator"

/* Module parameters */
static char *openai_api_key = NULL;
module_param(openai_api_key, charp, 0600);
MODULE_PARM_DESC(openai_api_key, "OpenAI API Key");

static char *anthropic_api_key = NULL;
module_param(anthropic_api_key, charp, 0600);
MODULE_PARM_DESC(anthropic_api_key, "Anthropic API Key");

static char *google_gemini_api_key = NULL;
module_param(google_gemini_api_key, charp, 0600);
MODULE_PARM_DESC(google_gemini_api_key, "Google Gemini API Key");

static int prune_threshold_mins = 60; /* 1 hour */
module_param(prune_threshold_mins, int, 0600);
MODULE_PARM_DESC(prune_threshold_mins, "Auto-prune threshold for old conversations in minutes (0 to disable)");


static char *secure_api_keys[3] = { NULL, NULL, NULL };
/* Character device globals */
static int major_number;
static struct class *orchestrator_class;
static struct device *orchestrator_device;
static struct cdev orchestrator_cdev;
static DEFINE_MUTEX(orchestrator_mutex);

/* Global state */
static struct scheduler_state global_scheduler;
static struct llm_response global_response;

/* Maintenance timer */
static struct timer_list maintenance_timer;

/* Function prototypes */
static int orchestrator_open(struct inode *inode, struct file *file);
static int orchestrator_release(struct inode *inode, struct file *file);
static ssize_t orchestrator_read(struct file *file, char __user *buf, size_t count, loff_t *offset);
static ssize_t orchestrator_write(struct file *file, const char __user *buf, size_t count, loff_t *offset);

/* Character device operations */
static struct file_operations orchestrator_fops = {
        .owner = THIS_MODULE,
        .open = orchestrator_open,
        .release = orchestrator_release,
        .read = orchestrator_read,
        .write = orchestrator_write,
};


/*
 * Implementation of provider-specific API functions
 * These were mentioned in header files but missing from the implementation
 */
#define MAX_SCHEDULER_REGISTRY 64

struct scheduler_registry_entry {
    pid_t pid;
    struct scheduler_state *state;
    atomic_t in_use;
};

static struct scheduler_registry_entry scheduler_registry[MAX_SCHEDULER_REGISTRY];
static DEFINE_SPINLOCK(scheduler_registry_lock);
static atomic_t registry_initialized = ATOMIC_INIT(0);

/* Initialize the scheduler registry */
static int scheduler_registry_init(void)
{
    int i;

    if (atomic_read(&registry_initialized) != 0)
        return 0;  /* Already initialized */

    spin_lock(&scheduler_registry_lock);

    for (i = 0; i < MAX_SCHEDULER_REGISTRY; i++) {
        scheduler_registry[i].pid = 0;
        scheduler_registry[i].state = NULL;
        atomic_set(&scheduler_registry[i].in_use, 0);
    }

    spin_unlock(&scheduler_registry_lock);
    atomic_set(&registry_initialized, 1);

    pr_info("Scheduler registry initialized\n");
    return 0;
}

/* Clean up the scheduler registry */
static void scheduler_registry_cleanup(void)
{
    int i;

    if (atomic_read(&registry_initialized) == 0)
        return;  /* Not initialized */

    spin_lock(&scheduler_registry_lock);

    for (i = 0; i < MAX_SCHEDULER_REGISTRY; i++) {
        if (atomic_read(&scheduler_registry[i].in_use) != 0) {
            scheduler_registry[i].pid = 0;
            scheduler_registry[i].state = NULL;
            atomic_set(&scheduler_registry[i].in_use, 0);
        }
    }

    spin_unlock(&scheduler_registry_lock);
    atomic_set(&registry_initialized, 0);

    pr_info("Scheduler registry cleaned up\n");
}

/* Store scheduler state in registry */
void set_scheduler_state(struct scheduler_state *state)
{
    int i, free_idx = -1;
    pid_t current_pid;

    if (!state || atomic_read(&registry_initialized) == 0) {
        pr_err("set_scheduler_state: Registry not initialized or invalid state\n");
        return;
    }

    current_pid = task_pid_nr(current);

    spin_lock(&scheduler_registry_lock);

    /* Check if entry already exists or find a free slot */
    for (i = 0; i < MAX_SCHEDULER_REGISTRY; i++) {
        if (scheduler_registry[i].pid == current_pid &&
            atomic_read(&scheduler_registry[i].in_use) != 0) {
            /* Update existing entry */
            scheduler_registry[i].state = state;
            spin_unlock(&scheduler_registry_lock);
            return;
        }

        if (free_idx == -1 && atomic_read(&scheduler_registry[i].in_use) == 0)
            free_idx = i;
    }

    /* Add new entry if free slot found */
    if (free_idx != -1) {
        scheduler_registry[free_idx].pid = current_pid;
        scheduler_registry[free_idx].state = state;
        atomic_set(&scheduler_registry[free_idx].in_use, 1);
        spin_unlock(&scheduler_registry_lock);
        return;
    }

    spin_unlock(&scheduler_registry_lock);
    pr_warn("set_scheduler_state: No free slots in registry\n");
}

/* Get scheduler state from registry */
struct scheduler_state *get_scheduler_state(void)
{
    int i;
    pid_t current_pid;
    struct scheduler_state *state = NULL;

    if (atomic_read(&registry_initialized) == 0) {
        pr_err("get_scheduler_state: Registry not initialized\n");
        return NULL;
    }

    current_pid = task_pid_nr(current);

    spin_lock(&scheduler_registry_lock);

    for (i = 0; i < MAX_SCHEDULER_REGISTRY; i++) {
        if (scheduler_registry[i].pid == current_pid &&
            atomic_read(&scheduler_registry[i].in_use) != 0) {
            state = scheduler_registry[i].state;
            break;
        }
    }

    spin_unlock(&scheduler_registry_lock);

    /* If no entry found, use global scheduler as fallback */
    if (!state) {
        state = &global_scheduler;
    }

    return state;
}

/* Remove scheduler state from registry */
void remove_scheduler_state(void)
{
    int i;
    pid_t current_pid;

    if (atomic_read(&registry_initialized) == 0)
        return;

    current_pid = task_pid_nr(current);

    spin_lock(&scheduler_registry_lock);

    for (i = 0; i < MAX_SCHEDULER_REGISTRY; i++) {
        if (scheduler_registry[i].pid == current_pid &&
            atomic_read(&scheduler_registry[i].in_use) != 0) {
            scheduler_registry[i].pid = 0;
            scheduler_registry[i].state = NULL;
            atomic_set(&scheduler_registry[i].in_use, 0);
            break;
        }
    }

    spin_unlock(&scheduler_registry_lock);
}
/* OpenAI API implementation */
int llm_send_openai(const char *api_key, struct llm_request *req, struct llm_response *resp)
{
    char *openai_host = "104.18.6.192";  /* OpenAI API IP - would use DNS in real implementation */
    int port = 443;
    char *path = "/v1/chat/completions";
    struct llm_json_buffer json_buf;
    int ret;
    char auth_header[128];
    const char *model;

    if (!api_key || !req || !resp) {
        pr_err("llm_send_openai: Invalid parameters\n");
        return -EINVAL;
    }

    /* Initialize response */
    memset(resp, 0, sizeof(*resp));
    resp->provider_used = PROVIDER_OPENAI;

    /* Check/use default model if none specified */
    if (req->model_name[0] == '\0') {
        model = get_default_model(PROVIDER_OPENAI);
        strscpy(resp->model_used, model, MAX_MODEL_NAME);
    } else {
        if (!is_model_supported(PROVIDER_OPENAI, req->model_name)) {
            pr_warn("llm_send_openai: Unsupported model %s, using default\n",
                    req->model_name);
            model = get_default_model(PROVIDER_OPENAI);
        } else {
            model = req->model_name;
        }
        strscpy(resp->model_used, model, MAX_MODEL_NAME);
    }

    /* Format auth header */
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s\r\n", api_key);

    /* Initialize JSON buffer */
    ret = json_buffer_init(&json_buf, 4096);
    if (ret) {
        pr_err("llm_send_openai: Failed to initialize JSON buffer: %d\n", ret);
        return ret;
    }

    /* Create OpenAI-specific JSON payload */
    ret = append_json_string(&json_buf, "{\"model\":\"");
    if (ret) goto cleanup;

    ret = append_json_value(&json_buf, model);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "\",\"messages\":[");
    if (ret) goto cleanup;

    /* Add conversation context if available */
    if (req->conversation_id > 0) {
        struct llm_json_buffer context_buf;

        ret = json_buffer_init(&context_buf, 16384);  /* Larger buffer for context */
        if (ret) {
            pr_err("llm_send_openai: Failed to initialize context buffer: %d\n", ret);
            goto cleanup;
        }

        ret = context_get_conversation(req->conversation_id, &context_buf);
        if (ret == 0 && context_buf.used > 2) { /* Has valid context (more than just "[]") */
            /* Add context entries, but skip trailing ']' */
            context_buf.data[context_buf.used - 1] = '\0';
            ret = append_json_string(&json_buf, context_buf.data + 1);
            if (ret) {
                json_buffer_free(&context_buf);
                goto cleanup;
            }

            /* Add comma if we have context */
            ret = append_json_string(&json_buf, ",");
            if (ret) {
                json_buffer_free(&context_buf);
                goto cleanup;
            }
        }

        json_buffer_free(&context_buf);
    }

    /* Add current message */
    ret = append_json_string(&json_buf, "{\"role\":\"");
    if (ret) goto cleanup;

    ret = append_json_value(&json_buf, req->role);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "\",\"content\":\"");
    if (ret) goto cleanup;

    ret = append_json_value(&json_buf, req->prompt);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "\"}");
    if (ret) goto cleanup;

    /* Complete the JSON request */
    ret = append_json_string(&json_buf, "],\"temperature\":");
    if (ret) goto cleanup;

    ret = append_json_float(&json_buf, req->temperature_x100);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, ",\"max_tokens\":");
    if (ret) goto cleanup;

    ret = append_json_number(&json_buf, req->max_tokens);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "}");
    if (ret) goto cleanup;

    /* Send request to OpenAI API */
    ret = network_send_request(openai_host, port, path,
                               NULL, auth_header, true,
                               req->timeout_ms, &json_buf, resp);

    /* Handle network errors */
    if (ret < 0) {
        if (ret == -LLM_ERR_RATE_LIMIT) {
            pr_warn("llm_send_openai: Rate limited, will reset in %lu ms\n",
                    resp->rate_limit_reset_ms);
            /* Update scheduler state to mark OpenAI as rate limited */
            handle_rate_limit(PROVIDER_OPENAI, get_scheduler_state(), resp->rate_limit_reset_ms);
        } else {
            pr_err("llm_send_openai: Request failed: %d\n", ret);
        }
        goto cleanup;
    }

    /* Store prompt in conversation context if needed */
    if (req->conversation_id > 0) {
        context_add_entry(req->conversation_id, req->role, req->prompt);

        /* Also store the response in the context */
        if (resp->content_length > 0) {
            context_add_entry(req->conversation_id, "assistant", resp->content);
        }
    }

    cleanup:
    json_buffer_free(&json_buf);
    return ret;
}

/* Anthropic API implementation */
int llm_send_anthropic(const char *api_key, struct llm_request *req, struct llm_response *resp)
{
    char *anthropic_host = "104.18.6.119";  /* Anthropic API IP - would use DNS in real implementation */
    int port = 443;
    char *path = "/v1/messages";
    struct llm_json_buffer json_buf;
    int ret;
    char auth_header[128];
    const char *model;

    if (!api_key || !req || !resp) {
        pr_err("llm_send_anthropic: Invalid parameters\n");
        return -EINVAL;
    }

    /* Initialize response */
    memset(resp, 0, sizeof(*resp));
    resp->provider_used = PROVIDER_ANTHROPIC;

    /* Check/use default model if none specified */
    if (req->model_name[0] == '\0') {
        model = get_default_model(PROVIDER_ANTHROPIC);
        strscpy(resp->model_used, model, MAX_MODEL_NAME);
    } else {
        if (!is_model_supported(PROVIDER_ANTHROPIC, req->model_name)) {
            pr_warn("llm_send_anthropic: Unsupported model %s, using default\n",
                    req->model_name);
            model = get_default_model(PROVIDER_ANTHROPIC);
        } else {
            model = req->model_name;
        }
        strscpy(resp->model_used, model, MAX_MODEL_NAME);
    }

    /* Format auth header */
    snprintf(auth_header, sizeof(auth_header), "x-api-key: %s\r\nanthropic-version: 2023-06-01\r\n", api_key);

    /* Initialize JSON buffer */
    ret = json_buffer_init(&json_buf, 4096);
    if (ret) {
        pr_err("llm_send_anthropic: Failed to initialize JSON buffer: %d\n", ret);
        return ret;
    }

    /* Create Anthropic-specific JSON payload */
    ret = append_json_string(&json_buf, "{\"model\":\"");
    if (ret) goto cleanup;

    ret = append_json_value(&json_buf, model);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "\",\"messages\":[");
    if (ret) goto cleanup;

    /* Add conversation context if available */
    if (req->conversation_id > 0) {
        struct llm_json_buffer context_buf;

        ret = json_buffer_init(&context_buf, 16384);  /* Larger buffer for context */
        if (ret) {
            pr_err("llm_send_anthropic: Failed to initialize context buffer: %d\n", ret);
            goto cleanup;
        }

        ret = context_get_conversation(req->conversation_id, &context_buf);
        if (ret == 0 && context_buf.used > 2) { /* Has valid context (more than just "[]") */
            /* Add context entries, but skip trailing ']' */
            context_buf.data[context_buf.used - 1] = '\0';
            ret = append_json_string(&json_buf, context_buf.data + 1);
            if (ret) {
                json_buffer_free(&context_buf);
                goto cleanup;
            }

            /* Add comma if we have context */
            ret = append_json_string(&json_buf, ",");
            if (ret) {
                json_buffer_free(&context_buf);
                goto cleanup;
            }
        }

        json_buffer_free(&context_buf);
    }

    /* Add current message */
    ret = append_json_string(&json_buf, "{\"role\":\"");
    if (ret) goto cleanup;

    ret = append_json_value(&json_buf, req->role);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "\",\"content\":\"");
    if (ret) goto cleanup;

    ret = append_json_value(&json_buf, req->prompt);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "\"}");
    if (ret) goto cleanup;

    /* Complete the JSON request */
    ret = append_json_string(&json_buf, "],\"max_tokens\":");
    if (ret) goto cleanup;

    ret = append_json_number(&json_buf, req->max_tokens);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, ",\"temperature\":");
    if (ret) goto cleanup;

    ret = append_json_float(&json_buf, req->temperature_x100);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "}");
    if (ret) goto cleanup;

    /* Send request to Anthropic API */
    ret = network_send_request(anthropic_host, port, path,
                               NULL, auth_header, true,
                               req->timeout_ms, &json_buf, resp);

    /* Handle network errors */
    if (ret < 0) {
        if (ret == -LLM_ERR_RATE_LIMIT) {
            pr_warn("llm_send_anthropic: Rate limited, will reset in %lu ms\n",
                    resp->rate_limit_reset_ms);
            /* Update scheduler state to mark Anthropic as rate limited */
            handle_rate_limit(PROVIDER_ANTHROPIC, get_scheduler_state(), resp->rate_limit_reset_ms);
        } else {
            pr_err("llm_send_anthropic: Request failed: %d\n", ret);
        }
        goto cleanup;
    }

    /* Store prompt in conversation context if needed */
    if (req->conversation_id > 0) {
        context_add_entry(req->conversation_id, req->role, req->prompt);

        /* Also store the response in the context */
        if (resp->content_length > 0) {
            context_add_entry(req->conversation_id, "assistant", resp->content);
        }
    }

    cleanup:
    json_buffer_free(&json_buf);
    return ret;
}

/* Google Gemini API implementation */
int llm_send_google_gemini(const char *api_key, struct llm_request *req, struct llm_response *resp)
{
    char *gemini_host = "104.18.6.14";  /* Google Gemini API IP - would use DNS in real implementation */
    struct llm_json_buffer json_buf;
    int port = 443;
    char *path = "/v1/models/gemini-pro:generateContent";  /* Path will include API key */
    int ret;
    char auth_path[256];
    const char *model;

    if (!api_key || !req || !resp) {
        pr_err("llm_send_google_gemini: Invalid parameters\n");
        return -EINVAL;
    }

    /* Initialize response */
    memset(resp, 0, sizeof(*resp));
    resp->provider_used = PROVIDER_GOOGLE_GEMINI;

    /* Check/use default model if none specified */
    if (req->model_name[0] == '\0') {
        model = get_default_model(PROVIDER_GOOGLE_GEMINI);
        strscpy(resp->model_used, model, MAX_MODEL_NAME);
    } else {
        if (!is_model_supported(PROVIDER_GOOGLE_GEMINI, req->model_name)) {
            pr_warn("llm_send_google_gemini: Unsupported model %s, using default\n",
                    req->model_name);
            model = get_default_model(PROVIDER_GOOGLE_GEMINI);
        } else {
            model = req->model_name;
        }
        strscpy(resp->model_used, model, MAX_MODEL_NAME);
    }

    /* Format path with API key */
    snprintf(auth_path, sizeof(auth_path), "%s?key=%s", path, api_key);

    /* Initialize JSON buffer */
    ret = json_buffer_init(&json_buf, 4096);
    if (ret) {
        pr_err("llm_send_google_gemini: Failed to initialize JSON buffer: %d\n", ret);
        return ret;
    }

    /* Create Google Gemini-specific JSON payload */
    ret = append_json_string(&json_buf, "{\"contents\":[");
    if (ret) goto cleanup;

    /* Add conversation context if available - Gemini has a different format */
    if (req->conversation_id > 0) {
        struct context_entry *entry;
        unsigned long flags;
        struct conversation_context *ctx;

        /* Find the conversation */
        spin_lock_irqsave(&conversations_lock, flags);
        ctx = find_conversation(req->conversation_id);
        if (!ctx) {
            spin_unlock_irqrestore(&conversations_lock, flags);
            pr_debug("llm_send_google_gemini: Conversation %d not found\n", req->conversation_id);
        } else {
            spin_unlock_irqrestore(&conversations_lock, flags);

            /* Manually format Gemini-compatible conversation history */
            spin_lock_irqsave(&ctx->lock, flags);
            list_for_each_entry(entry, &ctx->entries, list) {
                ret = append_json_string(&json_buf, "{\"role\":\"");
                if (ret) {
                    spin_unlock_irqrestore(&ctx->lock, flags);
                    goto cleanup;
                }

                /* Convert "user" to "user", "assistant" to "model" for Gemini */
                if (strcmp(entry->role, "assistant") == 0) {
                    ret = append_json_string(&json_buf, "model");
                } else {
                    ret = append_json_value(&json_buf, entry->role);
                }

                if (ret) {
                    spin_unlock_irqrestore(&ctx->lock, flags);
                    goto cleanup;
                }

                ret = append_json_string(&json_buf, "\",\"parts\":[{\"text\":\"");
                if (ret) {
                    spin_unlock_irqrestore(&ctx->lock, flags);
                    goto cleanup;
                }

                ret = append_json_value(&json_buf, entry->content);
                if (ret) {
                    spin_unlock_irqrestore(&ctx->lock, flags);
                    goto cleanup;
                }

                ret = append_json_string(&json_buf, "\"}]},");
                if (ret) {
                    spin_unlock_irqrestore(&ctx->lock, flags);
                    goto cleanup;
                }
            }
            spin_unlock_irqrestore(&ctx->lock, flags);
        }
    }

    /* Add current message - Gemini format */
    ret = append_json_string(&json_buf, "{\"role\":\"user\",\"parts\":[{\"text\":\"");
    if (ret) goto cleanup;

    ret = append_json_value(&json_buf, req->prompt);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "\"}]}],");
    if (ret) goto cleanup;

    /* Add generation parameters */
    ret = append_json_string(&json_buf, "\"generationConfig\":{");
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "\"temperature\":");
    if (ret) goto cleanup;

    ret = append_json_float(&json_buf, req->temperature_x100);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, ",\"maxOutputTokens\":");
    if (ret) goto cleanup;

    ret = append_json_number(&json_buf, req->max_tokens);
    if (ret) goto cleanup;

    ret = append_json_string(&json_buf, "}}");
    if (ret) goto cleanup;

    /* Send request to Google Gemini API */
    ret = network_send_request(gemini_host, port, auth_path,
                               NULL, NULL, true,
                               req->timeout_ms, &json_buf, resp);

    /* Handle network errors */
    if (ret < 0) {
        if (ret == -LLM_ERR_RATE_LIMIT) {
            pr_warn("llm_send_google_gemini: Rate limited, will reset in %lu ms\n",
                    resp->rate_limit_reset_ms);
            /* Update scheduler state to mark Gemini as rate limited */
            handle_rate_limit(PROVIDER_GOOGLE_GEMINI, get_scheduler_state(), resp->rate_limit_reset_ms);
        } else {
            pr_err("llm_send_google_gemini: Request failed: %d\n", ret);
        }
        goto cleanup;
    }

    /* Store prompt in conversation context if needed */
    if (req->conversation_id > 0) {
        context_add_entry(req->conversation_id, req->role, req->prompt);

        /* Also store the response in the context */
        if (resp->content_length > 0) {
            context_add_entry(req->conversation_id, "assistant", resp->content);
        }
    }

    cleanup:
    json_buffer_free(&json_buf);
    return ret;
}


/* Function to safely set API keys */
static int set_api_key(int provider, const char *key)
{
    size_t key_len;
    char *secure_key;

    if (!key || provider < 0 || provider >= 3)
        return -EINVAL;

    key_len = strlen(key);
    if (key_len < 8 || key_len > 256)
        return -EINVAL;  /* Validate reasonable key length */

    /* Allocate secure memory */
    secure_key = kzalloc(key_len + 1, GFP_KERNEL);
    if (!secure_key)
        return -ENOMEM;

    /* Copy key */
    strscpy(secure_key, key, key_len + 1);

    /* Free old key if it exists */
    if (secure_api_keys[provider]) {
        memzero_explicit(secure_api_keys[provider], strlen(secure_api_keys[provider]));
        kfree(secure_api_keys[provider]);
    }

    secure_api_keys[provider] = secure_key;
    return 0;
}

/* Function to safely get API key without exposing it */
//static const char *get_api_key(int provider)
//{
//    if (provider < 0 || provider >= 3)
//        return NULL;
//
//    return secure_api_keys[provider];
//}

/* Clear all API keys securely */
static void clear_all_api_keys(void)
{
    int i;

    for (i = 0; i < 3; i++) {
        if (secure_api_keys[i]) {
            memzero_explicit(secure_api_keys[i], strlen(secure_api_keys[i]));
            kfree(secure_api_keys[i]);
            secure_api_keys[i] = NULL;
        }
    }
}

/* Main request orchestration function */
static int orchestrate_request(struct llm_request *req, struct llm_response *resp)
{
    int selected_provider, ret = -EINVAL;
    int i;

    /* Set default values if not provided */
    req->timeout_ms = (req->timeout_ms > 0) ? req->timeout_ms : 30000;
    if (req->temperature_x100 <= 0)
        req->temperature_x100 = 70;

    /* Initialize response */
    memset(resp, 0, sizeof(*resp));
    resp->timestamp = ktime_get();

    /* Select provider based on scheduling algorithm */
    selected_provider = select_provider(req, &global_scheduler);
    if (selected_provider < 0)
        return selected_provider;

    resp->provider_used = selected_provider;

    /* Special handling for fallback algorithm */
    if (req->scheduler_algorithm == SCHEDULER_FALLBACK) {
        /* Try each provider in sequence */
        ret = llm_send_openai(openai_api_key, req, resp);
        update_provider_metrics(PROVIDER_OPENAI, ret, resp->latency_ms, resp->tokens_used);
        if (ret == 0)
            return ret;

        pr_warn("orchestrate_request: OpenAI failed (ret=%d), trying Anthropic\n", ret);
        ret = llm_send_anthropic(anthropic_api_key, req, resp);
        update_provider_metrics(PROVIDER_ANTHROPIC, ret, resp->latency_ms, resp->tokens_used);
        if (ret == 0)
            return ret;

        pr_warn("orchestrate_request: Anthropic failed (ret=%d), trying Gemini\n", ret);
        ret = llm_send_google_gemini(google_gemini_api_key, req, resp);
        update_provider_metrics(PROVIDER_GOOGLE_GEMINI, ret, resp->latency_ms, resp->tokens_used);
        return ret;
    }

    /* Try the selected provider and fall back if needed */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        int provider = (selected_provider + i) % PROVIDER_COUNT;
        resp->provider_used = provider;

        switch (provider) {
            case PROVIDER_OPENAI:
                ret = llm_send_openai(openai_api_key, req, resp);
                break;
            case PROVIDER_ANTHROPIC:
                ret = llm_send_anthropic(anthropic_api_key, req, resp);
                break;
            case PROVIDER_GOOGLE_GEMINI:
                ret = llm_send_google_gemini(google_gemini_api_key, req, resp);
                break;
            default:
                ret = -EINVAL;
                break;
        }

        update_provider_metrics(provider, ret, resp->latency_ms, resp->tokens_used);

        if (ret == 0)
            break;

        if (i < PROVIDER_COUNT - 1) {
            int next_provider = (selected_provider + i + 1) % PROVIDER_COUNT;
            pr_warn("orchestrate_request: Provider %d failed (ret=%d), trying fallback %d\n",
                    provider, ret, next_provider);
        }
    }

    return ret;
}

/* Maintenance timer callback */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 15, 0)
static void maintenance_timer_callback(struct timer_list *t)
#else
static void maintenance_timer_callback(unsigned long data)
#endif
{
    if (prune_threshold_mins > 0)
        context_prune_old_conversations(prune_threshold_mins * 60 * 1000);

    mod_timer(&maintenance_timer, jiffies + HZ * 60 * 10); /* Run every 10 minutes */
}

/* Update orchestrator_open function */
static int orchestrator_open(struct inode *inode, struct file *file)
{
    /* Initialize registry if needed */
    if (atomic_read(&registry_initialized) == 0)
        scheduler_registry_init();

    /* Store scheduler state in registry */
    set_scheduler_state(&global_scheduler);
    return 0;
}

/* Update orchestrator_release function */
static int orchestrator_release(struct inode *inode, struct file *file)
{
    /* Remove from registry */
    remove_scheduler_state();
    return 0;
}

static ssize_t orchestrator_write(struct file *file, const char __user *buf, size_t count, loff_t *offset)
{
    struct llm_request* req;
    int ret;

    if (count != sizeof(struct llm_request)){
        return -EINVAL;
    }
	req = kmalloc(sizeof(struct llm_request), GFP_KERNEL);
    if (!req){
        return -ENOMEM;
    }
    if (copy_from_user(req, buf, sizeof(*req))) {
        kfree(req);
        return -EFAULT;
    }

    /* Comprehensive input validation */
    if (req->conversation_id <= 0) {
        pr_err("orchestrator_write: Invalid conversation_id: %d\n", req->conversation_id);
        return -EINVAL;
    }

    /* Validate prompt - ensure it's not empty and is null-terminated */
    if (req->prompt[0] == '\0') {
        pr_err("orchestrator_write: Empty prompt\n");
        return -EINVAL;
    }
    req->prompt[MAX_PROMPT_LENGTH - 1] = '\0';

    /* Validate role - use default if empty */
    if (req->role[0] == '\0') {
        strscpy(req->role, "user", MAX_ROLE_NAME);
    } else {
        req->role[MAX_ROLE_NAME - 1] = '\0';
    }

    /* Validate model name - if empty, we'll use a default in the provider code */
    req->model_name[MAX_MODEL_NAME - 1] = '\0';

    /* Validate and set reasonable defaults for numeric parameters */
    if (req->max_tokens <= 0 || req->max_tokens > 32000) {
        req->max_tokens = 4000; /* Reasonable default */
    }

    if (req->temperature_x100 <= 0 || req->temperature_x100 > 200) {
        req->temperature_x100 = 70; /* Default to 0.7 */
    }

    if (req->timeout_ms <= 0) {
        req->timeout_ms = 30000; /* 30 seconds default */
    } else if (req->timeout_ms > 300000) {
        req->timeout_ms = 300000; /* Cap at 5 minutes */
    }

    /* Validate scheduler algorithm */
    if (req->scheduler_algorithm != -1 && (req->scheduler_algorithm < 0 || req->scheduler_algorithm > SCHEDULER_MAX_ALGORITHM)) {
        pr_warn("orchestrator_write: Invalid scheduler algorithm: %d, using default\n",
        req->scheduler_algorithm);
        req->scheduler_algorithm = -1; /* Use default from state */
    }

    /* Validate priority */
    if (req->priority < 0) {
        req->priority = 0;
    } else if (req->priority > 100) {
        req->priority = 100;
    }

    mutex_lock(&orchestrator_mutex);
    ret = orchestrate_request(req, &global_response);
    mutex_unlock(&orchestrator_mutex);
    kfree(req);
    return ret < 0 ? ret : count;
}


static ssize_t orchestrator_read(struct file *file, char __user *buf, size_t count, loff_t *offset)
{
    ssize_t ret;

    mutex_lock(&orchestrator_mutex);

    if (global_response.content_length == 0) {
        mutex_unlock(&orchestrator_mutex);
        return 0;
    }

    if (count < global_response.content_length) {
        mutex_unlock(&orchestrator_mutex);
        return -EINVAL;
    }

    if (copy_to_user(buf, global_response.content, global_response.content_length)) {
        mutex_unlock(&orchestrator_mutex);
        return -EFAULT;
    }

    ret = global_response.content_length;
    global_response.content_length = 0;
    mutex_unlock(&orchestrator_mutex);
    return ret;
}
static ssize_t scheduler_algorithm_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    int algorithm = atomic_read(&global_scheduler.current_algorithm);
    const char *algorithm_name;

    switch (algorithm) {
        case SCHEDULER_ROUND_ROBIN:
            algorithm_name = "Round Robin";
            break;
        case SCHEDULER_WEIGHTED:
            algorithm_name = "Weighted";
            break;
        case SCHEDULER_PRIORITY:
            algorithm_name = "Priority";
            break;
        case SCHEDULER_PERFORMANCE:
            algorithm_name = "Performance";
            break;
        case SCHEDULER_COST_AWARE:
            algorithm_name = "Cost Aware";
            break;
        case SCHEDULER_FALLBACK:
            algorithm_name = "Fallback";
            break;
        case SCHEDULER_FIFO:
            algorithm_name = "FIFO";
            break;
        default:
            algorithm_name = "Unknown";
            break;
    }

    return scnprintf(buf, PAGE_SIZE, "Current algorithm: %s (%d)\n", algorithm_name, algorithm);
}

/* Sets the scheduler algorithm */
static ssize_t scheduler_algorithm_store(struct device *dev, struct device_attribute *attr,
                                        const char *buf, size_t count)
{
    int algorithm;
    int ret;

    ret = kstrtoint(buf, 10, &algorithm);
    if (ret < 0)
        return ret;

    if (algorithm < 0 || algorithm > SCHEDULER_MAX_ALGORITHM)
        return -EINVAL;

    atomic_set(&global_scheduler.current_algorithm, algorithm);
    return count;
}

/* Shows provider metrics */
static ssize_t provider_metrics_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    int i;
    ssize_t len = 0;
    const char *provider_names[PROVIDER_COUNT] = {"OpenAI", "Anthropic", "Google Gemini"};

    len += scnprintf(buf + len, PAGE_SIZE - len, "Provider Metrics:\n");
    len += scnprintf(buf + len, PAGE_SIZE - len, "----------------\n");

    for (i = 0; i < PROVIDER_COUNT; i++) {
        int total = atomic_read(&global_scheduler.metrics[i].total_requests);
        int successful = atomic_read(&global_scheduler.metrics[i].successful_requests);
        int failed = atomic_read(&global_scheduler.metrics[i].failed_requests);
        int timeouts = atomic_read(&global_scheduler.metrics[i].timeouts);
        int rate_limited = atomic_read(&global_scheduler.metrics[i].rate_limited);
        int status = atomic_read(&global_scheduler.metrics[i].current_status);
        s64 total_latency = atomic64_read(&global_scheduler.metrics[i].total_latency_ms);
        s64 avg_latency = total < 1 ? 0 : div64_s64(total_latency, max(successful, 1));
        int tokens = atomic_read(&global_scheduler.metrics[i].total_tokens);

        len += scnprintf(buf + len, PAGE_SIZE - len, "\n%s:\n", provider_names[i]);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Status: %s\n",
                        status ? "Available" : "Rate Limited");
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Total Requests: %d\n", total);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Successful: %d\n", successful);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Failed: %d\n", failed);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Timeouts: %d\n", timeouts);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Rate Limited: %d\n", rate_limited);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Success Rate: %d%%\n",
                        total > 0 ? (successful * 100) / total : 0);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Avg Latency: %lld ms\n", avg_latency);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Min Latency: %lu ms\n",
                        global_scheduler.metrics[i].min_latency_ms == ULONG_MAX ? 0 :
                        global_scheduler.metrics[i].min_latency_ms);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Max Latency: %lu ms\n",
                        global_scheduler.metrics[i].max_latency_ms);
        len += scnprintf(buf + len, PAGE_SIZE - len, "  Total Tokens: %d\n", tokens);
    }

    return len;
}

/* Resets provider metrics */
static ssize_t reset_metrics_store(struct device *dev, struct device_attribute *attr,
                                 const char *buf, size_t count)
{
    if (buf[0] == '1' || buf[0] == 'y' || buf[0] == 'Y') {
        scheduler_reset_metrics(&global_scheduler);
        pr_info("LLM Orchestrator: Provider metrics reset\n");
    }
    return count;
}

/* Shows scheduler weights */
static ssize_t scheduler_weights_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    const char *provider_names[PROVIDER_COUNT] = {"OpenAI", "Anthropic", "Google Gemini"};
    int i;
    ssize_t len = 0;
    int total = 0;

    len += scnprintf(buf + len, PAGE_SIZE - len, "Provider Weights:\n");

    for (i = 0; i < PROVIDER_COUNT; i++) {
        len += scnprintf(buf + len, PAGE_SIZE - len, "  %s: %d%%\n",
                       provider_names[i], global_scheduler.weights[i]);
        total += global_scheduler.weights[i];
    }

    len += scnprintf(buf + len, PAGE_SIZE - len, "Total: %d%%\n", total);
    len += scnprintf(buf + len, PAGE_SIZE - len, "Auto-adjust: %s\n",
                    global_scheduler.auto_adjust ? "Enabled" : "Disabled");

    return len;
}

/* Sets scheduler weights */
static ssize_t scheduler_weights_store(struct device *dev, struct device_attribute *attr,
                                     const char *buf, size_t count)
{
    int w1, w2, w3;
    int ret;

    /* Format: "w1,w2,w3" for the three providers */
    ret = sscanf(buf, "%d,%d,%d", &w1, &w2, &w3);
    if (ret != 3)
        return -EINVAL;

    /* Validate weights */
    if (w1 < 0 || w2 < 0 || w3 < 0)
        return -EINVAL;
    if (w1 + w2 + w3 != 100)
        return -EINVAL;

    /* Update weights */
    global_scheduler.weights[0] = w1;
    global_scheduler.weights[1] = w2;
    global_scheduler.weights[2] = w3;

    pr_info("LLM Orchestrator: Weights updated to %d%%,%d%%,%d%%\n", w1, w2, w3);
    return count;
}

/* Shows auto-adjust setting */
static ssize_t auto_adjust_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    return scnprintf(buf, PAGE_SIZE, "Auto-adjust: %s\n",
                    global_scheduler.auto_adjust ? "Enabled" : "Disabled");
}

/* Sets auto-adjust setting */
static ssize_t auto_adjust_store(struct device *dev, struct device_attribute *attr,
                               const char *buf, size_t count)
{
    bool enable;

    if (kstrtobool(buf, &enable))
        return -EINVAL;

    global_scheduler.auto_adjust = enable;
    pr_info("LLM Orchestrator: Auto-adjust %s\n", enable ? "enabled" : "disabled");
    return count;
}

/* Shows FIFO queue status */
static ssize_t fifo_status_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    ssize_t len = 0;

    len += scnprintf(buf + len, PAGE_SIZE - len, "FIFO Queue Status:\n");
    len += scnprintf(buf + len, PAGE_SIZE - len, "  Queue Size: %d/%d\n",
                    global_scheduler.fifo.count, MAX_FIFO_QUEUE_SIZE);
    len += scnprintf(buf + len, PAGE_SIZE - len, "  Head: %d\n", global_scheduler.fifo.head);
    len += scnprintf(buf + len, PAGE_SIZE - len, "  Tail: %d\n", global_scheduler.fifo.tail);

    return len;
}

/* Shows context status */
static ssize_t context_status_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    int total_conversations, total_entries, entries_added, entries_pruned;
    context_get_stats(&total_conversations, &total_entries, &entries_added, &entries_pruned);

    return scnprintf(buf, PAGE_SIZE,
                   "Context Status:\n"
                   "  Active Conversations: %d\n"
                   "  Total Entries: %d\n"
                   "  Entries Added: %d\n"
                   "  Entries Pruned: %d\n"
                   "  Auto-prune Threshold: %d minutes\n",
                   total_conversations,
                   total_entries,
                   entries_added,
                   entries_pruned,
                   prune_threshold_mins);
}

/* Sysfs interfaces for configuration and statistics */
static DEVICE_ATTR(scheduler_algorithm, 0644, scheduler_algorithm_show, scheduler_algorithm_store);
static DEVICE_ATTR(provider_metrics, 0444, provider_metrics_show, NULL);
static DEVICE_ATTR(reset_metrics, 0200, NULL, reset_metrics_store);
static DEVICE_ATTR(scheduler_weights, 0644, scheduler_weights_show, scheduler_weights_store);
static DEVICE_ATTR(auto_adjust, 0644, auto_adjust_show, auto_adjust_store);
static DEVICE_ATTR(fifo_status, 0444, fifo_status_show, NULL);
static DEVICE_ATTR(context_status, 0444, context_status_show, NULL);
static DEVICE_ATTR(memory_stats, 0444, memory_stats_show, NULL);
static DEVICE_ATTR(memory_limits, 0644, memory_limits_show, memory_limits_store);

/* Fix for module initialization order in orchestrator_init() in orchestrator_main.c */
static int __init orchestrator_init(void)
{
    int ret;
    dev_t dev;

    pr_info("LLM Orchestrator: Initializing module version %s\n", DRIVER_VERSION);

    /* Step 1: Initialize memory management first as it's a dependency for everything else */
    ret = memory_management_init();
    if (ret) {
        pr_err("orchestrator_init: Failed to initialize memory management: %d\n", ret);
        return ret;
    }
    pr_info("LLM Orchestrator: Memory management initialized\n");

    /* Step 2: Initialize JSON manager */
    ret = json_manager_init();
    if (ret) {
        pr_err("orchestrator_init: Failed to initialize JSON manager: %d\n", ret);
        goto fail_json;
    }
    pr_info("LLM Orchestrator: JSON manager initialized\n");

    /* Step 3: Initialize context management */
    ret = context_management_init();
    if (ret) {
        pr_err("orchestrator_init: Failed to initialize context management: %d\n", ret);
        goto fail_context;
    }
    pr_info("LLM Orchestrator: Context management initialized\n");

    /* Step 4: Initialize network subsystem */
    ret = network_init();
    if (ret) {
        pr_err("orchestrator_init: Failed to initialize network subsystem: %d\n", ret);
        goto fail_network;
    }
    pr_info("LLM Orchestrator: Network subsystem initialized\n");

    /* Step 5: Initialize TLS subsystem */
    ret = tls_init();
    if (ret) {
        pr_err("orchestrator_init: Failed to initialize TLS subsystem: %d\n", ret);
        goto fail_tls;
    }
    pr_info("LLM Orchestrator: TLS subsystem initialized\n");
    /*Step 8 Scheduler Registry Init*/
    ret = scheduler_registry_init();
    if (ret) {
        pr_err("orchestrator_init: Failed to initialize scheduler registry: %d\n", ret);
        goto fail_registry;
    }
    pr_info("LLM Orchestrator: Scheduler registry initialized\n");


    /* Step 7: Initialize scheduler and global state */
    scheduler_init(&global_scheduler);
    set_scheduler_state(&global_scheduler);
    memset(&global_response, 0, sizeof(global_response));
    pr_info("LLM Orchestrator: Scheduler initialized\n");

    /* Step 7: Set API keys securely */
    if (openai_api_key && strlen(openai_api_key) > 0) {
        if (set_api_key(PROVIDER_OPENAI, openai_api_key) == 0) {
            /* Clear the module parameter after securely storing it */
            memzero_explicit(openai_api_key, strlen(openai_api_key));
        } else {
            pr_warn("Failed to securely store OpenAI API key\n");
        }
    } else {
        pr_warn("No OpenAI API key provided\n");
    }

    if (anthropic_api_key && strlen(anthropic_api_key) > 0) {
        if (set_api_key(PROVIDER_ANTHROPIC, anthropic_api_key) == 0) {
            /* Clear the module parameter after securely storing it */
            memzero_explicit(anthropic_api_key, strlen(anthropic_api_key));
        } else {
            pr_warn("Failed to securely store Anthropic API key\n");
        }
    } else {
        pr_warn("No Anthropic API key provided\n");
    }

    if (google_gemini_api_key && strlen(google_gemini_api_key) > 0) {
        if (set_api_key(PROVIDER_GOOGLE_GEMINI, google_gemini_api_key) == 0) {
            /* Clear the module parameter after securely storing it */
            memzero_explicit(google_gemini_api_key, strlen(google_gemini_api_key));
        } else {
            pr_warn("Failed to securely store Google Gemini API key\n");
        }
    } else {
        pr_warn("No Google Gemini API key provided\n");
    }

    /* Step 8: Register character device */
    ret = alloc_chrdev_region(&dev, 0, 1, MODULE_NAME);
    if (ret < 0) {
        pr_err("orchestrator_init: Failed to allocate chrdev region: %d\n", ret);
        goto fail_chrdev;
    }
    major_number = MAJOR(dev);

    cdev_init(&orchestrator_cdev, &orchestrator_fops);
    orchestrator_cdev.owner = THIS_MODULE;

    ret = cdev_add(&orchestrator_cdev, dev, 1);
    if (ret < 0) {
        pr_err("orchestrator_init: Failed to add cdev: %d\n", ret);
        goto fail_cdev_add;
    }

    /* Step 9: Create device class and device */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 14, 0)
    orchestrator_class = class_create(MODULE_NAME);
#else
    orchestrator_class = class_create(MODULE_NAME);
#endif

    if (IS_ERR(orchestrator_class)) {
        ret = PTR_ERR(orchestrator_class);
        pr_err("orchestrator_init: Failed to create device class: %d\n", ret);
        goto fail_class;
    }

    orchestrator_device = device_create(orchestrator_class, NULL, dev, NULL, MODULE_NAME);
    if (IS_ERR(orchestrator_device)) {
        ret = PTR_ERR(orchestrator_device);
        pr_err("orchestrator_init: Failed to create device: %d\n", ret);
        goto fail_device;
    }

    /* Step 10: Create sysfs attributes */
    ret = device_create_file(orchestrator_device, &dev_attr_scheduler_algorithm);
    if (ret) {
        pr_err("orchestrator_init: Failed to create scheduler_algorithm sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_provider_metrics);
    if (ret) {
        pr_err("orchestrator_init: Failed to create provider_metrics sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_reset_metrics);
    if (ret) {
        pr_err("orchestrator_init: Failed to create reset_metrics sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_scheduler_weights);
    if (ret) {
        pr_err("orchestrator_init: Failed to create scheduler_weights sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_auto_adjust);
    if (ret) {
        pr_err("orchestrator_init: Failed to create auto_adjust sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_fifo_status);
    if (ret) {
        pr_err("orchestrator_init: Failed to create fifo_status sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_context_status);
    if (ret) {
        pr_err("orchestrator_init: Failed to create context_status sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_memory_stats);
    if (ret) {
        pr_err("orchestrator_init: Failed to create memory_stats sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_memory_limits);
    if (ret) {
        pr_err("orchestrator_init: Failed to create memory_limits sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    /* Step 11: Initialize mutex */
    mutex_init(&orchestrator_mutex);

    /* Step 12: Setup maintenance timer */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 15, 0)
    timer_setup(&maintenance_timer, maintenance_timer_callback, 0);
#else
    setup_timer(&maintenance_timer, maintenance_timer_callback, 0);
#endif
    mod_timer(&maintenance_timer, jiffies + HZ * 60 * 10);

    pr_info("LLM Orchestrator: Module loaded successfully with major number %d\n", major_number);
    return 0;

    /* Error handling with proper cleanup */
    fail_registry:
    pr_info("Registry Failed");
    fail_sysfs:
    device_destroy(orchestrator_class, MKDEV(major_number, 0));
    fail_device:
    class_destroy(orchestrator_class);
    fail_class:
    cdev_del(&orchestrator_cdev);
    fail_cdev_add:
    unregister_chrdev_region(MKDEV(major_number, 0), 1);
    fail_chrdev:
    clear_all_api_keys();
    tls_cleanup();
    fail_tls:
    network_cleanup();
    fail_network:
    context_cleanup_all();
    context_management_cleanup();
    fail_context:
    json_manager_cleanup();
    fail_json:
    memory_management_cleanup();
    return ret;
}


/* Fix for module exit to ensure proper cleanup order */
static void __exit orchestrator_exit(void)
{
    pr_info("LLM Orchestrator: Unloading module\n");

    /* Step 1: Stop the maintenance timer */
    del_timer_sync(&maintenance_timer);
    pr_debug("LLM Orchestrator: Maintenance timer stopped\n");

    /* Step 2: Remove sysfs attributes */
    device_remove_file(orchestrator_device, &dev_attr_scheduler_algorithm);
    device_remove_file(orchestrator_device, &dev_attr_provider_metrics);
    device_remove_file(orchestrator_device, &dev_attr_reset_metrics);
    device_remove_file(orchestrator_device, &dev_attr_scheduler_weights);
    device_remove_file(orchestrator_device, &dev_attr_auto_adjust);
    device_remove_file(orchestrator_device, &dev_attr_fifo_status);
    device_remove_file(orchestrator_device, &dev_attr_context_status);
    device_remove_file(orchestrator_device, &dev_attr_memory_stats);
    device_remove_file(orchestrator_device, &dev_attr_memory_limits);
    pr_debug("LLM Orchestrator: Sysfs attributes removed\n");

    /* Step 3: Unregister device */
    device_destroy(orchestrator_class, MKDEV(major_number, 0));
    class_destroy(orchestrator_class);
    cdev_del(&orchestrator_cdev);
    unregister_chrdev_region(MKDEV(major_number, 0), 1);
    pr_debug("LLM Orchestrator: Character device unregistered\n");

    /* Step 4: Clean up resources in reverse initialization order */
    clear_all_api_keys();
    pr_debug("LLM Orchestrator: API keys securely cleared\n");

    /* Step 5: Clean up TLS */
    tls_cleanup();
    pr_debug("LLM Orchestrator: TLS subsystem cleaned up\n");

    /* Step 6: Clean up network */
    network_cleanup();
    pr_debug("LLM Orchestrator: Network subsystem cleaned up\n");

    /* Step 7: Clean up context */
    context_cleanup_all();
    context_management_cleanup();
    pr_debug("LLM Orchestrator: Context management cleaned up\n");

    /* Step 8: Clean up JSON */
    json_manager_cleanup();
    pr_debug("LLM Orchestrator: JSON manager cleaned up\n");

    /* Step 9: Clean up FIFO queue and memory management last */
    fifo_cleanup(&global_scheduler.fifo);
    memory_management_cleanup();
    pr_debug("LLM Orchestrator: Memory management cleaned up\n");
	scheduler_registry_cleanup();
    pr_debug("LLM Orchestrator: Scheduler registry cleaned up\n");
    
    pr_info("LLM Orchestrator: Module unloaded\n");
}



module_init(orchestrator_init);
module_exit(orchestrator_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LLM Orchestrator");
MODULE_DESCRIPTION("Enhanced LLM Orchestrator with Context Management and Advanced Scheduling");
MODULE_VERSION(DRIVER_VERSION);
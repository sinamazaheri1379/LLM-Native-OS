/**
 * orchestrator_main.c - LLM Orchestrator with CPU-like Scheduling
 *
 * This implementation uses a CPU-like scheduling approach where:
 * 1. Client threads submit requests through orchestrator_write()
 * 2. Requests are queued in priority queues managed by the scheduler
 * 3. A dispatcher thread assigns requests to provider worker threads
 * 4. Each provider worker processes requests independently
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/timer.h>
#include <linux/wait.h>
#include <linux/kthread.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/hashtable.h>
#include <linux/ktime.h>
#include "orchestrator_main.h"

#define MODULE_NAME "llm_orchestrator"
#define RESPONSE_HASH_BITS 8  /* 256 buckets */

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

/* Character device globals */
static int major_number;
static struct class *orchestrator_class;
static struct device *orchestrator_device;
static struct cdev orchestrator_cdev;
static DEFINE_MUTEX(orchestrator_mutex);

/* Global state */
static struct scheduler_state global_scheduler;

/* Maintenance timer */
static struct timer_list maintenance_timer;

/* CPU-like scheduler components */
static atomic_t scheduler_running = ATOMIC_INIT(0);
static struct task_struct *dispatcher_thread;
static struct task_struct *provider_threads[PROVIDER_COUNT];
static wait_queue_head_t dispatcher_wait_queue;
static wait_queue_head_t provider_wait_queues[PROVIDER_COUNT];
static wait_queue_head_t read_wait_queue; /* Wait queue for clients reading responses */
static atomic_t next_request_waiting = ATOMIC_INIT(0);

/* Queue for each provider */
struct request_queue {
    struct llm_request *request;   /* Current request */
    atomic_t has_request;          /* Flag indicating if request is present */
};

static struct request_queue provider_queues[PROVIDER_COUNT];
static spinlock_t provider_queue_locks[PROVIDER_COUNT];
static atomic_t provider_available[PROVIDER_COUNT];

/* Response storage */
struct response_entry {
    int request_id;
    struct llm_response response;
    atomic_t complete;
    struct hlist_node node;
};

static DEFINE_HASHTABLE(response_table, RESPONSE_HASH_BITS);
static DEFINE_SPINLOCK(response_lock);

/* Request ID counter */
static atomic_t request_counter = ATOMIC_INIT(0);

/* File to request ID mapping */
struct file_request_mapping {
    struct file *file;
    int request_id;
    struct hlist_node node;
};

static DEFINE_HASHTABLE(file_request_map, RESPONSE_HASH_BITS);
static DEFINE_SPINLOCK(file_map_lock);

/* Provider configurations */
struct llm_provider_config provider_configs[PROVIDER_COUNT] = {
    { "api.openai.com", "127.0.0.1", 8080, "/v1/chat/completions" },      /* OpenAI */
    { "api.anthropic.com", "127.0.0.1", 8080, "/v1/messages" },           /* Anthropic */
    { "generativelanguage.googleapis.com", "127.0.0.1", 8080, "/v1/models/gemini-1.5-pro:generateContent" } /* Gemini */
};

/* Function prototypes */
static int orchestrator_open(struct inode *inode, struct file *file);
static int orchestrator_release(struct inode *inode, struct file *file);
static ssize_t orchestrator_read(struct file *file, char __user *buf, size_t count, loff_t *offset);
static ssize_t orchestrator_write(struct file *file, const char __user *buf, size_t count, loff_t *offset);
static long orchestrator_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
static int dispatcher_thread_fn(void *data);
static int provider_worker_thread(void *data);
static void store_response(struct llm_request *req, struct llm_response *resp);
static struct file *find_file_by_request_id(int request_id);
static void mark_request_complete(int request_id);
static void copy_response_to_wrapper(int request_id, struct llm_response_wrapper *wrapper);
static ssize_t provider_host_show(struct device *dev, struct device_attribute *attr, char *buf);
static ssize_t provider_host_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count);
static ssize_t scheduler_algorithm_show(struct device *dev, struct device_attribute *attr, char *buf);
static ssize_t scheduler_algorithm_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count);
static void maintenance_timer_callback(struct timer_list *t);

/* Device attributes */
static DEVICE_ATTR(provider_hosts, 0644, provider_host_show, provider_host_store);
static DEVICE_ATTR(scheduler_algorithm, 0644, scheduler_algorithm_show, scheduler_algorithm_store);
// Create a simple implementation for get_scheduler_state():
struct scheduler_state *get_scheduler_state(void)
{
    return &global_scheduler;
}

// And remove_scheduler_state can be a no-op:
void remove_scheduler_state(void)
{
    // No operation needed
}

/* File operations */
static struct file_operations orchestrator_fops = {
    .owner = THIS_MODULE,
    .open = orchestrator_open,
    .release = orchestrator_release,
    .read = orchestrator_read,
    .write = orchestrator_write,
    .unlocked_ioctl = orchestrator_ioctl
};

/* Function to get provider configuration */
struct llm_provider_config *get_provider_config(int provider_id) {
    if (provider_id >= 0 && provider_id < PROVIDER_COUNT)
        return &provider_configs[provider_id];
    return NULL;
}

/* Dispatcher thread function - selects requests and assigns to provider workers */
static int dispatcher_thread_fn(void *data)
{
    while (!kthread_should_stop()) {
        struct llm_request *req;
        int provider_id = -1;
        int i;

        /* Get next request from scheduler */
        req = scheduler_get_next_request();
        if (!req) {
            /* No requests, wait for notification */
            wait_event_interruptible_timeout(
                dispatcher_wait_queue,
                atomic_read(&next_request_waiting) || kthread_should_stop(),
                msecs_to_jiffies(100)
            );
            atomic_set(&next_request_waiting, 0);
            continue;
        }

        /* Find available provider */
        for (i = 0; i < PROVIDER_COUNT; i++) {
            if (atomic_read(&provider_available[i])) {
                provider_id = i;
                break;
            }
        }

        /* If provider override is specified, check if it's available */
        if (req->provider_override >= 0 && req->provider_override < PROVIDER_COUNT) {
            if (atomic_read(&provider_available[req->provider_override])) {
                provider_id = req->provider_override;
            } else {
                /* If specified provider is unavailable, requeue and wait */
                scheduler_submit_request(req, req->priority);
                kfree(req); /* Free the copy returned by scheduler_get_next_request */
                msleep(100);
                continue;
            }
        }

        /* If no providers available, requeue and wait */
        if (provider_id < 0) {
            scheduler_submit_request(req, req->priority);
            kfree(req); /* Free the copy returned by scheduler_get_next_request */
            msleep(100);
            continue;
        }

        /* Assign request to provider */
        spin_lock(&provider_queue_locks[provider_id]);
        if (provider_queues[provider_id].request != NULL) {
            /* This shouldn't happen if available flag is accurate */
            pr_warn("Provider %d queue not empty despite available flag\n", provider_id);
            scheduler_submit_request(req, req->priority);
            kfree(req);
            spin_unlock(&provider_queue_locks[provider_id]);
            continue;
        }

        provider_queues[provider_id].request = req;
        atomic_set(&provider_queues[provider_id].has_request, 1);
        spin_unlock(&provider_queue_locks[provider_id]);

        /* Wake up provider worker */
        wake_up(&provider_wait_queues[provider_id]);
    }

    return 0;
}

static int provider_worker_thread(void *data)
{
    int provider_id = (int)(long)data;

    while (!kthread_should_stop()) {
        struct llm_request *req = NULL;
        const char *api_key;
        struct llm_response *resp; /* Changed to pointer */
        int ret;

        /* Wait for request */
        wait_event_interruptible_timeout(
            provider_wait_queues[provider_id],
            atomic_read(&provider_queues[provider_id].has_request) ||
            kthread_should_stop(),
            msecs_to_jiffies(100)
        );

        if (kthread_should_stop())
            break;

        /* Check if we have a request */
        if (!atomic_read(&provider_queues[provider_id].has_request))
            continue;

        /* Mark provider as busy */
        atomic_set(&provider_available[provider_id], 0);

        /* Get request */
        spin_lock(&provider_queue_locks[provider_id]);
        req = provider_queues[provider_id].request;
        provider_queues[provider_id].request = NULL;
        atomic_set(&provider_queues[provider_id].has_request, 0);
        spin_unlock(&provider_queue_locks[provider_id]);

        if (!req) {
            atomic_set(&provider_available[provider_id], 1);
            continue;
        }

        /* Allocate response struct on the heap instead of stack */
        resp = kmalloc(sizeof(struct llm_response), GFP_KERNEL);
        if (!resp) {
            pr_err("Failed to allocate response buffer\n");
            atomic_set(&provider_available[provider_id], 1);
            kfree(req);
            continue;
        }

        /* Get API key */
        switch (provider_id) {
            case PROVIDER_OPENAI:
                api_key = openai_api_key;
                break;
            case PROVIDER_ANTHROPIC:
                api_key = anthropic_api_key;
                break;
            case PROVIDER_GOOGLE_GEMINI:
                api_key = google_gemini_api_key;
                break;
            default:
                api_key = NULL;
        }

        /* Process request */
        memset(resp, 0, sizeof(struct llm_response));
        switch (provider_id) {
            case PROVIDER_OPENAI:
                ret = llm_send_openai(api_key, req, resp);
                break;
            case PROVIDER_ANTHROPIC:
                ret = llm_send_anthropic(api_key, req, resp);
                break;
            case PROVIDER_GOOGLE_GEMINI:
                ret = llm_send_google_gemini(api_key, req, resp);
                break;
            default:
                ret = -EINVAL;
        }

        /* Handle rate limiting */
        if (ret == -LLM_ERR_RATE_LIMIT) {
            struct scheduler_state *state = get_scheduler_state();
            if (state) {
                handle_rate_limit(provider_id, state, resp->rate_limit_reset_ms);
            }
        } else {
            /* Mark provider as available */
            atomic_set(&provider_available[provider_id], 1);
        }

        /* Store response in requestor's buffer */
        store_response(req, resp);

        /* Free allocated memory */
        kfree(resp);
        kfree(req);
    }
    return 0;
}

/* Store response for a request */
static void store_response(struct llm_request *req, struct llm_response *resp)
{
    struct response_entry *entry;
    unsigned long flags;

    /* Allocate response entry */
    entry = kmalloc(sizeof(*entry), GFP_KERNEL);
    if (!entry)
        return;

    /* Initialize entry */
    entry->request_id = req->request_id;
    memcpy(&entry->response, resp, sizeof(*resp));
    atomic_set(&entry->complete, 1);

    /* Add to hash table */
    spin_lock_irqsave(&response_lock, flags);
    hash_add(response_table, &entry->node, entry->request_id);
    spin_unlock_irqrestore(&response_lock, flags);

    /* Mark the request complete in wrapper */
    mark_request_complete(req->request_id);
}

/* Find file associated with a request ID */
static struct file *find_file_by_request_id(int request_id)
{
    struct file_request_mapping *mapping;
    struct file *file = NULL;
    unsigned long flags;

    spin_lock_irqsave(&file_map_lock, flags);
    hash_for_each_possible(file_request_map, mapping, node, request_id) {
        if (mapping->request_id == request_id) {
            file = mapping->file;
            break;
        }
    }
    spin_unlock_irqrestore(&file_map_lock, flags);

    return file;
}

/* Mark request as complete and notify waiting clients */
static void mark_request_complete(int request_id)
{
    struct file *file;
    struct llm_response_wrapper *wrapper;

    file = find_file_by_request_id(request_id);
    if (!file)
        return;

    wrapper = file->private_data;
    if (wrapper) {
        atomic_set(&wrapper->completed, 1);
        copy_response_to_wrapper(request_id, wrapper);
        wake_up(&read_wait_queue);
    }
}

/* Copy response data to wrapper */
static void copy_response_to_wrapper(int request_id, struct llm_response_wrapper *wrapper)
{
    struct response_entry *entry;
    unsigned long flags;
    bool found = false;

    spin_lock_irqsave(&response_lock, flags);
    hash_for_each_possible(response_table, entry, node, request_id) {
        if (entry->request_id == request_id) {
            memcpy(&wrapper->resp, &entry->response, sizeof(struct llm_response));
            found = true;
            break;
        }
    }
    spin_unlock_irqrestore(&response_lock, flags);

    if (!found) {
        pr_warn("copy_response_to_wrapper: No response found for request ID %d\n", request_id);
    }
}
/* OpenAI API implementation */
int llm_send_openai(const char *api_key, struct llm_request *req, struct llm_response *resp)
{
    struct llm_provider_config *config = get_provider_config(PROVIDER_OPENAI);
    char *openai_host = config->host_ip;
    int port = config->port;
    char *path = config->path;
    struct llm_json_buffer json_buf;
    int ret;
    char auth_header[512];
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

    /* Send request to OpenAI API */
    ret = network_send_request(openai_host, port, path,
                               NULL, auth_header, false,
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
    struct llm_provider_config *config = get_provider_config(PROVIDER_ANTHROPIC);
    char *anthropic_host = config->host_ip;
    int port = config->port;
    char *path = config->path;
    struct llm_json_buffer json_buf;
    int ret;
    char auth_header[512];
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
                               NULL, auth_header, false,
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
    struct llm_provider_config *config = get_provider_config(PROVIDER_GOOGLE_GEMINI);
    char *gemini_host = config->host_ip;
    int port = config->port;
    char *path = config->path;
    struct llm_json_buffer json_buf;
    int ret;
    char auth_path[512];
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
    if (strchr(path, '?') != NULL) {
    	snprintf(auth_path, sizeof(auth_path), "/gemini%s&key=%s", path, api_key);
	} else {
    	snprintf(auth_path, sizeof(auth_path), "/gemini%s?key=%s", path, api_key);
	}

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
                               NULL, NULL, false,
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

/* File open operation */
static int orchestrator_open(struct inode *inode, struct file *file)
{
    struct llm_response_wrapper *wrapper;

    /* Allocate and initialize response wrapper for this file */
    wrapper = kmalloc(sizeof(*wrapper), GFP_KERNEL);
    if (!wrapper)
        return -ENOMEM;

    memset(&wrapper->resp, 0, sizeof(wrapper->resp));
    wrapper->request_id = atomic_inc_return(&request_counter);
    atomic_set(&wrapper->completed, 0);
    wrapper->priority = PRIORITY_NORMAL; /* Default priority */
    wrapper->preferred_provider = -1;    /* No preference by default */

    /* Store wrapper in file's private data */
    file->private_data = wrapper;
    return 0;
}

/* File release operation */
static int orchestrator_release(struct inode *inode, struct file *file)
{
    struct llm_response_wrapper *wrapper = file->private_data;
    struct file_request_mapping *mapping;
    struct hlist_node *tmp;
    unsigned long flags;

    /* Remove file from mapping table */
    spin_lock_irqsave(&file_map_lock, flags);
    hash_for_each_possible_safe(file_request_map, mapping, tmp, node, wrapper->request_id) {
        if (mapping->file == file) {
            hash_del(&mapping->node);
            kfree(mapping);
            break;
        }
    }
    spin_unlock_irqrestore(&file_map_lock, flags);

    /* Free response wrapper */
    if (wrapper) {
        kfree(wrapper);
        file->private_data = NULL;
    }

    /* Remove scheduler state from registry */
    remove_scheduler_state();
    return 0;
}

static ssize_t orchestrator_write(struct file *file, const char __user *buf, size_t count, loff_t *offset)
{
    struct llm_response_wrapper *wrapper = file->private_data;
    struct file_request_mapping *mapping;
    unsigned long flags;
    int ret;
    struct llm_request *user_req = kmalloc(sizeof(struct llm_request), GFP_KERNEL);

    if (!user_req)
        return -ENOMEM;

    if (!wrapper || count != sizeof(struct llm_request)) {
        kfree(user_req);  /* Free allocation on error */
        return -EINVAL;
    }

    /* Copy and validate request from user */
    if (copy_from_user(user_req, buf, sizeof(*user_req))) {
        kfree(user_req);  /* Free allocation on error */
        return -EFAULT;
    }

    /* Validate request fields */
    if (user_req->conversation_id <= 0 || user_req->prompt[0] == '\0') {
        kfree(user_req);  /* Free allocation on error */
        return -EINVAL;
    }

    /* Override priority from wrapper */
    user_req->priority = wrapper->priority;

    /* Override provider if set via ioctl */
    if (user_req->provider_override < 0 && wrapper->preferred_provider >= 0)
        user_req->provider_override = wrapper->preferred_provider;

    /* Assign request ID */
    user_req->request_id = wrapper->request_id;

    /* Add file to request mapping */
    mapping = kmalloc(sizeof(*mapping), GFP_KERNEL);
    if (!mapping) {
        kfree(user_req);  /* Free allocation on error */
        return -ENOMEM;
    }

    mapping->file = file;
    mapping->request_id = user_req->request_id;

    spin_lock_irqsave(&file_map_lock, flags);
    hash_add(file_request_map, &mapping->node, mapping->request_id);
    spin_unlock_irqrestore(&file_map_lock, flags);

    /* Reset completion flag */
    atomic_set(&wrapper->completed, 0);

    /* Submit to scheduler directly - NON-BLOCKING */
    ret = scheduler_submit_request(user_req, user_req->priority);
    if (ret < 0) {
        /* Clean up mapping on error */
        spin_lock_irqsave(&file_map_lock, flags);
        hash_del(&mapping->node);
        spin_unlock_irqrestore(&file_map_lock, flags);
        kfree(mapping);
        kfree(user_req);  /* Free allocation on error */
        return ret;
    }

    /* Signal dispatcher that a new request is waiting */
    atomic_set(&next_request_waiting, 1);
    wake_up(&dispatcher_wait_queue);

    /* We can free user_req here because scheduler_submit_request makes a copy */
    kfree(user_req);

    return count;
}

/* Read operation - get response from provider */
static ssize_t orchestrator_read(struct file *file, char __user *buf, size_t count, loff_t *offset)
{
    struct llm_response_wrapper *wrapper = file->private_data;
    ssize_t ret;
    char *extracted_content = NULL;

    if (!wrapper)
        return -EINVAL;

    /* Wait for completion if necessary */
    if (!atomic_read(&wrapper->completed)) {
        if (file->f_flags & O_NONBLOCK)
            return -EAGAIN;

        /* Wait for completion with timeout */
        ret = wait_event_interruptible_timeout(
            read_wait_queue,
            atomic_read(&wrapper->completed) != 0,
            msecs_to_jiffies(30000)  /* 30-second timeout */
        );

        if (ret == 0)
            return -ETIMEDOUT;

        if (ret < 0)
            return ret;
    }

    /* Allocate buffer for the extracted content */
    extracted_content = kmalloc(MAX_RESPONSE_LENGTH, GFP_KERNEL);
    if (!extracted_content) {
        pr_err("orchestrator_read: Failed to allocate content buffer\n");
        return -ENOMEM;
    }

    /* Choose the appropriate extractor based on the provider used */
    switch (wrapper->resp.provider_used) {
        case PROVIDER_OPENAI:
            pr_debug("Providing OpenAI response\n");
            ret = extract_openai_content(wrapper->resp.content, extracted_content, MAX_RESPONSE_LENGTH);
            break;

        case PROVIDER_ANTHROPIC:
            pr_debug("Providing Anthropic response\n");
            ret = extract_anthropic_content(wrapper->resp.content, extracted_content, MAX_RESPONSE_LENGTH);
            break;

        case PROVIDER_GOOGLE_GEMINI:
            pr_debug("Providing Google Gemini response\n");
            ret = extract_gemini_content(wrapper->resp.content, extracted_content, MAX_RESPONSE_LENGTH);
            break;

        default:
            pr_warn("orchestrator_read: Unknown provider %d, using generic extractor\n",
                   wrapper->resp.provider_used);
            ret = extract_response_content(wrapper->resp.content, extracted_content, MAX_RESPONSE_LENGTH);
            break;
    }

    if (ret <= 0) {
        /* If specialized extraction fails, fall back to generic extractor */
        pr_debug("orchestrator_read: Provider-specific extractor failed, trying generic extractor\n");
        ret = extract_response_content(wrapper->resp.content, extracted_content, MAX_RESPONSE_LENGTH);
    }

    if (ret > 0) {
        /* Return the extracted content */
        if (count < ret) {
            kfree(extracted_content);
            return -EINVAL;
        }

        if (copy_to_user(buf, extracted_content, ret)) {
            kfree(extracted_content);
            return -EFAULT;
        }
    } else {
        /* Fallback: return the full response if extraction failed */
        pr_warn("orchestrator_read: Content extraction failed, returning full response\n");
        if (copy_to_user(buf, wrapper->resp.content, wrapper->resp.content_length)) {
            kfree(extracted_content);
            return -EFAULT;
        }
        ret = wrapper->resp.content_length;
    }

    kfree(extracted_content);
    wrapper->resp.content_length = 0; /* Reset for next read */
    return ret;
}

/* IOCTL handler for additional controls */
static long orchestrator_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct llm_response_wrapper *wrapper = file->private_data;
    int value;

    if (!wrapper)
        return -EINVAL;

    switch (cmd) {
        case IOCTL_SET_PREFERRED_PROVIDER:
            if (copy_from_user(&value, (int __user *)arg, sizeof(int)))
                return -EFAULT;

            if (value < -1 || value >= PROVIDER_COUNT)
                return -EINVAL;

            wrapper->preferred_provider = value;
            pr_debug("orchestrator_ioctl: Set preferred provider to %d\n", value);
            return 0;

        case IOCTL_SET_REQUEST_PRIORITY:
            if (copy_from_user(&value, (int __user *)arg, sizeof(int)))
                return -EFAULT;

            if (value < 0 || value >= PRIORITY_LEVELS)
                return -EINVAL;

            wrapper->priority = value;
            pr_debug("orchestrator_ioctl: Set priority to %d\n", value);
            return 0;

        case IOCTL_GET_REQUEST_STATUS:
            value = atomic_read(&wrapper->completed);
            if (copy_to_user((int __user *)arg, &value, sizeof(int)))
                return -EFAULT;
            return 0;

        default:
            return -ENOTTY; /* Inappropriate ioctl for device */
    }
}

/* Show provider host information */
static ssize_t provider_host_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    int i, len = 0;

    for (i = 0; i < PROVIDER_COUNT; i++) {
        len += scnprintf(buf + len, PAGE_SIZE - len, "Provider %d (%s): %s:%d%s\n",
                        i, provider_configs[i].domain_name,
                        provider_configs[i].host_ip, provider_configs[i].port,
                        provider_configs[i].path);
    }

    return len;
}

/* Store provider host information */
static ssize_t provider_host_store(struct device *dev, struct device_attribute *attr,
                                 const char *buf, size_t count)
{
    int provider, port;
    char ip[MAX_IP_LENGTH];

    if (sscanf(buf, "%d,%[^,],%d", &provider, ip, &port) != 3) {
        pr_err("provider_host_store: Invalid format, expected: provider_id,ip,port\n");
        return -EINVAL;
    }

    if (provider < 0 || provider >= PROVIDER_COUNT) {
        pr_err("provider_host_store: Invalid provider ID: %d\n", provider);
        return -EINVAL;
    }

    if (!is_ip_address_valid(ip)) {
        pr_err("provider_host_store: Invalid IP address: %s\n", ip);
        return -EINVAL;
    }

    if (port <= 0 || port > 65535) {
        pr_err("provider_host_store: Invalid port: %d\n", port);
        return -EINVAL;
    }

    /* Update the configuration */
    mutex_lock(&orchestrator_mutex);

    strncpy(provider_configs[provider].host_ip, ip, MAX_IP_LENGTH - 1);
    provider_configs[provider].host_ip[MAX_IP_LENGTH - 1] = '\0';
    provider_configs[provider].port = port;

    mutex_unlock(&orchestrator_mutex);

    pr_info("Updated provider %d (%s) to use IP: %s, port: %d\n",
            provider, provider_configs[provider].domain_name, ip, port);

    return count;
}

/* Show current scheduler algorithm */
static ssize_t scheduler_algorithm_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    int algorithm = atomic_read(&global_scheduler.current_algorithm);
    const char *name;

    switch (algorithm) {
        case SCHEDULER_ROUND_ROBIN: name = "Round Robin"; break;
        case SCHEDULER_WEIGHTED: name = "Weighted"; break;
        case SCHEDULER_PRIORITY: name = "Priority"; break;
        case SCHEDULER_PERFORMANCE: name = "Performance"; break;
        case SCHEDULER_COST_AWARE: name = "Cost Aware"; break;
        case SCHEDULER_FALLBACK: name = "Fallback"; break;
        case SCHEDULER_FIFO: name = "FIFO"; break;
        default: name = "Unknown"; break;
    }

    return scnprintf(buf, PAGE_SIZE, "%d (%s)\n", algorithm, name);
}

/* Set scheduler algorithm */
static ssize_t scheduler_algorithm_store(struct device *dev, struct device_attribute *attr,
                                        const char *buf, size_t count)
{
    int algorithm;

    if (kstrtoint(buf, 10, &algorithm) != 0)
        return -EINVAL;

    if (algorithm < 0 || algorithm > SCHEDULER_MAX_ALGORITHM)
        return -EINVAL;

    atomic_set(&global_scheduler.current_algorithm, algorithm);
    return count;
}

/* Initialize CPU-like scheduler components */
static int cpu_scheduler_init(void)
{
    int i, ret;

    /* Initialize wait queues */
    init_waitqueue_head(&dispatcher_wait_queue);
    init_waitqueue_head(&read_wait_queue);
    for (i = 0; i < PROVIDER_COUNT; i++) {
        init_waitqueue_head(&provider_wait_queues[i]);
        spin_lock_init(&provider_queue_locks[i]);
        atomic_set(&provider_available[i], 1);
        provider_queues[i].request = NULL;
        atomic_set(&provider_queues[i].has_request, 0);
    }

    /* Create provider worker threads */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        provider_threads[i] = kthread_run(provider_worker_thread,
                                         (void *)(long)i,
                                         "llm_worker_%d", i);
        if (IS_ERR(provider_threads[i])) {
            ret = PTR_ERR(provider_threads[i]);
            pr_err("Failed to create provider worker thread %d: %d\n", i, ret);
            goto fail_workers;
        }
    }

    /* Create dispatcher thread */
    dispatcher_thread = kthread_run(dispatcher_thread_fn, NULL, "llm_dispatcher");
    if (IS_ERR(dispatcher_thread)) {
        ret = PTR_ERR(dispatcher_thread);
        pr_err("Failed to create dispatcher thread: %d\n", ret);
        goto fail_dispatcher;
    }

    atomic_set(&scheduler_running, 1);
    pr_info("CPU-like LLM scheduler initialized\n");
    return 0;

fail_dispatcher:
    i = PROVIDER_COUNT;

fail_workers:
    atomic_set(&scheduler_running, 0);
    while (--i >= 0) {
        if (!IS_ERR_OR_NULL(provider_threads[i]))
            kthread_stop(provider_threads[i]);
    }
    return ret;
}

/* Clean up CPU-like scheduler components */
static void cpu_scheduler_cleanup(void)
{
    int i;

    atomic_set(&scheduler_running, 0);

    /* Stop dispatcher thread */
    if (!IS_ERR_OR_NULL(dispatcher_thread)) {
        kthread_stop(dispatcher_thread);
    }

    /* Stop provider worker threads */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (!IS_ERR_OR_NULL(provider_threads[i])) {
            kthread_stop(provider_threads[i]);
        }
    }

    pr_info("CPU-like LLM scheduler cleaned up\n");
}

/* Module initialization */
static int __init orchestrator_init(void)
{
    int ret;
    dev_t dev;

    pr_info("LLM Orchestrator: Initializing module version %s\n", DRIVER_VERSION);

    /* Step 1: Initialize memory management */
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

    /* Step 6: Initialize scheduler registry */
    pr_info("Initializing global scheduler\n");
	ret = 0;

    /* Step 7: Initialize scheduler and global state */
    scheduler_init(&global_scheduler);
    pr_info("LLM Orchestrator: Scheduler initialized\n");

    /* Step 8: Initialize CPU-like scheduler components */
    ret = cpu_scheduler_init();
    if (ret) {
        pr_err("orchestrator_init: Failed to initialize CPU-like scheduler: %d\n", ret);
        goto fail_cpu_scheduler;
    }
    pr_info("LLM Orchestrator: CPU-like scheduler initialized\n");

    /* Step 9: Register character device */
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

    /* Step 10: Create device class and device */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 14, 0)
    orchestrator_class = class_create(MODULE_NAME);
#else
    orchestrator_class = class_create(THIS_MODULE, MODULE_NAME);
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

    /* Step 11: Create sysfs attributes */
    ret = device_create_file(orchestrator_device, &dev_attr_provider_hosts);
    if (ret) {
        pr_err("orchestrator_init: Failed to create provider_hosts sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

    ret = device_create_file(orchestrator_device, &dev_attr_scheduler_algorithm);
    if (ret) {
        pr_err("orchestrator_init: Failed to create scheduler_algorithm sysfs attribute: %d\n", ret);
        goto fail_sysfs;
    }

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
    fail_sysfs:
    device_destroy(orchestrator_class, MKDEV(major_number, 0));
    fail_device:
    class_destroy(orchestrator_class);
    fail_class:
    cdev_del(&orchestrator_cdev);
    fail_cdev_add:
    unregister_chrdev_region(MKDEV(major_number, 0), 1);
    fail_chrdev:
    cpu_scheduler_cleanup();
    fail_cpu_scheduler:
    /* TLS cleanup */
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

/* Module exit */
static void __exit orchestrator_exit(void)
{
    pr_info("LLM Orchestrator: Unloading module\n");

    /* Step 1: Stop the maintenance timer */
    del_timer_sync(&maintenance_timer);
    pr_debug("LLM Orchestrator: Maintenance timer stopped\n");

    /* Step 2: Remove sysfs attributes */
    device_remove_file(orchestrator_device, &dev_attr_provider_hosts);
    device_remove_file(orchestrator_device, &dev_attr_scheduler_algorithm);
    pr_debug("LLM Orchestrator: Sysfs attributes removed\n");

    /* Step 3: Unregister device */
    device_destroy(orchestrator_class, MKDEV(major_number, 0));
    class_destroy(orchestrator_class);
    cdev_del(&orchestrator_cdev);
    unregister_chrdev_region(MKDEV(major_number, 0), 1);
    pr_debug("LLM Orchestrator: Character device unregistered\n");

    /* Step 4: Clean up CPU-like scheduler */
    cpu_scheduler_cleanup();
    pr_debug("LLM Orchestrator: CPU-like scheduler cleaned up\n");

    /* Step 5: Clean up scheduler registry */
    pr_debug("LLM Orchestrator: Scheduler registry cleaned up\n");

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

    /* Step 9: Clean up memory management */
    memory_management_cleanup();
    pr_debug("LLM Orchestrator: Memory management cleaned up\n");

    pr_info("LLM Orchestrator: Module unloaded\n");
}

module_init(orchestrator_init);
module_exit(orchestrator_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LLM Orchestrator");
MODULE_DESCRIPTION("Enhanced LLM Orchestrator with CPU-like Scheduling");
MODULE_VERSION(DRIVER_VERSION);
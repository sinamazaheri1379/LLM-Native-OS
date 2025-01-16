#include <linux/moduleparam.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/mutex.h>
#include <linux/list.h>
#include <linux/jiffies.h>
#include <linux/device.h>
#include <linux/fs.h>
#include "llm_providers.h"

static int major_number;
static struct class *llm_class;
static struct device *llm_device;
static struct llm_conn_pool *conn_pool = NULL;
static struct llm_mem_tracker mem_tracker;


static const struct llm_config_validator config_validators[] = {
    DEFINE_VALIDATOR("api_key", validate_api_key, "API key validation"),
    DEFINE_VALIDATOR("model", validate_model, "Model name validation"),
    DEFINE_VALIDATOR("endpoint", validate_endpoint, "Endpoint URL validation"),
    // Add more validators as needed
};

static inline struct llm_config *get_global_config(void) {
    struct llm_config *config;
    mutex_lock(&llm_mutex);
    config = global_config;
    mutex_unlock(&llm_mutex);
    return config;
}

static inline int llm_check_version(struct llm_version *version) {
    if (!version) return -LLM_ERR_INVALID_PARAM;
    if (version->major != LLM_API_VERSION_MAJOR) return -LLM_ERR_INVALID_PARAM;
    if (version->minor > LLM_API_VERSION_MINOR) return -LLM_ERR_INVALID_PARAM;
    return LLM_ERR_SUCCESS;
}

static inline void llm_config_cleanup(struct llm_config *config) {
    struct llm_message *msg, *tmp_msg;
    struct llm_tool *tool, *tmp_tool;

    if (!config || !mutex_is_locked(&config->config_lock))
        return;

    /* Clean message history */
    list_for_each_entry_safe(msg, tmp_msg, &config->message_history, list) {
        list_del(&msg->list);
        llm_message_free(msg);
    }

    /* Clean tools */
    list_for_each_entry_safe(tool, tmp_tool, &config->tools, list) {
        list_del(&tool->list);
        llm_tool_free(tool);
    }

    mutex_destroy(&config->message_lock);
    mutex_destroy(&config->tool_lock);
    /* config_lock destroyed last */
    mutex_destroy(&config->config_lock);
}
static inline int llm_json_buffer_init(struct llm_json_buffer *buf, size_t size) {
    if (!buf || size == 0)
        return -LLM_ERR_INVALID_PARAM;

    buf->data = kmalloc(size, GFP_KERNEL);
    if (!buf->data)
        return -LLM_ERR_NOMEM;

    buf->size = size;
    buf->used = 0;
    buf->data[0] = '\0';
    return 0;
}
/* Core validation functions */
static inline int llm_validate_config(const struct llm_config *config) {
    if (!config) return LLM_ERR_INVALID_PARAM;
    if (!mutex_is_locked(&config->config_lock)) return LLM_ERR_MUTEX_LOCK;
    if (!LLM_VALID_MAX_TOKENS(config->max_tokens)) return LLM_ERR_INVALID_PARAM;
    if (!LLM_VALID_N_CHOICES(config->n_choices)) return LLM_ERR_INVALID_PARAM;
    if (!LLM_VALID_TEMP_RANGE(config->temperature_X100)) return LLM_ERR_INVALID_PARAM;
    if (!LLM_VALID_PENALTY_RANGE(config->presence_penalty_X100)) return LLM_ERR_INVALID_PARAM;
    if (!LLM_VALID_TIMEOUT(config->timeout_ms)) return LLM_ERR_INVALID_PARAM;
    if (!LLM_VALID_REQ_LIMIT(config->max_requests_per_min)) return LLM_ERR_INVALID_PARAM;
    if (!LLM_VALID_ENDPOINT(config->endpoint)) return LLM_ERR_INVALID_PARAM;
    if (!LLM_VALID_USER_ID(config->user_id)) return LLM_ERR_INVALID_PARAM;
    if (config->num_stop_sequences > 4) return LLM_ERR_INVALID_PARAM;
    if (atomic_read(&config->ref_count) <= 0) return LLM_ERR_REF_COUNT;
    return LLM_ERR_SUCCESS;
}


/* Reference counting */
static inline void llm_config_get(struct llm_config *config) {
    smp_mb__before_atomic();
    atomic_inc(&config->ref_count);
    smp_mb__after_atomic();
}

static inline void llm_config_put(struct llm_config *config) {
    if (!config)
        return;

    smp_mb__before_atomic();
    if (atomic_dec_and_test(&config->ref_count)) {
        smp_mb__after_atomic();
        mutex_lock(&config->config_lock);
        llm_config_cleanup(config);
        kfree(config);
    }
}

#define DEVICE_NAME "llm"

/* Device operations */
static int llm_open(struct inode *inode, struct file *file);
static int llm_release(struct inode *inode, struct file *file);
static ssize_t llm_read(struct file *file, char __user *buf, size_t count, loff_t *offset);
static ssize_t llm_write(struct file *file, const char __user *buf, size_t count, loff_t *offset);

static struct file_operations fops = {
    .open = llm_open,
    .read = llm_read,
    .write = llm_write,
    .release = llm_release
};



static char *api_key = NULL;
module_param(api_key, charp, 0600);
MODULE_PARM_DESC(api_key, "OpenAI API Key");

/* Global state */
static struct llm_config *global_config = NULL;
static DEFINE_MUTEX(llm_mutex);

/* JSON PARSING FUNCTIONS */
/* JSON Token Types */
enum json_token_type {
    JSON_STRING,
    JSON_NUMBER,
    JSON_OBJECT_START,
    JSON_OBJECT_END,
    JSON_ARRAY_START,
    JSON_ARRAY_END,
    JSON_TRUE,
    JSON_FALSE,
    JSON_NULL,
    JSON_COLON,
    JSON_COMMA,
    JSON_ERROR
};

struct json_token {
    enum json_token_type type;
    char *value;
    size_t len;
};

struct json_parser {
    const char *input;
    size_t pos;
    size_t len;
};

static void init_json_parser(struct json_parser *parser,
                           const char *input, size_t len) {
    parser->input = input;
    parser->pos = 0;
    parser->len = len;
}

void llm_json_buffer_free(struct llm_json_buffer *buf) {
    if (!buf)
        return;

    if (buf->data) {
        memzero_explicit(buf->data, buf->size);
        kfree(buf->data);
    }
    buf->data = NULL;
    buf->size = 0;
    buf->used = 0;
}

static int append_to_json(struct llm_json_buffer *buf, const char *str) {
    size_t len;

    if (!buf || !str || !buf->data)
        return -EINVAL;

    len = strlen(str);
    if (buf->used + len + 1 > buf->size)
        return -E2BIG;

    memcpy(buf->data + buf->used, str, len);
    buf->used += len;
    buf->data[buf->used] = '\0';
    return 0;
}
static int append_json_number(struct llm_json_buffer *buf, long long num) {
    char number[32];
    int ret;

    ret = snprintf(number, sizeof(number), "%lld", num);
    if (ret < 0 || ret >= sizeof(number))
        return -EINVAL;

    return append_to_json(buf, number);
}

static int append_json_float(struct llm_json_buffer *buf, int value_X100) {
    char number[32];
    int ret;

    ret = snprintf(number, sizeof(number), "%.2f",
                  (float)value_X100 / 100.0f);
    if (ret < 0 || ret >= sizeof(number))
        return -EINVAL;

    return append_to_json(buf, number);
}

static int append_json_string(struct llm_json_buffer *buf, const char *str) {
    char *escaped;
    size_t len, max_size;
    int ret;

    if (!buf || !str)
        return -EINVAL;

    len = strlen(str);
    if (len == 0)
        return append_to_json(buf, "\"\"");

    /* Check if escaping is needed first */
    bool needs_escape = false;
    for (size_t i = 0; i < len; i++) {
        if (str[i] < 32 || str[i] == '"' || str[i] == '\\') {
            needs_escape = true;
            break;
        }
    }

    if (!needs_escape) {
        /* Fast path - no escaping needed */
        ret = append_to_json(buf, "\"");
        if (ret) return ret;
        ret = append_to_json(buf, str);
        if (ret) return ret;
        return append_to_json(buf, "\"");
    }

    /* Slow path - escaping needed */
    max_size = len * 6 + 1; /* Worst case: each char needs escaping */
    escaped = kmalloc(max_size, GFP_KERNEL);
    if (!escaped)
        return -ENOMEM;

    ret = json_escape_string(str, escaped, max_size);
    if (ret < 0) {
        kfree(escaped);
        return ret;
    }

    ret = append_to_json(buf, "\"");
    if (ret) {
        kfree(escaped);
        return ret;
    }

    ret = append_to_json(buf, escaped);
    kfree(escaped);
    if (ret)
        return ret;

    return append_to_json(buf, "\"");
}

/* JSON Parser */


static void skip_whitespace(struct json_parser *parser) {
    while (parser->pos < parser->len &&
           isspace(parser->input[parser->pos]))
        parser->pos++;
}



/* Safe string handling */
static int json_unescape_string(const char *input, char *output, size_t outlen) {
    size_t i, j = 0;
    uint32_t unicode_val;

    if (!input || !output || outlen < 2)
        return -EINVAL;

    for (i = 0; input[i] && j + 1 < outlen; i++) {
        if (i >= strlen(input))
            return -EINVAL;

        if (input[i] == '\\') {
            i++;
            if (i >= strlen(input))
                return -EINVAL;

            switch (input[i]) {
                case '"':
                case '\\':
                case '/':
                    if (j + 1 >= outlen) return -E2BIG;
                    output[j++] = input[i];
                    break;
                case 'b':
                    if (j + 1 >= outlen) return -E2BIG;
                    output[j++] = '\b';
                    break;
                case 'f':
                    if (j + 1 >= outlen) return -E2BIG;
                    output[j++] = '\f';
                    break;
                case 'n':
                    if (j + 1 >= outlen) return -E2BIG;
                    output[j++] = '\n';
                    break;
                case 'r':
                    if (j + 1 >= outlen) return -E2BIG;
                    output[j++] = '\r';
                    break;
                case 't':
                    if (j + 1 >= outlen) return -E2BIG;
                    output[j++] = '\t';
                    break;
                case 'u': {
                    char hex[5] = {0};
                    int k;

                    /* Need 4 hex digits */
                    if (i + 4 >= strlen(input))
                        return -EINVAL;

                    /* Copy 4 hex digits */
                    for (k = 0; k < 4; k++)
                        hex[k] = input[i + k + 1];

                    /* Convert hex to integer */
                    if (kstrtou32(hex, 16, &unicode_val) < 0)
                        return -EINVAL;

                    /* Handle UTF-8 encoding */
                    if (unicode_val < 0x80) {
                        if (j + 1 >= outlen) return -E2BIG;
                        output[j++] = (char)unicode_val;
                    } else if (unicode_val < 0x800) {
                        if (j + 2 >= outlen) return -E2BIG;
                        output[j++] = (char)(0xC0 | (unicode_val >> 6));
                        output[j++] = (char)(0x80 | (unicode_val & 0x3F));
                    } else {
                        if (j + 3 >= outlen) return -E2BIG;
                        output[j++] = (char)(0xE0 | (unicode_val >> 12));
                        output[j++] = (char)(0x80 | ((unicode_val >> 6) & 0x3F));
                        output[j++] = (char)(0x80 | (unicode_val & 0x3F));
                    }
                    i += 4;  /* Skip the 4 hex digits */
                    break;
                }
                default:
                    return -EINVAL;
            }
        } else {
            if (j + 1 >= outlen) return -E2BIG;
            output[j++] = input[i];
        }
    }

    output[j] = '\0';
    return j;
}

static int json_escape_string(const char *input, char *output, size_t outlen) {
    size_t i, j = 0;

    for (i = 0; input[i]; i++) {
        if (j + 2 >= outlen) return -E2BIG;

        switch (input[i]) {
            case '"':
                if (j + 2 >= outlen) return -E2BIG;
                output[j++] = '\\';
                output[j++] = '"';
                break;
            case '\\':
                if (j + 2 >= outlen) return -E2BIG;
                output[j++] = '\\';
                output[j++] = '\\';
                break;
            case '\b':
                if (j + 2 >= outlen) return -E2BIG;
                output[j++] = '\\';
                output[j++] = 'b';
                break;
            case '\f':
                if (j + 2 >= outlen) return -E2BIG;
                output[j++] = '\\';
                output[j++] = 'f';
                break;
            case '\n':
                if (j + 2 >= outlen) return -E2BIG;
                output[j++] = '\\';
                output[j++] = 'n';
                break;
            case '\r':
                if (j + 2 >= outlen) return -E2BIG;
                output[j++] = '\\';
                output[j++] = 'r';
                break;
            case '\t':
                if (j + 2 >= outlen) return -E2BIG;
                output[j++] = '\\';
                output[j++] = 't';
                break;
            default:
                if (input[i] < 32) {
                    if (j + 6 >= outlen) return -E2BIG;
                    snprintf(&output[j], 7, "\\u%04x", input[i]);
                    j += 6;
                } else {
                    output[j++] = input[i];
                }
        }
    }

    output[j] = '\0';
    return j;
}

/* JSON Generation with validation */




static struct json_token get_next_token(struct json_parser *parser) {
    struct json_token token = {0};
    char c;

    skip_whitespace(parser);
    if (parser->pos >= parser->len) {
        token.type = JSON_ERROR;
        return token;
    }

    c = parser->input[parser->pos++];
    switch (c) {
        case '{':
            token.type = JSON_OBJECT_START;
            break;
        case '}':
            token.type = JSON_OBJECT_END;
            break;
        case '[':
            token.type = JSON_ARRAY_START;
            break;
        case ']':
            token.type = JSON_ARRAY_END;
            break;
        case ':':
            token.type = JSON_COLON;
            break;
        case ',':
            token.type = JSON_COMMA;
            break;
        case '"': {
            const char *start = &parser->input[parser->pos];
            const char *end = start;
            bool escaped = false;

            while (parser->pos < parser->len) {
                if (*end == '\\') {
                    escaped = !escaped;
                } else if (*end == '"' && !escaped) {
                    break;
                } else {
                    escaped = false;
                }
                end++;
                parser->pos++;
            }

            if (parser->pos >= parser->len) {
                token.type = JSON_ERROR;
                return token;
            }

            token.type = JSON_STRING;
            token.value = (char *)start;
            token.len = end - start;
            parser->pos++; // Skip closing quote
            break;
        }
        case 't':
            if (parser->pos + 3 <= parser->len &&
                strncmp(&parser->input[parser->pos-1], "true", 4) == 0) {
                token.type = JSON_TRUE;
                parser->pos += 3;
            } else {
                token.type = JSON_ERROR;
            }
            break;
        case 'f':
            if (parser->pos + 4 <= parser->len &&
                strncmp(&parser->input[parser->pos-1], "false", 5) == 0) {
                token.type = JSON_FALSE;
                parser->pos += 4;
            } else {
                token.type = JSON_ERROR;
            }
            break;
        case 'n':
            if (parser->pos + 3 <= parser->len &&
                strncmp(&parser->input[parser->pos-1], "null", 4) == 0) {
                token.type = JSON_NULL;
                parser->pos += 3;
            } else {
                token.type = JSON_ERROR;
            }
            break;
        default:
            if (isdigit(c) || c == '-') {
                const char *start = &parser->input[parser->pos - 1];
                const char *end = start;
                bool has_decimal = false;

                if (c == '-') {
                    end++;
                    parser->pos++;
                }

                while (parser->pos < parser->len &&
                       (isdigit(parser->input[parser->pos]) ||
                        parser->input[parser->pos] == '.')) {
                    if (parser->input[parser->pos] == '.') {
                        if (has_decimal) {
                            token.type = JSON_ERROR;
                            return token;
                        }
                        has_decimal = true;
                    }
                    end++;
                    parser->pos++;
                }

                token.type = JSON_NUMBER;
                token.value = (char *)start;
                token.len = end - start;
            } else {
                token.type = JSON_ERROR;
            }
    }

    return token;
}
static int format_request_json(struct llm_config *config,
                             struct llm_message *msg,
                             struct llm_json_buffer *buf) {
    int ret;
    struct llm_message *history_msg;
    size_t required_size = 0;
    bool first_message = true;

    if (!buf || !config || !msg)
        return -EINVAL;

    /* Validate configuration parameters */
    if (!LLM_VALID_MAX_TOKENS(config->max_tokens) ||
        !LLM_VALID_TEMP_RANGE(config->temperature_X100))
        return -EINVAL;

    /* Calculate required buffer size first */
    required_size = strlen("{\"model\":\"") + strlen(config->model) +
                   strlen("\",\"messages\":[]}") + 1;

    /* Lock for thread safety */
    mutex_lock(&config->message_lock);

    /* Calculate size needed for message history */
    list_for_each_entry(history_msg, &config->message_history, list) {
        required_size += strlen(history_msg->role) + strlen(history_msg->content) +
                        50; /* Additional space for JSON formatting */
    }

    /* Add current message size */
    required_size += strlen(msg->role) + strlen(msg->content) + 50;

    /* Initialize buffer with calculated size */
    ret = llm_json_buffer_init(buf, required_size);
    if (ret) {
        mutex_unlock(&config->message_lock);
        return ret;
    }

    /* Start building JSON */
    ret = append_to_json(buf, "{\"model\":\"");
    if (ret)
        goto cleanup;

    ret = append_to_json(buf, config->model);
    if (ret)
        goto cleanup;

    ret = append_to_json(buf, "\",\"messages\":[");
    if (ret)
        goto cleanup;

    /* Add message history */
    list_for_each_entry(history_msg, &config->message_history, list) {
        if (!first_message) {
            ret = append_to_json(buf, ",");
            if (ret)
                goto cleanup;
        }
        first_message = false;

        ret = append_to_json(buf, "{\"role\":\"");
        if (ret)
            goto cleanup;

        ret = append_to_json(buf, history_msg->role);
        if (ret)
            goto cleanup;

        ret = append_to_json(buf, "\",\"content\":\"");
        if (ret)
            goto cleanup;

        ret = append_json_string(buf, history_msg->content);
        if (ret)
            goto cleanup;

        ret = append_to_json(buf, "\"}");
        if (ret)
            goto cleanup;
    }

    /* Add current message */
    if (!first_message) {
        ret = append_to_json(buf, ",");
        if (ret)
            goto cleanup;
    }

    ret = append_to_json(buf, "{\"role\":\"");
    if (ret)
        goto cleanup;

    ret = append_to_json(buf, msg->role);
    if (ret)
        goto cleanup;

    ret = append_to_json(buf, "\",\"content\":\"");
    if (ret)
        goto cleanup;

    ret = append_json_string(buf, msg->content);
    if (ret)
        goto cleanup;

    ret = append_to_json(buf, "\"}]}");
    if (ret)
        goto cleanup;

    mutex_unlock(&config->message_lock);
    return 0;

cleanup:
    mutex_unlock(&config->message_lock);
    llm_json_buffer_free(buf);
    return ret;
}
/* Improved Response Parser */
static int parse_json_value(struct json_parser *parser,
                          const char *key,
                          void *out,
                          int type) {
    struct json_token token;
    char temp[256];
    int ret = 0;

    if (!parser || !key || !out)
        return -EINVAL;

    if (strlen(key) >= 256)  // Prevent overflow
        return -E2BIG;

    token = get_next_token(parser);
    if (token.type == JSON_ERROR)
        return -EINVAL;

    switch (type) {
        case JSON_STRING:
            if (token.type != JSON_STRING)
                return -EINVAL;

            if (token.len >= sizeof(temp) - 1)
                return -E2BIG;

            memcpy(temp, token.value, token.len);
            temp[token.len] = '\0';

            ret = json_unescape_string(temp, out, 256);
            break;

        case JSON_NUMBER:
            if (token.type != JSON_NUMBER)
                return -EINVAL;

            if (token.len >= sizeof(temp) - 1)
                return -E2BIG;

            memcpy(temp, token.value, token.len);
            temp[token.len] = '\0';

            ret = kstrtoint(temp, 10, (int *)out);
            break;

        default:
            return -EINVAL;
    }

    return ret;
}

static int parse_usage_object(struct json_parser *parser,
                            struct llm_response_usage *usage) {
    struct json_token token;
    int ret;

    while ((token = get_next_token(parser)).type != JSON_OBJECT_END) {
        if (token.type != JSON_STRING)
            return -LLM_ERR_JSON_PARSE;

        char key[64];
        if (token.len >= sizeof(key))
            return -E2BIG;

        memcpy(key, token.value, token.len);
        key[token.len] = '\0';

        token = get_next_token(parser);
        if (token.type != JSON_COLON)
            return -LLM_ERR_JSON_PARSE;

        if (strcmp(key, "prompt_tokens") == 0) {
            ret = parse_json_value(parser, key, &usage->prompt_tokens, JSON_NUMBER);
            if (ret < 0)
                return ret;
        } else if (strcmp(key, "completion_tokens") == 0) {
            ret = parse_json_value(parser, key, &usage->completion_tokens, JSON_NUMBER);
            if (ret < 0)
                return ret;
        } else if (strcmp(key, "total_tokens") == 0) {
            ret = parse_json_value(parser, key, &usage->total_tokens, JSON_NUMBER);
            if (ret < 0)
                return ret;
        }
    }

    return 0;
}

static int parse_json_response(const char *buffer, size_t len, struct llm_response *resp) {
    struct json_parser parser;
    int ret;

    if (!buffer || !resp || len == 0)
        return -EINVAL;

    init_json_parser(&parser, buffer, len);

    /* Find start of JSON object */
    struct json_token token = get_next_token(&parser);
    if (token.type != JSON_OBJECT_START)
        return -LLM_ERR_JSON_PARSE;

    /* Parse response fields */
    while ((token = get_next_token(&parser)).type != JSON_OBJECT_END) {
        if (token.type != JSON_STRING)
            return -LLM_ERR_JSON_PARSE;

        char key[256];
        if (token.len >= sizeof(key))
            return -E2BIG;

        memcpy(key, token.value, token.len);
        key[token.len] = '\0';

        token = get_next_token(&parser);
        if (token.type != JSON_COLON)
            return -LLM_ERR_JSON_PARSE;

        if (strcmp(key, "id") == 0) {
            ret = parse_json_value(&parser, key, resp->id, JSON_STRING);
            if (ret < 0)
                return ret;
        } else if (strcmp(key, "model") == 0) {
            ret = parse_json_value(&parser, key, resp->model, JSON_STRING);
            if (ret < 0)
                return ret;
        } else if (strcmp(key, "usage") == 0) {
            /* Parse usage object */
            token = get_next_token(&parser);
            if (token.type != JSON_OBJECT_START)
                return -LLM_ERR_JSON_PARSE;

            ret = parse_usage_object(&parser, &resp->usage);
            if (ret < 0)
                return ret;
        }
    }

    return 0;
}
//////////////////////////////////////////////////////////
/* Initialize message queue */
static int init_message_queue(struct llm_message_queue *queue, size_t max_size) {
    if (!queue)
        return -EINVAL;

    INIT_LIST_HEAD(&queue->messages);
    spin_lock_init(&queue->queue_lock);
    init_waitqueue_head(&queue->wait_queue);
    atomic_set(&queue->count, 0);
    queue->max_size = max_size;
    atomic_set(&queue->is_active, 1);

    return 0;
}

/* Enqueue message with priority */
static int enqueue_message(struct llm_message_queue *queue,
                         struct llm_message *msg,
                         int priority) {
    struct queue_item *item;
    struct queue_item *pos;
    struct list_head *insert_pos;
    unsigned long flags;

    if (!queue || !msg)
        return -EINVAL;

    if (atomic_read(&queue->count) >= queue->max_size)
        return -ENOSPC;

    item = llm_kmalloc(sizeof(*item));
    if (!item)
        return -ENOMEM;

    item->msg = msg;
    item->timestamp = jiffies;
    item->priority = priority;

    spin_lock_irqsave(&queue->queue_lock, flags);

    /* Find insertion point based on priority */
    insert_pos = &queue->messages;
    list_for_each_entry(pos, &queue->messages, list) {
        if (priority > pos->priority) {
            insert_pos = &pos->list;
            break;
        }
    }

    list_add_tail(&item->list, insert_pos);
    atomic_inc(&queue->count);

    spin_unlock_irqrestore(&queue->queue_lock, flags);

    /* Wake up waiting consumers */
    wake_up(&queue->wait_queue);

    return 0;
}

/* Dequeue message with timeout */
static struct llm_message *dequeue_message(struct llm_message_queue *queue,
                                         unsigned long timeout_ms) {
    struct queue_item *item;
    struct llm_message *msg = NULL;
    unsigned long flags;
    int ret;

    if (!queue)
        return NULL;

    if (timeout_ms) {
        ret = wait_event_interruptible_timeout(queue->wait_queue,
                                             atomic_read(&queue->count) > 0 ||
                                             !atomic_read(&queue->is_active),
                                             msecs_to_jiffies(timeout_ms));
        if (ret <= 0)
            return NULL;
    }

    spin_lock_irqsave(&queue->queue_lock, flags);

    if (!list_empty(&queue->messages)) {
        item = list_first_entry(&queue->messages, struct queue_item, list);
        list_del(&item->list);
        atomic_dec(&queue->count);
        msg = item->msg;
        llm_kfree(item);
    }

    spin_unlock_irqrestore(&queue->queue_lock, flags);

    return msg;
}

/* Cleanup message queue */
static void cleanup_message_queue(struct llm_message_queue *queue) {
    struct queue_item *item, *tmp;
    unsigned long flags;

    if (!queue)
        return;

    atomic_set(&queue->is_active, 0);
    wake_up_all(&queue->wait_queue);

    spin_lock_irqsave(&queue->queue_lock, flags);

    list_for_each_entry_safe(item, tmp, &queue->messages, list) {
        list_del(&item->list);
        llm_message_cleanup(item->msg);
        llm_kfree(item);
    }

    atomic_set(&queue->count, 0);
    spin_unlock_irqrestore(&queue->queue_lock, flags);
}


/* Initialize response handler */
static int init_response_handler(struct llm_response_handler *handler,
                               struct llm_response *resp,
                               size_t max_size,
                               bool streaming) {
    if (!handler || !resp)
        return -EINVAL;

    handler->resp = resp;
    handler->max_size = max_size;
    handler->streaming = streaming;
    atomic_set(&handler->is_complete, 0);
    spin_lock_init(&handler->handler_lock);

    /* Initialize buffer */
    handler->buffer = llm_kmalloc(sizeof(*handler->buffer));
    if (!handler->buffer)
        return -ENOMEM;

    return llm_json_buffer_init(handler->buffer, max_size);
}

/* Process response chunks */
static int process_response_chunk(struct llm_response_handler *handler,
                                const char *chunk,
                                size_t chunk_size) {
    int ret = 0;
    unsigned long flags;

    if (!handler || !chunk)
        return -EINVAL;

    spin_lock_irqsave(&handler->handler_lock, flags);

    /* Check buffer space */
    if (handler->buffer->used + chunk_size > handler->max_size) {
        ret = -E2BIG;
        goto unlock;
    }

    /* Append chunk to buffer */
    ret = append_to_json(handler->buffer, chunk);
    if (ret)
        goto unlock;

    /* Process if streaming or complete chunk */
    if (handler->streaming || strstr(chunk, "\n\n")) {
        ret = parse_json_response(handler->buffer->data,
                                handler->buffer->used,
                                handler->resp);
        if (ret == 0 && handler->callback) {
            handler->callback(handler->resp, handler->callback_data);
        }

        /* Reset buffer if streaming */
        if (handler->streaming) {
            handler->buffer->used = 0;
            handler->buffer->data[0] = '\0';
        }
    }

unlock:
    spin_unlock_irqrestore(&handler->handler_lock, flags);
    return ret;
}

/* Complete response handling */
static int complete_response(struct llm_response_handler *handler) {
    int ret = 0;
    unsigned long flags;

    if (!handler)
        return -EINVAL;

    spin_lock_irqsave(&handler->handler_lock, flags);

    if (handler->buffer->used > 0) {
        ret = parse_json_response(handler->buffer->data,
                                handler->buffer->used,
                                handler->resp);
        if (ret == 0 && handler->callback) {
            handler->callback(handler->resp, handler->callback_data);
        }
    }

    atomic_set(&handler->is_complete, 1);
    spin_unlock_irqrestore(&handler->handler_lock, flags);

    return ret;
}

/* Clean up response handler */
static void cleanup_response_handler(struct llm_response_handler *handler) {
    if (!handler)
        return;

    if (handler->buffer) {
        llm_json_buffer_free(handler->buffer);
        llm_kfree(handler->buffer);
    }

    memzero_explicit(handler, sizeof(*handler));
}


/* Initialize rate limiter */
static int init_rate_limiter(struct llm_rate_limiter *limiter,
                           unsigned long tokens_per_min) {
    if (!limiter)
        return -EINVAL;

    atomic_set(&limiter->tokens, tokens_per_min);
    atomic_set(&limiter->max_tokens, tokens_per_min);
    atomic64_set(&limiter->last_refill, ktime_get_ms());
    spin_lock_init(&limiter->limiter_lock);
    limiter->refill_interval_ms = 60000; // 1 minute
    limiter->tokens_per_interval = tokens_per_min;
    atomic_set(&limiter->is_limited, 0);

    return 0;
}

/* Refill tokens */
static void refill_tokens(struct llm_rate_limiter *limiter) {
    unsigned long now = ktime_get_ms();
    unsigned long last_refill;
    unsigned long elapsed;
    int tokens_to_add;
    unsigned long flags;

    spin_lock_irqsave(&limiter->limiter_lock, flags);

    last_refill = atomic64_read(&limiter->last_refill);
    elapsed = now - last_refill;

    if (elapsed >= limiter->refill_interval_ms) {
        /* Reset tokens to max */
        atomic_set(&limiter->tokens, atomic_read(&limiter->max_tokens));
        atomic64_set(&limiter->last_refill, now);
        atomic_set(&limiter->is_limited, 0);
    } else if (elapsed > 0) {
        /* Proportional refill */
        tokens_to_add = (elapsed * limiter->tokens_per_interval) /
                       limiter->refill_interval_ms;
        if (tokens_to_add > 0) {
            int current = atomic_read(&limiter->tokens);
            int max = atomic_read(&limiter->max_tokens);
            int new_tokens = min(current + tokens_to_add, max);
            atomic_set(&limiter->tokens, new_tokens);
            atomic64_set(&limiter->last_refill,
                        last_refill + (tokens_to_add * limiter->refill_interval_ms) /
                        limiter->tokens_per_interval);

            if (new_tokens > 0) {
                atomic_set(&limiter->is_limited, 0);
            }
        }
    }

    spin_unlock_irqrestore(&limiter->limiter_lock, flags);
}

/* Check and consume token */
static int consume_token(struct llm_rate_limiter *limiter) {
    int tokens;

    refill_tokens(limiter);

    tokens = atomic_read(&limiter->tokens);
    if (tokens <= 0) {
        atomic_set(&limiter->is_limited, 1);
        return -LLM_ERR_RATE_LIMIT;
    }

    if (atomic_dec_return(&limiter->tokens) < 0) {
        atomic_inc(&limiter->tokens);
        atomic_set(&limiter->is_limited, 1);
        return -LLM_ERR_RATE_LIMIT;
    }

    return 0;
}

/* Get wait time if rate limited */
static unsigned long get_rate_limit_wait_time(struct llm_rate_limiter *limiter) {
    unsigned long now = ktime_get_ms();
    unsigned long last_refill = atomic64_read(&limiter->last_refill);
    unsigned long elapsed = now - last_refill;

    if (elapsed >= limiter->refill_interval_ms)
        return 0;

    return limiter->refill_interval_ms - elapsed;
}
/* Initialize error tracking */
/* Safe configuration update */
static int update_config_safe(struct llm_config *config,
                            const char *key,
                            const void *value,
                            size_t size) {
    int ret = 0;
    bool found = false;
    int i;

    if (!config || !key || !value)
        return -EINVAL;

    mutex_lock(&config->config_lock);

    /* Find and run validator */
    for (i = 0; i < ARRAY_SIZE(config_validators); i++) {
        if (strcmp(config_validators[i].name, key) == 0) {
            ret = config_validators[i].validate(value, size);
            found = true;
            break;
        }
    }

    if (!found) {
        ret = -EINVAL;
        goto unlock;
    }

    if (ret != 0)
        goto unlock;

    /* Update configuration */
    ret = update_config_value(config, key, value, size);
    if (ret == 0) {
        atomic_set(&config->state.is_modified, 1);
        config->state.last_modified = jiffies;
        strlcpy(config->state.last_modifier, current->comm,
                sizeof(config->state.last_modifier));
    }

unlock:
    mutex_unlock(&config->config_lock);
    return ret;
}

/* Configuration initialization */
static int init_config_safe(struct llm_config *config) {
    int ret;

    if (!config)
        return -EINVAL;

    /* Initialize state tracking */
    atomic_set(&config->state.is_initialized, 0);
    atomic_set(&config->state.is_modified, 0);
    atomic_set(&config->state.ref_count, 1);
    mutex_init(&config->state.state_lock);

    /* Initialize locks */
    mutex_init(&config->config_lock);
    mutex_init(&config->message_lock);
    mutex_init(&config->tool_lock);

    /* Initialize rate limiting */
    spin_lock_init(&config->rate_limit.lock);
    atomic_set(&config->rate_limit.requests_remaining,
               config->max_requests_per_min);

    /* Validate initial configuration */
    ret = validate_full_config(config);
    if (ret != 0)
        return ret;

    atomic_set(&config->state.is_initialized, 1);
    return 0;
}

/* Configuration backup/restore */
static int backup_config(struct llm_config *config,
                        char *buffer,
                        size_t size) {
    int ret;

    mutex_lock(&config->config_lock);
    ret = serialize_config(config, buffer, size);
    mutex_unlock(&config->config_lock);

    return ret;
}

static int restore_config(struct llm_config *config,
                         const char *buffer,
                         size_t size) {
    int ret;

    mutex_lock(&config->config_lock);
    ret = deserialize_config(config, buffer, size);
    if (ret == 0) {
        atomic_set(&config->state.is_modified, 1);
        config->state.last_modified = jiffies;
        strlcpy(config->state.last_modifier, "restore",
                sizeof(config->state.last_modifier));
    }
    mutex_unlock(&config->config_lock);

    return ret;
}



static void llm_error_init(void) {
    atomic_set(&error_state.error_count, 0);
    atomic64_set(&error_state.first_error_time, 0);
    atomic64_set(&error_state.last_error_time, 0);
    spin_lock_init(&error_state.error_lock);
    error_state.last_error = 0;
    error_state.last_error_func[0] = '\0';
    error_state.last_error_line = 0;
}

/* Log error with context */
static void llm_log_error(enum llm_log_level level, const char *func, int line,
                         int error, const char *fmt, ...) {
    va_list args;
    char buf[256];
    unsigned long flags;

    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    switch (level) {
        case LLM_LOG_ERROR:
            pr_err("LLM ERROR [%s:%d]: %s (err=%d)\n", func, line, buf, error);
            break;
        case LLM_LOG_WARN:
            pr_warn("LLM WARN [%s:%d]: %s\n", func, line, buf);
            break;
        case LLM_LOG_INFO:
            pr_info("LLM INFO [%s:%d]: %s\n", func, line, buf);
            break;
        case LLM_LOG_DEBUG:
            pr_debug("LLM DEBUG [%s:%d]: %s\n", func, line, buf);
            break;
    }

    if (level == LLM_LOG_ERROR) {
        spin_lock_irqsave(&error_state.error_lock, flags);

        /* Update error state */
        if (atomic_read(&error_state.error_count) == 0) {
            atomic64_set(&error_state.first_error_time, ktime_get_ms());
        }

        atomic_inc(&error_state.error_count);
        atomic64_set(&error_state.last_error_time, ktime_get_ms());
        error_state.last_error = error;
        strlcpy(error_state.last_error_func, func, sizeof(error_state.last_error_func));
        error_state.last_error_line = line;

        spin_unlock_irqrestore(&error_state.error_lock, flags);
    }
}

/* Check if we should trigger circuit breaker */
static bool should_trigger_circuit_breaker(void) {
    unsigned long flags;
    bool trigger = false;
    u64 now = ktime_get_ms();

    spin_lock_irqsave(&error_state.error_lock, flags);

    if (atomic_read(&error_state.error_count) > LLM_MAX_ERROR_RETRIES) {
        u64 first_error = atomic64_read(&error_state.first_error_time);
        if (now - first_error < LLM_ERROR_WINDOW_MS) {
            trigger = true;
        }
    }

    spin_unlock_irqrestore(&error_state.error_lock, flags);
    return trigger;
}
static void reset_error_state(void) {
    unsigned long flags;

    spin_lock_irqsave(&error_state.error_lock, flags);
    atomic_set(&error_state.error_count, 0);
    atomic64_set(&error_state.first_error_time, 0);
    atomic64_set(&error_state.last_error_time, 0);
    error_state.last_error = 0;
    error_state.last_error_func[0] = '\0';
    error_state.last_error_line = 0;
    spin_unlock_irqrestore(&error_state.error_lock, flags);
}
static int llm_retry_operation(struct retry_context *ctx,
                             int (*operation)(void *data),
                             void *data) {
    int ret;

    while (ctx->current_retry < ctx->max_retries) {
        ret = operation(data);
        if (ret == 0) {
            /* Success */
            if (ctx->current_retry > 0) {
                llm_log_info("Operation '%s' succeeded after %d retries",
                            ctx->operation, ctx->current_retry);
            }
            return 0;
        }

        ctx->current_retry++;

        if (should_trigger_circuit_breaker()) {
            llm_log_error(LLM_LOG_ERROR, __func__, __LINE__,
                         ret, "Circuit breaker triggered for '%s'",
                         ctx->operation);
            return -ECONNABORTED;
        }

        if (ctx->current_retry < ctx->max_retries) {
            /* Exponential backoff with jitter */
            unsigned long delay = min(ctx->delay_ms * (1 << ctx->current_retry),
                                    ctx->max_delay_ms);
            delay += (get_random_int() % (delay / 4));
            msleep(delay);
        }
    }

    return ret;
}

static enum llm_error_category categorize_error(int error) {
    switch (error) {
        case -ENOMEM:
            return LLM_ERR_CAT_MEMORY;
        case -ECONNREFUSED:
        case -ETIMEDOUT:
        case -ENETUNREACH:
            return LLM_ERR_CAT_NETWORK;
        case -LLM_ERR_API_RESPONSE:
        case -LLM_ERR_RATE_LIMIT:
            return LLM_ERR_CAT_API;
        case -LLM_ERR_SSL:
            return LLM_ERR_CAT_SECURITY;
        default:
            return LLM_ERR_CAT_INTERNAL;
    }
}

/* Error recovery strategies */
static int handle_error_with_recovery(int error, void *context) {
    enum llm_error_category category = categorize_error(error);
    struct llm_config *config = (struct llm_config *)context;
    int ret = error;

    switch (category) {
        case LLM_ERR_CAT_NETWORK:
            /* Try to re-establish connection */
            ret = reestablish_connection(config);
            break;

        case LLM_ERR_CAT_API:
            if (error == -LLM_ERR_RATE_LIMIT) {
                /* Wait for rate limit reset */
                unsigned long reset_time = atomic64_read(&config->rate_limit_reset);
                unsigned long wait_time = jiffies_to_msecs(reset_time - jiffies);
                msleep(wait_time);
                ret = 0;
            }
            break;

        case LLM_ERR_CAT_MEMORY:
            /* Trigger memory cleanup */
            trigger_memory_cleanup();
            break;

        case LLM_ERR_CAT_SECURITY:
            /* Reset SSL/TLS connection */
            ret = reset_tls_connection(config);
            break;

        default:
            /* Log unhandled error */
            llm_log_error(LLM_LOG_ERROR, __func__, __LINE__,
                         error, "Unhandled error category");
            break;
    }

    return ret;
}


/* Initialize connection pool */
static int init_connection_pool(void) {
    int i;

    conn_pool = llm_kmalloc(sizeof(*conn_pool));
    if (!conn_pool)
        return -ENOMEM;

    spin_lock_init(&conn_pool->pool_lock);
    atomic_set(&conn_pool->total_conns, 0);

    for (i = 0; i < MAX_CONN_POOL_SIZE; i++) {
        atomic_set(&conn_pool->conn_refs[i], 0);
        conn_pool->connections[i] = NULL;
    }

    return 0;
}

/* Get connection from pool */
static struct llm_connection *get_connection(struct llm_config *config) {
    struct llm_connection *conn = NULL;
    unsigned long flags;
    int i, found = -1;

    spin_lock_irqsave(&conn_pool->pool_lock, flags);

    /* Try to find existing connection */
    for (i = 0; i < MAX_CONN_POOL_SIZE; i++) {
        if (conn_pool->connections[i] &&
            atomic_read(&conn_pool->conn_refs[i]) == 0) {
            found = i;
            break;
        }
    }

    /* If no existing connection, try to create new one */
    if (found == -1) {
        for (i = 0; i < MAX_CONN_POOL_SIZE; i++) {
            if (!conn_pool->connections[i]) {
                found = i;
                break;
            }
        }
    }

    if (found >= 0) {
        atomic_inc(&conn_pool->conn_refs[found]);
        conn = conn_pool->connections[found];
    }

    spin_unlock_irqrestore(&conn_pool->pool_lock, flags);

    /* Create new connection if needed */
    if (found >= 0 && !conn) {
        conn = llm_kmalloc(sizeof(*conn));
        if (!conn)
            return ERR_PTR(-ENOMEM);

        ret = establish_tls_connection(config, conn);
        if (ret < 0) {
            llm_kfree(conn);
            atomic_dec(&conn_pool->conn_refs[found]);
            return ERR_PTR(ret);
        }

        spin_lock_irqsave(&conn_pool->pool_lock, flags);
        conn_pool->connections[found] = conn;
        atomic_inc(&conn_pool->total_conns);
        spin_unlock_irqrestore(&conn_pool->pool_lock, flags);
    }

    return conn ? conn : ERR_PTR(-EBUSY);
}

/* Establish TLS connection */
static int establish_tls_connection(struct llm_config *config,
                                  struct llm_connection *conn) {
    struct tls_context *tls;
    int ret;

    /* Initialize connection */
    memset(conn, 0, sizeof(*conn));
    conn->config = config;
    conn->state = CONN_STATE_INIT;
    conn->timeout_ms = config->timeout_ms;
    spin_lock_init(&conn->lock);

    /* Create socket */
    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM,
                          IPPROTO_TCP, &conn->sock);
    if (ret < 0)
        return ret;

    /* Set socket options */
    ret = set_sock_opts(conn->sock, conn->timeout_ms);
    if (ret < 0)
        goto cleanup_sock;

    /* Setup TLS */
    tls = kzalloc(sizeof(*tls), GFP_KERNEL);
    if (!tls) {
        ret = -ENOMEM;
        goto cleanup_sock;
    }

    /* Initialize TLS context */
    ret = tls_ctx_init(tls);
    if (ret < 0)
        goto cleanup_tls;

    /* Connect socket */
    ret = connect_socket(conn);
    if (ret < 0)
        goto cleanup_tls;

    /* Start TLS handshake */
    ret = tls_do_handshake(tls);
    if (ret < 0)
        goto cleanup_tls;

    conn->tls = tls;
    conn->state = CONN_STATE_CONNECTED;
    return 0;

cleanup_tls:
    kfree(tls);
cleanup_sock:
    sock_release(conn->sock);
    return ret;
}

/* SSL Session Implementation */

/* Initialize memory tracker */
static inline void llm_mem_init(void) {
    atomic_set(&mem_tracker.alloc_count, 0);
    atomic_set(&mem_tracker.free_count, 0);
    atomic64_set(&mem_tracker.total_bytes, 0);
    spin_lock_init(&mem_tracker.track_lock);
    INIT_LIST_HEAD(&mem_tracker.alloc_list);
}

/* Secure allocation with tracking */
static inline void *llm_malloc(size_t size, const char *func, int line) {
    void *ptr;
    struct llm_mem_block *block;
    unsigned long flags;

    if (size == 0)
        return NULL;

    /* Allocate memory block tracker */
    block = kmalloc(sizeof(*block), GFP_KERNEL);
    if (!block)
        return NULL;

    /* Allocate requested memory */
    ptr = kzalloc(size, GFP_KERNEL);
    if (!ptr) {
        kfree(block);
        return NULL;
    }

    /* Initialize tracking info */
    block->ptr = ptr;
    block->size = size;
    block->func = func;
    block->line = line;

    /* Update statistics */
    spin_lock_irqsave(&mem_tracker.track_lock, flags);
    atomic_inc(&mem_tracker.alloc_count);
    atomic64_add(size, &mem_tracker.total_bytes);
    list_add(&block->list, &mem_tracker.alloc_list);
    spin_unlock_irqrestore(&mem_tracker.track_lock, flags);

    return ptr;
}

/* Secure free with tracking */
static inline void llm_free(void *ptr) {
    struct llm_mem_block *block, *tmp;
    unsigned long flags;

    if (!ptr)
        return;

    spin_lock_irqsave(&mem_tracker.track_lock, flags);
    list_for_each_entry_safe(block, tmp, &mem_tracker.alloc_list, list) {
        if (block->ptr == ptr) {
            /* Update statistics */
            atomic_inc(&mem_tracker.free_count);
            atomic64_sub(block->size, &mem_tracker.total_bytes);

            /* Remove from tracking list */
            list_del(&block->list);

            /* Secure cleanup */
            memzero_explicit(ptr, block->size);
            kfree(ptr);
            kfree(block);
            break;
        }
    }
    spin_unlock_irqrestore(&mem_tracker.track_lock, flags);
}

/* Memory leak detection */
static void llm_check_leaks(void) {
    struct llm_mem_block *block;
    unsigned long flags;
    int leak_count = 0;

    spin_lock_irqsave(&mem_tracker.track_lock, flags);

    list_for_each_entry(block, &mem_tracker.alloc_list, list) {
        pr_err("LLM: Memory leak detected: %zu bytes allocated in %s:%d\n",
               block->size, block->func, block->line);
        leak_count++;
    }

    if (leak_count > 0) {
        pr_err("LLM: Total memory leaks: %d\n", leak_count);
        pr_err("LLM: Allocations: %d, Frees: %d, Total bytes: %lld\n",
               atomic_read(&mem_tracker.alloc_count),
               atomic_read(&mem_tracker.free_count),
               atomic64_read(&mem_tracker.total_bytes));
    }

    spin_unlock_irqrestore(&mem_tracker.track_lock, flags);
}

#define llm_kmalloc(size) llm_malloc(size, __func__, __LINE__)
#define llm_kfree(ptr) llm_free(ptr)
static int init_ssl_session(struct ssl_context *ssl) {
    int ret;
    struct crypto_aead *tfm;

    if (!ssl)
        return -EINVAL;

    /* Initialize AEAD transform */
    tfm = crypto_alloc_aead("gcm(aes)", 0, CRYPTO_ALG_ASYNC);
    if (IS_ERR(tfm))
        return PTR_ERR(tfm);

    ssl->tfm = tfm;
    ssl->key_size = crypto_aead_keysize(tfm);
    ssl->iv_size = crypto_aead_ivsize(tfm);

    /* Allocate key and IV */
    ssl->key = kzalloc(ssl->key_size, GFP_KERNEL);
    ssl->iv = kzalloc(ssl->iv_size, GFP_KERNEL);
    if (!ssl->key || !ssl->iv) {
        ret = -ENOMEM;
        goto cleanup;
    }

    /* Generate random key material */
    get_random_bytes(ssl->key, ssl->key_size);
    get_random_bytes(ssl->iv, ssl->iv_size);

    /* Set key */
    ret = crypto_aead_setkey(ssl->tfm, ssl->key, ssl->key_size);
    if (ret)
        goto cleanup;

    /* Initialize scatterlists */
    ssl->sg_tx = kcalloc(2, sizeof(struct scatterlist), GFP_KERNEL);
    ssl->sg_rx = kcalloc(2, sizeof(struct scatterlist), GFP_KERNEL);
    if (!ssl->sg_tx || !ssl->sg_rx) {
        ret = -ENOMEM;
        goto cleanup;
    }

    sg_init_table(ssl->sg_tx, 2);
    sg_init_table(ssl->sg_rx, 2);

    /* Allocate request */
    ssl->req = aead_request_alloc(ssl->tfm, GFP_KERNEL);
    if (!ssl->req) {
        ret = -ENOMEM;
        goto cleanup;
    }

    return 0;

cleanup:
    if (ssl->req)
        aead_request_free(ssl->req);
    kfree(ssl->sg_tx);
    kfree(ssl->sg_rx);
    kfree(ssl->key);
    kfree(ssl->iv);
    crypto_free_aead(ssl->tfm);
    return ret;
}

/* Helper Functions */



static int check_rate_limit(struct llm_config *config) {
    unsigned long now = jiffies;
    int ret = 0;
    unsigned long old_reset;

    spin_lock(&config->rate_limit.lock);
    old_reset = atomic64_read(&config->rate_limit_reset);

    if (time_after(now, old_reset)) {
        atomic_set(&config->rate_limit.requests_remaining,
                  config->max_requests_per_min);
        atomic64_set(&config->rate_limit_reset, now + HZ * 60);
    }

    if (atomic_read(&config->rate_limit.requests_remaining) <= 0)
        ret = -LLM_ERR_RATE_LIMIT;
    else
        atomic_dec(&config->rate_limit.requests_remaining);

    spin_unlock(&config->rate_limit.lock);
    return ret;
}


static int validate_message(const struct llm_message *msg)
{
    if (!msg || !msg->content)
        return -EINVAL;

    if (msg->content_length > MAX_PROMPT_LENGTH)
        return -E2BIG;

    /* Validate role */
    if (strcmp(msg->role, ROLE_SYSTEM) != 0 &&
        strcmp(msg->role, ROLE_USER) != 0 &&
        strcmp(msg->role, ROLE_ASSISTANT) != 0 &&
        strcmp(msg->role, ROLE_TOOL) != 0)
        return -EINVAL;

    return 0;
}





static void cleanup_connection(struct llm_connection *conn) {
    if (!conn) return;

    if (conn->ssl_enabled && conn->ssl_context) {
        struct ssl_context *ssl = conn->ssl_context;
        crypto_free_aead(ssl->tfm);
        kfree(ssl);
    }

    if (conn->sock) {
        kernel_sock_shutdown(conn->sock, SHUT_RDWR);
        sock_release(conn->sock);
    }
}
static int establish_ssl_connection(struct llm_connection *conn) {
    int ret;
    struct ssl_context *ssl = NULL;

    if (!conn) {
        return -EINVAL;
    }

    /* Allocate SSL context with proper cleanup tracking */
    ssl = kzalloc(sizeof(*ssl), GFP_KERNEL);
    if (!ssl) {
        return -ENOMEM;
    }

    /* Initialize crypto with proper error handling */
    ssl->tfm = crypto_alloc_aead("gcm(aes)", 0, CRYPTO_ALG_ASYNC);
    if (IS_ERR(ssl->tfm)) {
        ret = PTR_ERR(ssl->tfm);
        goto cleanup_ssl;
    }

    /* Get required sizes */
    ssl->key_size = crypto_aead_keysize(ssl->tfm);
    ssl->iv_size = crypto_aead_ivsize(ssl->tfm);

    /* Allocate key and IV with proper cleanup */
    ssl->key = kmalloc(ssl->key_size, GFP_KERNEL);
    ssl->iv = kmalloc(ssl->iv_size, GFP_KERNEL);
    if (!ssl->key || !ssl->iv) {
        ret = -ENOMEM;
        goto cleanup_crypto;
    }

    /* Generate random key material */
    get_random_bytes(ssl->key, ssl->key_size);
    get_random_bytes(ssl->iv, ssl->iv_size);

    /* Set key with error handling */
    ret = crypto_aead_setkey(ssl->tfm, ssl->key, ssl->key_size);
    if (ret) {
        goto cleanup_key_iv;
    }

    /* Allocate and initialize scatterlists */
    ssl->sg_tx = kcalloc(2, sizeof(struct scatterlist), GFP_KERNEL);
    ssl->sg_rx = kcalloc(2, sizeof(struct scatterlist), GFP_KERNEL);
    if (!ssl->sg_tx || !ssl->sg_rx) {
        ret = -ENOMEM;
        goto cleanup_key_iv;
    }

    sg_init_table(ssl->sg_tx, 2);
    sg_init_table(ssl->sg_rx, 2);

    /* Allocate request */
    ssl->req = aead_request_alloc(ssl->tfm, GFP_KERNEL);
    if (!ssl->req) {
        ret = -ENOMEM;
        goto cleanup_scatterlists;
    }

    conn->ssl_context = ssl;
    return 0;

cleanup_scatterlists:
    kfree(ssl->sg_tx);
    kfree(ssl->sg_rx);
cleanup_key_iv:
    if (ssl->key) {
        memzero_explicit(ssl->key, ssl->key_size);
        kfree(ssl->key);
    }
    if (ssl->iv) {
        memzero_explicit(ssl->iv, ssl->iv_size);
        kfree(ssl->iv);
    }
cleanup_crypto:
    crypto_free_aead(ssl->tfm);
cleanup_ssl:
    kfree(ssl);
    return ret;
}
static int establish_connection(struct llm_config *config,
                              struct llm_connection *conn) {
    struct socket *sock;
    int ret;

    if (!config || !conn)
        return -EINVAL;

    /* Initialize connection structure */
    memset(conn, 0, sizeof(*conn));
    conn->config = config;
    conn->timeout_ms = config->timeout_ms;
    spin_lock_init(&conn->lock);
    atomic_set(&conn->ref_count, 1);
    conn->state = CONN_STATE_INIT;

    /* Create socket */
    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM,
                          IPPROTO_TCP, &sock);
    if (ret < 0)
        return ret;

    /* Set socket options */
    ret = set_socket_options(sock, conn->timeout_ms);
    if (ret < 0)
        goto cleanup_socket;

    /* Connect to server */
    ret = connect_to_server(sock, config->endpoint);
    if (ret < 0)
        goto cleanup_socket;

    /* Setup TLS if enabled */
    if (config->use_ssl) {
        ret = setup_tls(sock);
        if (ret < 0)
            goto cleanup_socket;
    }

    conn->sock = sock;
    conn->state = CONN_STATE_CONNECTED;
    conn->last_used = jiffies;

    return 0;

cleanup_socket:
    sock_release(sock);
    return ret;
}
static int establish_connection_with_timeout(struct llm_config *config,
                                           struct llm_connection *conn) {
    int ret;
    unsigned long timeout = msecs_to_jiffies(config->timeout_ms);
    struct socket *sock = conn->sock;

    /* Set socket timeout */
    ret = sock_setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
                         (char *)&timeout, sizeof(timeout));
    if (ret < 0)
        return ret;

    ret = sock_setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                         (char *)&timeout, sizeof(timeout));
    if (ret < 0)
        return ret;

    /* Modify establish_connection to use timeout */
    ret = establish_connection(config, conn);
    if (ret)
        return ret;

    return 0;
}


int llm_set_api_key(struct llm_config *config, const char *key) {
    if (!config || !key)
        return -EINVAL;

    if (strlen(key) >= MAX_API_KEY_LENGTH)
        return -E2BIG;

    mutex_lock(&config->config_lock);
    strncpy(config->api_key, key, MAX_API_KEY_LENGTH - 1);
    config->api_key[MAX_API_KEY_LENGTH - 1] = '\0';
    mutex_unlock(&config->config_lock);

    return 0;
}

static int send_http_request(struct llm_connection *conn,
                           struct llm_json_buffer *buf) {
    struct msghdr msg = {0};
    struct kvec iov[2];
    char headers[512];
    int ret;

     ret = snprintf(headers, sizeof(headers),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: api.openai.com\r\n"
        "Authorization: Bearer %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "\r\n", conn->config->api_key, buf->used);

    if (ret >= sizeof(headers))
        return -EOVERFLOW;

    iov[0].iov_base = headers;
    iov[0].iov_len = ret;
    iov[1].iov_base = buf->data;
    iov[1].iov_len = buf->used;

    ret = kernel_sendmsg(conn->sock, &msg, iov, 2, ret + buf->used);
    if (ret < 0)
        return ret;

    return 0;
}
static int receive_http_response(struct llm_connection *conn,
                               struct llm_response *resp,
                               char *buffer,
                               size_t buffer_size) {
    struct msghdr msg = {0};
    struct kvec iov;
    int ret, received = 0;
    bool headers_complete = false;
    long timeout = msecs_to_jiffies(conn->timeout_ms);

    if (!conn || !resp || !buffer || buffer_size < 1)
        return -EINVAL;

    while (received < buffer_size - 1) {
        /* Set socket timeout */
        ret = sock_setsockopt(conn->sock, SOL_SOCKET, SO_RCVTIMEO,
                             (char *)&timeout, sizeof(timeout));
        if (ret < 0)
            return ret;

        iov.iov_base = buffer + received;
        iov.iov_len = min_t(size_t, buffer_size - received - 1, PAGE_SIZE);

        ret = kernel_recvmsg(conn->sock, &msg, &iov, 1, iov.iov_len, 0);
        if (ret <= 0) {
            if (ret == -EAGAIN)
                return -LLM_ERR_TIMEOUT;
            return ret ? ret : -ECONNRESET;
        }

        received += ret;
        buffer[received] = '\0';

        /* Process headers and body */
        if (!headers_complete && received > 4) {
            char *body = strstr(buffer, "\r\n\r\n");
            if (body) {
                headers_complete = true;
                /* Validate response code */
                if (sscanf(buffer, "HTTP/1.1 %d", &resp->status_code) != 1)
                    return -LLM_ERR_API_RESPONSE;

                if (resp->status_code < 200 || resp->status_code >= 300)
                    return -LLM_ERR_API_RESPONSE;

                /* Process body */
                size_t header_len = (body - buffer) + 4;
                memmove(buffer, body + 4, received - header_len);
                received -= header_len;
            }
        }

        /* Check for complete response */
        if (headers_complete && strchr(buffer, '}'))
            break;
    }

    if (received >= buffer_size - 1)
        return -E2BIG;

    return received;
}

void llm_cleanup(void)
{
    mutex_lock(&llm_mutex);

    if (global_config) {
        cleanup_message_history(global_config);
        cleanup_tools(global_config);
        global_config = NULL;
    }

    mutex_unlock(&llm_mutex);
    pr_info("LLM: Cleanup complete\n");
}

/* Message Sending and Response Handling */
int llm_send_message(struct llm_config *config, struct llm_message *msg) {
    struct llm_connection conn = {0};
    struct llm_json_buffer req_buf = {0};
    int ret;

    if (!config || !msg)
        return -EINVAL;

    ret = validate_message(msg);
    if (ret != 0)
        return ret;

    ret = check_rate_limit(config);
    if (ret != 0)
        return ret;

    ret = format_request_json(config, msg, &req_buf);
    if (ret != 0)
        return ret;

    ret = establish_connection(config, &conn);
    if (ret != 0)
        goto cleanup_buf;

    ret = send_http_request(&conn, &req_buf);

cleanup_conn:
    cleanup_connection(&conn);
cleanup_buf:
    llm_json_buffer_free(&req_buf);
    return ret;
}

int llm_receive_response(struct llm_config *config, struct llm_response *resp) {
    struct llm_connection conn;
    char *buffer;
    int ret;

    if (!config || !resp)
        return -EINVAL;

    buffer = kmalloc(MAX_RESPONSE_LENGTH, GFP_KERNEL);
    if (!buffer)
        return -ENOMEM;

    /* Establish connection */
    ret = establish_connection(config, &conn);
    if (ret)
        goto cleanup_buffer;

    /* Receive response with timeout */
    ret = receive_http_response(&conn, resp, buffer, MAX_RESPONSE_LENGTH);
    if (ret < 0)
        goto cleanup_conn;

    /* Parse JSON response */
    ret = parse_json_response(buffer, ret, resp);

cleanup_conn:
    cleanup_connection(&conn);
cleanup_buffer:
    memzero_explicit(buffer, MAX_RESPONSE_LENGTH);
    kfree(buffer);
    return ret;
}

static int __init llm_provider_init(void) {
    int ret;
    struct llm_config *config;

    /* Allocate and initialize global config */
    config = kzalloc(sizeof(*config), GFP_KERNEL);
    if (!config)
        return -ENOMEM;

    /* Initialize mutexes */
    mutex_init(&config->config_lock);
    mutex_init(&config->message_lock);
    mutex_init(&config->tool_lock);
    spin_lock_init(&config->rate_limit.lock);

    /* Initialize lists */
    INIT_LIST_HEAD(&config->message_history);
    INIT_LIST_HEAD(&config->tools);

    /* Set defaults */
    config->max_tokens = 2048;
    config->temperature_X100 = 70;
    config->top_p_X100 = 100;
    config->n_choices = 1;
    config->stream = false;
    config->use_ssl = true;
    config->timeout_ms = 30000;
    config->max_requests_per_min = 60;

    /* Set endpoint */
    strncpy(config->endpoint, "api.openai.com", MAX_ENDPOINT_LENGTH - 1);

    /* Initialize rate limiting */
    atomic_set(&config->rate_limit.requests_remaining,
               config->max_requests_per_min);
    atomic64_set(&config->rate_limit_reset, jiffies + HZ * 60);

    /* Validate and set API key if provided */
    if (api_key) {
        ret = llm_set_api_key(config, api_key);
        if (ret < 0)
            goto cleanup_config;
    }

    /* Register character device */
    major_number = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_number < 0) {
        ret = major_number;
        goto cleanup_config;
    }

    /* Create device class */
    llm_class = class_create(THIS_MODULE, DEVICE_NAME);
    if (IS_ERR(llm_class)) {
        ret = PTR_ERR(llm_class);
        goto cleanup_chrdev;
    }

    /* Create device */
    llm_device = device_create(llm_class, NULL,
                             MKDEV(major_number, 0),
                             NULL, DEVICE_NAME);
    if (IS_ERR(llm_device)) {
        ret = PTR_ERR(llm_device);
        goto cleanup_class;
    }

    mutex_lock(&llm_mutex);
    global_config = config;
    mutex_unlock(&llm_mutex);

    pr_info("LLM: Provider module loaded successfully\n");
    return 0;

cleanup_class:
    class_destroy(llm_class);
cleanup_chrdev:
    unregister_chrdev(major_number, DEVICE_NAME);
cleanup_config:
    llm_config_cleanup(config);
    kfree(config);
    return ret;
}

static void __exit llm_provider_exit(void)
{
    llm_cleanup();
    pr_info("LLM: Provider module unloaded\n");
}

static int llm_open(struct inode *inode, struct file *file) {
    return 0;
}

static int llm_release(struct inode *inode, struct file *file) {
    return 0;
}

static ssize_t llm_read(struct file *file, char __user *buf,
                       size_t count, loff_t *offset) {
    struct llm_response resp;
    int ret;

    /* Get response from OpenAI */
    ret = llm_receive_response(global_config, &resp);
    if (ret < 0)
        return ret;

    /* Copy to user space */
    if (copy_to_user(buf, resp.message->content,
                    resp.message->content_length))
        return -EFAULT;

    return resp.message->content_length;
}

static ssize_t llm_write(struct file *file, const char __user *buf,
                        size_t count, loff_t *offset) {
    struct llm_request req;
    struct llm_message *msg;
    int ret;

    if (count != sizeof(struct llm_request))
        return -EINVAL;

    /* Copy from user space */
    if (copy_from_user(&req, buf, sizeof(req)))
        return -EFAULT;

    /* Create and send message */
    msg = llm_message_alloc(req.role, req.prompt);
    if (!msg)
        return -ENOMEM;

    ret = llm_send_message(global_config, msg);
    llm_message_free(msg);

    return ret ? ret : count;
}
static void llm_message_cleanup(struct llm_message *msg)
{
    if (!msg)
        return;

    if (msg->content) {
        memzero_explicit(msg->content, msg->content_length);
        kfree(msg->content);
    }
    memzero_explicit(msg, sizeof(*msg));
    kfree(msg);
}

static void cleanup_message_history(struct llm_config *config)
{
    struct llm_message *msg, *tmp;

    list_for_each_entry_safe(msg, tmp, &config->message_history, list) {
        list_del(&msg->list);
        llm_message_free(msg);
    }
}



/* Message Management */
struct llm_message *llm_message_alloc(const char *role, const char *content)
{
    struct llm_message *msg;
    size_t content_len;

    if (!role || !content)
        return NULL;

    msg = kmalloc(sizeof(*msg), GFP_KERNEL);
    if (!msg)
        return NULL;

    /* Initialize role */
    strncpy(msg->role, role, MAX_ROLE_LENGTH - 1);
    msg->role[MAX_ROLE_LENGTH - 1] = '\0';

    /* Allocate and copy content */
    content_len = strlen(content) + 1;
    msg->content = kmalloc(content_len, GFP_KERNEL);
    if (!msg->content) {
        kfree(msg);
        return NULL;
    }

    strncpy(msg->content, content, content_len);
    msg->content_length = content_len - 1;
    INIT_LIST_HEAD(&msg->list);

    return msg;
}



void llm_message_free(struct llm_message *msg)
{
    if (!msg)
        return;

    if (msg->content)
        kfree(msg->content);
    kfree(msg);
}

int llm_add_message(struct llm_config *config, struct llm_message *msg) {
    if (!config || !msg)
        return -EINVAL;

    mutex_lock(&config->message_lock);
    list_add_tail(&msg->list, &config->message_history);
    mutex_unlock(&config->message_lock);
    return 0;
}

/* Tool Management */
struct llm_tool *llm_tool_alloc(const char *name, const char *description)
{
    struct llm_tool *tool;

    if (!name || !description)
        return NULL;

    tool = kmalloc(sizeof(*tool), GFP_KERNEL);
    if (!tool)
        return NULL;

    strncpy(tool->name, name, MAX_TOOL_NAME - 1);
    tool->name[MAX_TOOL_NAME - 1] = '\0';

    strncpy(tool->description, description, MAX_TOOL_DESC - 1);
    tool->description[MAX_TOOL_DESC - 1] = '\0';

    INIT_LIST_HEAD(&tool->parameters);
    INIT_LIST_HEAD(&tool->list);

    return tool;
}
static void cleanup_tools(struct llm_config *config)
{
    struct llm_tool *tool, *tmp_tool;
    struct llm_tool_param *param, *tmp_param;

    list_for_each_entry_safe(tool, tmp_tool, &config->tools, list) {
        list_for_each_entry_safe(param, tmp_param, &tool->parameters, list) {
            list_del(&param->list);
            kfree(param);
        }
        list_del(&tool->list);
        llm_tool_free(tool);
    }
}
void llm_tool_free(struct llm_tool *tool)
{
    struct llm_tool_param *param, *tmp;

    if (!tool)
        return;

    list_for_each_entry_safe(param, tmp, &tool->parameters, list) {
        list_del(&param->list);
        kfree(param);
    }

    kfree(tool);
}

int llm_add_tool_param(struct llm_tool *tool, const char *name,
                      const char *description, bool required)
{
    struct llm_tool_param *param;

    if (!tool || !name || !description)
        return -EINVAL;

    param = kmalloc(sizeof(*param), GFP_KERNEL);
    if (!param)
        return -ENOMEM;

    strncpy(param->name, name, MAX_TOOL_NAME - 1);
    param->name[MAX_TOOL_NAME - 1] = '\0';

    strncpy(param->description, description, MAX_TOOL_DESC - 1);
    param->description[MAX_TOOL_DESC - 1] = '\0';

    param->required = required;
    INIT_LIST_HEAD(&param->list);

    list_add_tail(&param->list, &tool->parameters);
    return 0;
}

/* Replace global mutex with config-specific locks */
int llm_add_tool(struct llm_config *config, struct llm_tool *tool) {
    if (!config || !tool)
        return -EINVAL;

    mutex_lock(&config->tool_lock);
    list_add_tail(&tool->list, &config->tools);
    mutex_unlock(&config->tool_lock);
    return 0;
}

/* Main Interface Implementation */





module_init(llm_provider_init);
module_exit(llm_provider_exit);


MODULE_LICENSE("GPL");
MODULE_AUTHOR("Sina Mazaheri");
MODULE_DESCRIPTION("OpenAI LLM Provider Implementation");
MODULE_VERSION("1.0");
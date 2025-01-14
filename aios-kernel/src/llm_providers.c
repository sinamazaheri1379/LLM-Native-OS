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

/* JSON Parser */
static void init_json_parser(struct json_parser *parser,
                           const char *input, size_t len) {
    parser->input = input;
    parser->pos = 0;
    parser->len = len;
}

static void skip_whitespace(struct json_parser *parser) {
    while (parser->pos < parser->len &&
           isspace(parser->input[parser->pos]))
        parser->pos++;
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
static inline struct llm_config *get_global_config(void) {
    struct llm_config *config;
    mutex_lock(&llm_mutex);
    config = global_config;
    mutex_unlock(&llm_mutex);
    return config;
}
/* Safe string handling */
static int json_unescape_string(const char *input, char *output, size_t outlen) {
    size_t i, j = 0;

    if (!input || !output || outlen < 2)
        return -EINVAL;

    for (i = 0; input[i] && j + 1 < outlen; i++) {
        if (i >= strlen(input))  // Bounds check
            return -EINVAL;

        if (input[i] == '\\') {
            i++;
            if (i >= strlen(input))  // Check escape sequence
                return -EINVAL;

            switch(input[i]) {
                case '"':
                case '\\':
                case '/':
                    if (j + 1 >= outlen) return -E2BIG;
                    output[j++] = input[i];
                    break;
                case 'u': {
                    unsigned int codepoint;
                    if (i + 4 >= strlen(input)) return -EINVAL;

                    /* Safe hex conversion */
                    if (hex2bin((u8 *)&codepoint, input + i + 1, 2) < 0)
                        return -EINVAL;

                    if (codepoint > 127)  // ASCII only
                        return -EINVAL;

                    if (j + 1 >= outlen) return -E2BIG;
                    output[j++] = (char)codepoint;
                    i += 4;
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
    bool locked = false;
    struct llm_message *history_msg;

    if (!buf || !config || !msg)
        return -EINVAL;

    if (!LLM_VALID_MAX_TOKENS(config->max_tokens) ||
        !LLM_VALID_TEMP_RANGE(config->temperature_X100))
        return -EINVAL;

    ret = llm_json_buffer_init(buf, MAX_PROMPT_LENGTH);
    if (ret)
        return ret;

    mutex_lock(&config->message_lock);
    locked = true;

    ret = append_to_json(buf, "{\"model\":\"");
    if (ret)
        goto cleanup;

    ret = append_to_json(buf, config->model);
    if (ret)
        goto cleanup;

    ret = append_to_json(buf, "\",\"messages\":[");
    if (ret)
        goto cleanup;

    list_for_each_entry(history_msg, &config->message_history, list) {
        ret = append_json_string(buf, history_msg->role);
        if (ret)
            goto cleanup;

        ret = append_json_string(buf, history_msg->content);
        if (ret)
            goto cleanup;
    }

    /* Add current message */
    ret = append_json_string(buf, msg->role);
    if (ret)
        goto cleanup;

    ret = append_json_string(buf, msg->content);
    if (ret)
        goto cleanup;

    ret = append_to_json(buf, "]}");
    if (ret)
        goto cleanup;

    mutex_unlock(&config->message_lock);
    return 0;

cleanup:
    if (locked)
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
/* SSL Session Implementation */


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
    struct ssl_context *ssl;

    if (!conn)
        return -EINVAL;

    ssl = kzalloc(sizeof(*ssl), GFP_KERNEL);  // Use kzalloc for zero initialization
    if (!ssl)
        return -ENOMEM;

    /* Initialize crypto with proper error handling */
    ssl->tfm = crypto_alloc_aead("gcm(aes)", 0, CRYPTO_ALG_ASYNC);
    if (IS_ERR(ssl->tfm)) {
        ret = PTR_ERR(ssl->tfm);
        goto cleanup_ssl;
    }

    /* Allocate scatterlists with boundary checks */
    ssl->sg_tx = kcalloc(2, sizeof(struct scatterlist), GFP_KERNEL);
    ssl->sg_rx = kcalloc(2, sizeof(struct scatterlist), GFP_KERNEL);
    if (!ssl->sg_tx || !ssl->sg_rx) {
        ret = -ENOMEM;
        goto cleanup_crypto;
    }

    /* Initialize SSL session */
    ret = init_ssl_session(ssl);
    if (ret)
        goto cleanup_all;

    conn->ssl_context = ssl;
    return 0;

cleanup_all:
    kfree(ssl->sg_tx);
    kfree(ssl->sg_rx);
cleanup_crypto:
    crypto_free_aead(ssl->tfm);
cleanup_ssl:
    kfree(ssl);
    return ret;
}
static int establish_connection(struct llm_config *config, struct llm_connection *conn) {
    int ret;
    struct sockaddr_in *addr;
    unsigned long timeout;

    if (!config || !conn)
        return -EINVAL;

    memset(conn, 0, sizeof(*conn));
    addr = &conn->addr;
    timeout = msecs_to_jiffies(config->timeout_ms);

    /* Create socket */
    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM, IPPROTO_TCP, &conn->sock);
    if (ret < 0)
        return -LLM_ERR_NETWORK_INIT;

    /* Set timeouts */
    ret = sock_setsockopt(conn->sock, SOL_SOCKET, SO_SNDTIMEO,
                         (char *)&timeout, sizeof(timeout));
    if (ret < 0)
        goto cleanup_socket;

    ret = sock_setsockopt(conn->sock, SOL_SOCKET, SO_RCVTIMEO,
                         (char *)&timeout, sizeof(timeout));
    if (ret < 0)
        goto cleanup_socket;

    /* Setup address */
    addr->sin_family = AF_INET;
    addr->sin_port = htons(443);
    ret = in4_pton(config->endpoint, strlen(config->endpoint),
                   (u8 *)&addr->sin_addr.s_addr, -1, NULL);
    if (ret != 1) {
        ret = -LLM_ERR_NETWORK_INIT;
        goto cleanup_socket;
    }

    /* Connect with SSL if enabled */
    ret = kernel_connect(conn->sock, (struct sockaddr *)addr,
                        sizeof(*addr), O_NONBLOCK);
    if (ret < 0 && ret != -EINPROGRESS) {
        ret = -LLM_ERR_NETWORK_CONN;
        goto cleanup_socket;
    }

    if (config->use_ssl) {
        ret = establish_ssl_connection(conn);
        if (ret) {
            ret = -LLM_ERR_SSL;
            goto cleanup_socket;
        }
    }

    return 0;

cleanup_socket:
    cleanup_connection(conn);
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

    if (strlen(key) >= API_KEY_MAX_LEN)
        return -E2BIG;

    mutex_lock(&config->config_lock);
    strncpy(config->api_key, key, API_KEY_MAX_LEN - 1);
    config->api_key[API_KEY_MAX_LEN - 1] = '\0';
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
static void cleanup_message_history(struct llm_config *config)
{
    struct llm_message *msg, *tmp;

    list_for_each_entry_safe(msg, tmp, &config->message_history, list) {
        list_del(&msg->list);
        llm_message_free(msg);
    }
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
int llm_init(struct llm_config *config)
{
    if (!config)
        return -EINVAL;

    mutex_lock(&llm_mutex);

    if (global_config) {
        mutex_unlock(&llm_mutex);
        return -EBUSY;
    }

    /* Initialize lists */
    INIT_LIST_HEAD(&config->message_history);
    INIT_LIST_HEAD(&config->tools);

    /* Set defaults */
    config->max_tokens = 2048;
    config->temperature_X100 = 70;  /* 0.7 */
    config->top_p_X100 = 100;      /* 1.0 */
    config->n_choices = 1;
    config->stream = false;
    config->presence_penalty_X100 = 0;
    config->frequency_penalty_X100 = 0;
    config->use_ssl = true;
    config->timeout_ms = 30000;
    config->max_requests_per_min = 60;
    config->remaining_requests = 60;
    config->rate_limit_reset = jiffies + HZ * 60;

    global_config = config;
    mutex_unlock(&llm_mutex);

    pr_info("LLM: Initialized with model %s\n", config->model);
    return 0;
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
    /* Register character device */
    major_number = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_number < 0) {
        pr_err("Failed to register character device\n");
        return major_number;
    }

    /* Create device class */
    llm_class = class_create(THIS_MODULE, DEVICE_NAME);
    if (IS_ERR(llm_class)) {
        unregister_chrdev(major_number, DEVICE_NAME);
        return PTR_ERR(llm_class);
    }

    /* Create device */
    llm_device = device_create(llm_class, NULL,
                             MKDEV(major_number, 0),
                             NULL, DEVICE_NAME);
    if (IS_ERR(llm_device)) {
        class_destroy(llm_class);
        unregister_chrdev(major_number, DEVICE_NAME);
        return PTR_ERR(llm_device);
    }

    pr_info("LLM: Provider module loaded\n");
    return 0;
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

/* Module initialization and cleanup */
static int __init llm_provider_init(void)
{
    pr_info("LLM: Provider module loaded\n");
    return 0;
}

static void __exit llm_provider_exit(void)
{
    llm_cleanup();
    pr_info("LLM: Provider module unloaded\n");
}

module_init(llm_provider_init);
module_exit(llm_provider_exit);




MODULE_LICENSE("GPL");
MODULE_AUTHOR("Sina Mazaheri");
MODULE_DESCRIPTION("OpenAI LLM Provider Implementation");
MODULE_VERSION("1.0");
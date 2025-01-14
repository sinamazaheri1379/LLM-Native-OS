//
// Created by sina-mazaheri on 12/17/24.
//
//
// Created by sina-mazaheri on 12/17/24.
//
#ifndef LLM_PROVIDERS_H
#define LLM_PROVIDERS_H

#ifndef _LINUX_TYPES_H
#include <linux/types.h>
#endif
#ifndef _LINUX_ATOMIC_H
#include <linux/atomic.h>
#endif
#ifndef _LINUX_MUTEX_H
#include <linux/mutex.h>
#endif
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/net.h>
#include <linux/in.h>
#include <net/sock.h>
#include <linux/list.h>
#include <linux/spinlock.h>

/* Maximum buffer sizes */
#define MAX_API_KEY_LENGTH     256
#define MAX_PROMPT_LENGTH      4096
#define MAX_RESPONSE_LENGTH    8192
#define MAX_MODEL_NAME         64
#define MAX_ENDPOINT_LENGTH    256
#define MAX_ERROR_LENGTH       256
#define MAX_ROLE_LENGTH        32
#define MAX_MESSAGES          100
#define MAX_TOOL_NAME         64
#define MAX_TOOL_DESC         256
#define MAX_FUNCTION_ARGS     1024
#define MAX_STOP_SEQ_LEN      64
#define MAX_INT32         0x7FFFFFFF

/* Message roles */
#define ROLE_SYSTEM     "system"
#define ROLE_USER       "user"
#define ROLE_ASSISTANT  "assistant"
#define ROLE_TOOL       "tool"

/* Error Codes */
#define LLM_ERR_SUCCESS         0
#define LLM_ERR_INVALID_PARAM  -1
#define LLM_ERR_OVERFLOW      -2
#define LLM_ERR_NOMEM         -3
#define LLM_ERR_BUSY          -4
#define LLM_ERR_IO            -5
#define LLM_ERR_NETWORK_INIT   -100
#define LLM_ERR_NETWORK_CONN   -101
#define LLM_ERR_SSL            -102
#define LLM_ERR_JSON_FORMAT    -103
#define LLM_ERR_JSON_PARSE     -104
#define LLM_ERR_API_RESPONSE   -105
#define LLM_ERR_RATE_LIMIT     -106
#define LLM_ERR_MUTEX_INIT     -107
#define LLM_ERR_MUTEX_LOCK     -108
#define LLM_ERR_MUTEX_UNLOCK   -109
#define LLM_ERR_REF_COUNT      -110

/* Validation macros */
#define LLM_VALID_TEMP_RANGE(x)    ((x) >= 0 && (x) <= 100)
#define LLM_VALID_PENALTY_RANGE(x) ((x) >= -200 && (x) <= 200)
#define LLM_VALID_MAX_TOKENS(x)    ((x) > 0 && (x) <= 32768)
#define LLM_VALID_N_CHOICES(x)     ((x) > 0 && (x) <= 10)
#define LLM_VALID_TIMEOUT(x)       ((x) > 0 && (x) <= 300000)
#define LLM_VALID_REQ_LIMIT(x)     ((x) > 0 && (x) <= 1000)
#define LLM_VALID_ENDPOINT(x)      ((x) != NULL && strlen(x) < MAX_ENDPOINT_LENGTH)
#define LLM_VALID_USER_ID(x)       ((x) == NULL || strlen(x) < 128)
#define LLM_VALID_API_KEY(x)       ((x) != NULL && strnlen((x), MAX_API_KEY_LENGTH) > 32)
#define LLM_VALID_MESSAGE(msg)     ((msg) != NULL && \
                                   (msg)->content != NULL && \
                                   (msg)->content_length > 0 && \
                                   (msg)->content_length <= MAX_PROMPT_LENGTH)
#define LLM_VALID_JSON_BUFFER(x)   ((x) != NULL && (x)->data != NULL && (x)->size > 0)
#define LLM_VALID_SSL_CTX(x)       ((x) != NULL && (x)->ssl_context != NULL)
#define LLM_VALID_RATE_LIMIT(x)    ((x) != NULL && atomic_read(&(x)->requests_remaining) >= 0)
#define LLM_VALID_RESPONSE(x)      ((x) != NULL && (x)->id[0] != '\0' && (x)->status_code >= 0)
#define LLM_VALID_TOOL(x)         ((x) != NULL && (x)->name[0] != '\0')
#define LLM_VALID_TOOL_PARAM(x)   ((x) != NULL && (x)->name[0] != '\0' && \
(x)->description[0] != '\0')
#define LLM_VALID_INIT(x)         ((x) != NULL && (x)->model[0] != '\0' && \
LLM_VALID_API_KEY((x)->api_key))
#define LLM_VALID_JSON_APPEND(x,s) ((x) != NULL && (s) != NULL && \
((x)->used + strlen(s) + 1 <= (x)->size))
/* Version information */
#define LLM_API_VERSION_MAJOR  1
#define LLM_API_VERSION_MINOR  0
#define OPENAI_API_KEY_ENV "OPENAI_API_KEY"
/**
 * Locking rules:
 * - config_lock protects configuration parameters
 * - message_lock protects message_history
 * - tool_lock protects tools list
 * - Lock order: config_lock -> message_lock -> tool_lock
 */

struct llm_ssl_context;
struct llm_message;
struct llm_tool;
struct llm_tool_call;
struct llm_connection;
struct llm_json_buffer;
typedef struct llm_ssl_context llm_ssl_context_t;

/**
 * struct llm_version - Version information
 * @major: Major version number
 * @minor: Minor version number
 */




struct llm_version {
    uint16_t major;
    uint16_t minor;
};

/**
 * struct llm_rate_limit - Rate limiting information
 */
struct llm_rate_limit {
    atomic_t requests_remaining;
    atomic_t tokens_remaining;
    atomic64_t reset_time;
    spinlock_t lock;
};

/**
 * struct llm_message - Single chat message
 */
struct llm_message {
    char role[MAX_ROLE_LENGTH];
    char *content;
    size_t content_length;
    struct list_head list;
};

/**
 * struct llm_connection - Network connection state
 */
struct llm_connection {
    struct socket *sock;
    struct sockaddr_in addr;
    bool ssl_enabled;
    llm_ssl_context_t *ssl_context;
    atomic_t in_use;
    struct mutex lock;
};

/**
 * struct llm_json_buffer - JSON data buffer
 */
struct llm_json_buffer {
    char *data;
    size_t size;
    size_t used;
};

/**
 * struct llm_tool_param - Function parameter definition
 */
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

/**
 * struct llm_response - API response container
 */
struct llm_response {
    char id[64];
    char model[MAX_MODEL_NAME];
    unsigned long created;
    struct {
        int prompt_tokens;
        int completion_tokens;
        int total_tokens;
    } usage;
    struct llm_message *message;
    struct list_head tool_calls;
    char finish_reason[32];
    int status_code;
    char error[MAX_ERROR_LENGTH];
};

/**
 * struct llm_config - Enhanced configuration for OpenAI
 */
struct ssl_context {
    struct crypto_aead *tfm;
    u8 *key;
    u8 *iv;
    size_t key_size;
    size_t iv_size;
    struct scatterlist *sg_tx;
    struct scatterlist *sg_rx;
    struct aead_request *req;
    bool handshake_complete;
    struct {
        u8 *data;
        size_t size;
    } session;
};


struct llm_config {
    /* Authentication */
    struct llm_version version;
    char api_key[MAX_API_KEY_LENGTH];
    char org_id[64];

    /* Model configuration */
    char model[MAX_MODEL_NAME];
    uint32_t max_tokens;
    uint16_t temperature_X100;
    uint16_t top_p_X100;
    uint32_t n_choices;
    bool stream;

    /* Message control */
    int16_t presence_penalty_X100;
    int16_t frequency_penalty_X100;
    char stop_sequences[4][MAX_STOP_SEQ_LEN];
    uint8_t num_stop_sequences;

    /* Network configuration */
    char endpoint[MAX_ENDPOINT_LENGTH];
    bool use_ssl;
    uint32_t timeout_ms;

    /* Rate limiting */
    uint32_t max_requests_per_min;
    uint32_t remaining_requests;
    atomic64_t rate_limit_reset;

    /* Runtime state */
    struct list_head message_history;
    struct list_head tools;
    char user_id[128];

    /* Thread safety */
    struct mutex config_lock;
    struct mutex message_lock;
    struct mutex tool_lock;
    atomic_t ref_count;
};

/**
 * struct llm_provider_ops - Provider operations
 */
struct llm_provider_ops {
    int (*init)(struct llm_config *config);
    void (*cleanup)(void);
    int (*send_message)(struct llm_message *msg);
    int (*receive_response)(struct llm_response *resp);
    int (*add_tool)(struct llm_tool *tool);
    int (*process_tool_call)(struct llm_tool_call *call);
};

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

static inline int llm_check_version(struct llm_version *version) {
    if (!version) return -LLM_ERR_INVALID_PARAM;
    if (version->major != LLM_API_VERSION_MAJOR) return -LLM_ERR_INVALID_PARAM;
    if (version->minor > LLM_API_VERSION_MINOR) return -LLM_ERR_INVALID_PARAM;
    return LLM_ERR_SUCCESS;
}

/* Reference counting */
static inline void llm_config_get(struct llm_config *config) {
    smp_mb__before_atomic();
    atomic_inc(&config->ref_count);
    smp_mb__after_atomic();
}

/* Corrected cleanup function */
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

int llm_set_api_key(struct llm_config *config, const char *api_key);
int llm_load_api_key_from_env(struct llm_config *config);
/* Complete the config_put implementation */
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

/* Add JSON buffer management declarations */
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
void llm_json_buffer_free(struct llm_json_buffer *buf);
int llm_json_buffer_append(struct llm_json_buffer *buf, const char *str);

/* Function declarations */
/**
 * llm_init - Initialize LLM configuration
 * @config: Configuration to initialize
 *
 * Must be called with config_lock held
 * Return: 0 on success, negative on error
 */
int llm_init(struct llm_config *config);

/**
 * llm_cleanup - Clean up LLM resources
 */
void llm_cleanup(void);

/* Message management */
/**
 * llm_message_alloc - Allocate new message
 * @role: Message role (system/user/assistant/tool)
 * @content: Message content
 *
 * Return: New message or NULL on error
 */
struct llm_message *llm_message_alloc(const char *role, const char *content);
void llm_message_free(struct llm_message *msg);

/**
 * llm_add_message - Add message to history
 * @config: Config to modify
 * @msg: Message to add
 *
 * Requires message_lock held
 * Return: 0 on success, negative on error
 */
int llm_add_message(struct llm_config *config, struct llm_message *msg);

/* Tool management */
struct llm_tool *llm_tool_alloc(const char *name, const char *description);
void llm_tool_free(struct llm_tool *tool);

/**
 * llm_add_tool_param - Add parameter to tool
 * @tool: Tool to modify
 * @name: Parameter name
 * @description: Parameter description
 * @required: Whether parameter is required
 *
 * Return: 0 on success, negative on error
 */
int llm_add_tool_param(struct llm_tool *tool, const char *name,
                      const char *description, bool required);

/**
 * llm_add_tool - Add tool to config
 * @config: Config to modify
 * @tool: Tool to add
 *
 * Requires tool_lock held
 * Return: 0 on success, negative on error
 */
int llm_add_tool(struct llm_config *config, struct llm_tool *tool);

/* Message handling */
/**
 * llm_send_message - Send message to LLM provider
 * @config: Config to use
 * @msg: Message to send
 *
 * Return: 0 on success, negative on error
 */
int llm_send_message(struct llm_config *config, struct llm_message *msg);

/**
 * llm_receive_response - Receive response from LLM provider
 * @config: Config to use
 * @resp: Response buffer
 *
 * Return: 0 on success, negative on error
 */
int llm_receive_response(struct llm_config *config, struct llm_response *resp);

#endif /* LLM_PROVIDERS_H */
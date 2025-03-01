//
// Created by sina-mazaheri on 12/17/24.
//
//
// Created by sina-mazaheri on 12/17/24.
//
#ifndef LLM_PROVIDERS_H
#define LLM_PROVIDERS_H


#include <linux/types.h>
#include <linux/atomic.h>
#include <linux/mutex.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/net.h>
#include <linux/in.h>
#include <net/sock.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/slab.h>
#include <linux/tls.h>
#include <net/tls.h>
#include <linux/wait.h>
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
#define MAX_CONN_POOL_SIZE 8
#define CONN_TIMEOUT_MS 30000
#define CONN_KEEPALIVE_MS 5000
#define LLM_MAX_ERROR_RETRIES 3
#define LLM_ERROR_WINDOW_MS 300000  // 5 minutes

/* Message roles */
#define ROLE_SYSTEM     "system"
#define ROLE_USER       "user"
#define ROLE_ASSISTANT  "assistant"
#define ROLE_TOOL       "tool"
/* Environment Setup */
#define OPENAI_API_KEY_ENV "OPENAI_API_KEY"
#define DEFINE_VALIDATOR(name, func, desc) \
    { name, func, desc }
#define LLM_API_VERSION_MAJOR  1
#define LLM_API_VERSION_MINOR  0
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
#define LLM_VALID_SSL_CTX(x) ((x) != NULL && (x)->tls != NULL)
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

/**
 * Locking rules:
 * - config_lock protects configuration parameters
 * - message_lock protects message_history
 * - tool_lock protects tools list
 * - Lock order: config_lock -> message_lock -> tool_lock
 */


/////////*  Enum Definitions  */////////

/* Rate limit states */
enum llm_rate_state {
    RATE_STATE_OK,
    RATE_STATE_LIMITED,
    RATE_STATE_RECOVERING
};

/* Response states */
enum response_state {
    RESP_STATE_INIT,
    RESP_STATE_HEADERS,
    RESP_STATE_BODY,
    RESP_STATE_COMPLETE,
    RESP_STATE_ERROR
};

/* Log Events */
enum llm_log_level {
    LLM_LOG_DEBUG,
    LLM_LOG_INFO,
    LLM_LOG_WARN,
    LLM_LOG_ERROR
};

/* Error State Recovery */
enum llm_error_category {
    LLM_ERR_CAT_NETWORK,
    LLM_ERR_CAT_API,
    LLM_ERR_CAT_MEMORY,
    LLM_ERR_CAT_INTERNAL,
    LLM_ERR_CAT_SECURITY
};

/* Connection State */
enum conn_state {
    CONN_STATE_INIT,
    CONN_STATE_CONNECTING,
    CONN_STATE_CONNECTED,
    CONN_STATE_ERROR,
    CONN_STATE_CLOSED
};


/////////////////// End //////////////////////

/////// Tool Related Structures //////
struct llm_tool_param {
    char name[MAX_TOOL_NAME];
    char description[MAX_TOOL_DESC];
    bool required;
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
 * struct llm_tool - Function calling definition
 */
struct llm_tool {
    char name[MAX_TOOL_NAME];
    char description[MAX_TOOL_DESC];
    struct list_head parameters;
    struct list_head list;
};
/////////////////// End //////////////////////


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
/* Memory tracking structure */
/* Error logging levels */
struct llm_rate_limiter {
    atomic_t tokens;
    atomic_t max_tokens;
    atomic64_t last_refill;
    spinlock_t limiter_lock;
    unsigned long refill_interval_ms;
    unsigned long tokens_per_interval;
    atomic_t is_limited;
};





struct llm_response_handler {
    struct llm_response *resp;
    struct llm_json_buffer *buffer;
    size_t max_size;
    bool streaming;
    void (*callback)(struct llm_response *resp, void *data);
    void *callback_data;
    atomic_t is_complete;
    spinlock_t handler_lock;
};




struct llm_message_queue {
    struct list_head messages;
    spinlock_t queue_lock;
    wait_queue_head_t wait_queue;
    atomic_t count;
    size_t max_size;
    atomic_t is_active;
};

/* Queue item structure */
struct queue_item {
    struct llm_message *msg;
    struct list_head list;
    unsigned long timestamp;
    int priority;
};

struct llm_config_state {
    atomic_t is_initialized;
    atomic_t is_modified;
    atomic_t ref_count;
    struct mutex state_lock;
    unsigned long last_modified;
    char last_modifier[64];
};


struct llm_config_validator {
    const char *name;
    int (*validate)(const void *value, size_t size);
    const char *description;
};





struct retry_context {
    int max_retries;
    int current_retry;
    unsigned long delay_ms;
    unsigned long max_delay_ms;
    const char *operation;
};
/* Error tracking structure */
struct llm_error_state {
    atomic_t error_count;
    atomic64_t first_error_time;
    atomic64_t last_error_time;
    spinlock_t error_lock;
    int last_error;
    char last_error_func[64];
    int last_error_line;
};
struct llm_mem_tracker {
    atomic_t alloc_count;
    atomic_t free_count;
    atomic64_t total_bytes;
    spinlock_t track_lock;
    struct list_head alloc_list;  // List of active allocations
};

/* Track individual allocations */
struct llm_mem_block {
    void *ptr;
    size_t size;
    const char *func;
    int line;
    struct list_head list;
};




struct llm_version {
    uint16_t major;
    uint16_t minor;
};

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
    struct tls_context *tls;  // Use kernel TLS instead of custom SSL
    enum conn_state state;
    atomic_t ref_count;
    unsigned long last_used;
    unsigned long timeout_ms;
    spinlock_t lock;
    struct llm_config *config;
    bool keepalive;
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


/* Connection pool */
struct llm_conn_pool {
    struct {
        struct llm_connection *conn;
        atomic_t ref_count;
        unsigned long last_used;
        spinlock_t lock;
    } slots[MAX_CONN_POOL_SIZE];
    atomic_t total_conns;
    struct work_struct cleanup_work;
    struct timer_list cleanup_timer;
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
struct llm_ssl_context {
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







/* Corrected cleanup function */


int llm_set_api_key(struct llm_config *config, const char *api_key);
int llm_load_api_key_from_env(struct llm_config *config);
/* Complete the config_put implementation */


/* Add JSON buffer management declarations */

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
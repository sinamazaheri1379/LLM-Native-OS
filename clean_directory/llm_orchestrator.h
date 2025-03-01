#ifndef LLM_ORCHESTRATOR_H
#define LLM_ORCHESTRATOR_H

#include <linux/types.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include <linux/atomic.h>

/* Maximum lengths */
#define MAX_API_KEY_LENGTH     256
#define MAX_PROMPT_LENGTH      4096
#define MAX_RESPONSE_LENGTH    8192
#define MAX_ROLE_LENGTH        32
#define MAX_PAYLOAD_SIZE       8192  /* Increased for larger context */
#define MAX_CONTEXT_ENTRIES    20    /* Increased max context entries */
#define MAX_CONVERSATION_ID    128   /* Increased max conversation ID length */
#define MAX_FIFO_QUEUE_SIZE    32    /* Maximum size for FIFO queue */
#define MAX_MODEL_NAME         128   /* Increased from 64 to 128 */

/* Provider identifiers */
#define PROVIDER_OPENAI         0
#define PROVIDER_ANTHROPIC      1
#define PROVIDER_GOOGLE_GEMINI  2
#define PROVIDER_COUNT          3

/* Scheduler algorithms */
#define SCHEDULER_ROUND_ROBIN   0
#define SCHEDULER_WEIGHTED      1
#define SCHEDULER_PRIORITY      2
#define SCHEDULER_PERFORMANCE   3
#define SCHEDULER_COST_AWARE    4
#define SCHEDULER_FALLBACK      5
#define SCHEDULER_FIFO          6    /* Added FIFO scheduler */

/* Provider status */
#define PROVIDER_STATUS_OK      0
#define PROVIDER_STATUS_ERROR   1
#define PROVIDER_STATUS_TIMEOUT 2
#define PROVIDER_STATUS_RATE_LIMITED 3  /* Added rate limiting status */

/* Error codes */
#define LLM_ERR_SUCCESS        0
#define LLM_ERR_INVALID_PARAM  -1
#define LLM_ERR_NOMEM          -2
#define LLM_ERR_BUSY           -3
#define LLM_ERR_NETWORK        -4
#define LLM_ERR_API_RESPONSE   -5
#define LLM_ERR_RATE_LIMIT     -6
#define LLM_ERR_TIMEOUT        -7
#define LLM_ERR_AUTH           -8    /* Authentication error */

/* Request structure sent from user space */
struct llm_request {
    char role[MAX_ROLE_LENGTH];       /* e.g., "user" */
    char prompt[MAX_PROMPT_LENGTH];   /* Prompt or query */
    int conversation_id;              /* ID to track context for this conversation */
    int scheduler_algorithm;          /* Which scheduling algorithm to use */
    int priority;                     /* Priority level (if using priority scheduler) */
    unsigned long timeout_ms;         /* Timeout in milliseconds */
    int provider_preference;          /* Preferred provider, -1 for no preference */
    char model_name[MAX_MODEL_NAME];  /* Specific model to use (if supported by provider) */
    int max_tokens;                   /* Maximum tokens for response */
    int temperature_x100;             /* Temperature x 100 (e.g., 70 = 0.7) */
};

/* Response structure returned to user space */
struct llm_response {
    char content[MAX_RESPONSE_LENGTH];
    size_t content_length;
    int provider_used;                /* Which provider generated this response */
    unsigned long latency_ms;         /* How long the request took */
    int status;                       /* Status code of the request */
    int tokens_used;                  /* Total tokens used (if available) */
    char model_used[MAX_MODEL_NAME];  /* Model that generated the response */
    ktime_t timestamp;                /* When the response was generated */
};

/* Context entry for a single exchange */
struct context_entry {
    char role[MAX_ROLE_LENGTH];
    char content[MAX_PROMPT_LENGTH];
    ktime_t timestamp;               /* When this entry was added */
    struct list_head list;
};

/* Context manager for a conversation */
struct conversation_context {
    int conversation_id;
    int entry_count;
    ktime_t last_updated;            /* When this conversation was last updated */
    struct list_head entries;
    struct list_head list;           /* For linking in global list */
    spinlock_t lock;
};

/* JSON buffer used for building requests */
struct llm_json_buffer {
    char *data;
    size_t size;
    size_t used;
};

/* Provider performance metrics */
struct provider_metrics {
    atomic_t total_requests;
    atomic_t successful_requests;
    atomic_t failed_requests;
    atomic_t timeouts;
    atomic_t rate_limited;            /* Count of rate limit errors */
    atomic64_t total_latency_ms;      /* Sum of all latencies */
    unsigned long min_latency_ms;
    unsigned long max_latency_ms;
    atomic64_t last_success_jiffies;
    atomic_t current_status;          /* Current provider status */
    spinlock_t lock;
    
    /* Token usage tracking */
    atomic_t total_tokens;
    atomic_t prompt_tokens;
    atomic_t completion_tokens;
    
    /* Quota tracking */
    atomic_t remaining_quota;         /* For quota-based systems */
    ktime_t quota_reset_time;         /* When quota resets */
};

/* FIFO queue entry for the FIFO scheduler */
struct fifo_entry {
    int provider;
    ktime_t enqueue_time;
    struct list_head list;
};

/* FIFO queue for scheduler */
struct fifo_queue {
    struct list_head entries;
    int count;
    spinlock_t lock;
};

/* Scheduler state */
struct scheduler_state {
    atomic_t current_algorithm;
    int weights[PROVIDER_COUNT];      /* For weighted scheduling */
    int priorities[PROVIDER_COUNT];   /* For priority scheduling */
    struct provider_metrics metrics[PROVIDER_COUNT];
    struct fifo_queue fifo;           /* Added FIFO queue */
    spinlock_t lock;
    
    /* Dynamic adjustment parameters */
    bool auto_adjust;                 /* Automatically adjust weights based on performance */
    unsigned long adjust_interval;    /* Interval between adjustments (in jiffies) */
    ktime_t last_adjustment;          /* Last time weights were adjusted */
};

/* Function prototypes for providers */
int llm_send_openai(const char *api_key,
                    struct llm_request *req,
                    struct llm_response *resp);
                    
int llm_send_anthropic(const char *api_key,
                       struct llm_request *req,
                       struct llm_response *resp);
                       
int llm_send_google_gemini(const char *api_key,
                           struct llm_request *req,
                           struct llm_response *resp);

/* Context management functions */
int context_add_entry(int conversation_id, const char *role, const char *content);
int context_get_conversation(int conversation_id, struct llm_json_buffer *json_buf);
int context_clear_conversation(int conversation_id);
void context_cleanup_all(void);
int context_get_entry_count(int conversation_id);
int context_prune_old_conversations(unsigned long age_threshold_ms);

/* Scheduler functions */
int select_provider(struct llm_request *req, struct scheduler_state *state);
void update_provider_metrics(int provider, int status, unsigned long latency_ms, int tokens_used);
void scheduler_init(struct scheduler_state *state);
void scheduler_reset_metrics(struct scheduler_state *state);
int fifo_enqueue_provider(struct fifo_queue *queue, int provider);
int fifo_dequeue_provider(struct fifo_queue *queue);
void fifo_init(struct fifo_queue *queue);
void fifo_cleanup(struct fifo_queue *queue);
void adjust_scheduler_weights(struct scheduler_state *state);

/* Rate limiting functions */
int check_rate_limit(int provider, struct scheduler_state *state);
void handle_rate_limit(int provider, struct scheduler_state *state, unsigned long reset_time_ms);

/* Helper functions */
int append_json_string(struct llm_json_buffer *buf, const char *str);
int append_json_value(struct llm_json_buffer *buf, const char *value);
int append_json_number(struct llm_json_buffer *buf, int number);
int append_json_float(struct llm_json_buffer *buf, int value_x100);
int append_json_boolean(struct llm_json_buffer *buf, bool value);
int json_buffer_init(struct llm_json_buffer *buf, size_t size);
void json_buffer_free(struct llm_json_buffer *buf);
int extract_response_content(const char *json, char *output, size_t output_size);
int parse_token_count(const char *json, int *prompt_tokens, int *completion_tokens, int *total_tokens);

/* Model selection helpers */
const char *get_default_model(int provider);
bool is_model_supported(int provider, const char *model_name);

#endif /* LLM_ORCHESTRATOR_H */

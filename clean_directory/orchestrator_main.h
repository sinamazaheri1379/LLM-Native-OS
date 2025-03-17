#ifndef ORCHESTRATOR_MAIN_H
#define ORCHESTRATOR_MAIN_H

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>
#include <linux/device.h>
/* Constants for buffer sizes, error codes */
#define DRIVER_VERSION "2.0"
#define MAX_PROMPT_LENGTH 4096
#define MAX_RESPONSE_LENGTH 8192
#define MAX_MODEL_NAME 64
#define MAX_ROLE_NAME 32
#define MAX_CONTENT_LENGTH 4096
#define MAX_PAYLOAD_SIZE (32 * 1024)
#define MAX_CONTEXT_ENTRIES 100
#define MAX_FIFO_QUEUE_SIZE 64

/* Error codes */
#define LLM_ERR_RATE_LIMIT 100
#define LLM_ERR_AUTH 101
#define LLM_ERR_API_RESPONSE 102

/* Provider definitions */
#define PROVIDER_COUNT 3
#define PROVIDER_OPENAI 0
#define PROVIDER_ANTHROPIC 1
#define PROVIDER_GOOGLE_GEMINI 2

/* Scheduler algorithms */
#define SCHEDULER_ROUND_ROBIN 0
#define SCHEDULER_WEIGHTED 1
#define SCHEDULER_PRIORITY 2
#define SCHEDULER_PERFORMANCE 3
#define SCHEDULER_COST_AWARE 4
#define SCHEDULER_FALLBACK 5
#define SCHEDULER_FIFO 6
#define SCHEDULER_MAX_ALGORITHM SCHEDULER_FIFO
/* Data structures for requests, responses, contexts */
extern spinlock_t conversations_lock;
/* Request timeout handling */
struct request_timeout_data {
    struct socket *sock;
    struct timer_list timer;
    atomic_t *completed_flag;
};
struct llm_json_buffer {
    char *data;
    size_t size;
    size_t used;
};

struct llm_request {
    char prompt[MAX_PROMPT_LENGTH];
    char role[MAX_ROLE_NAME];
    char model_name[MAX_MODEL_NAME];
    int conversation_id;
    int max_tokens;
    int temperature_x100;
    unsigned long timeout_ms;
    int scheduler_algorithm;
    int priority;
};

struct llm_response {
    char content[MAX_RESPONSE_LENGTH];
    size_t content_length;
    int status;
    int provider_used;
    char model_used[MAX_MODEL_NAME];
    ktime_t timestamp;
    s64 latency_ms;
    int tokens_used;
    unsigned long rate_limit_reset_ms;
};

struct context_entry {
    char role[MAX_ROLE_NAME];
    char content[MAX_CONTENT_LENGTH];
    ktime_t timestamp;
    struct list_head list;
};

struct conversation_context {
    int conversation_id;
    int entry_count;
    ktime_t last_updated;
    struct list_head entries;
    spinlock_t lock;
    struct list_head list;
};

struct provider_metrics {
    atomic_t current_status;
    atomic_t total_requests;
    atomic_t successful_requests;
    atomic_t failed_requests;
    atomic_t timeouts;
    atomic_t rate_limited;
    atomic64_t total_latency_ms;
    unsigned long min_latency_ms;
    unsigned long max_latency_ms;
    atomic_t total_tokens;
    ktime_t rate_limit_reset_time;
};

struct fifo_queue {
    int providers[MAX_FIFO_QUEUE_SIZE];  // Add this line
    struct llm_request requests[MAX_FIFO_QUEUE_SIZE]; // Keep this if needed
    int head;
    int tail;
    int count;
    spinlock_t lock;
};

struct scheduler_state {
    atomic_t current_algorithm;
    struct provider_metrics metrics[PROVIDER_COUNT];
    int weights[PROVIDER_COUNT];
    int provider_priority[PROVIDER_COUNT];
    int next_provider;
    struct fifo_queue fifo;
    bool auto_adjust;
};


/* FIFO queue functions */
void fifo_init(struct fifo_queue *fifo);
int fifo_add_provider(struct fifo_queue *fifo, int provider);
/* Function declarations for all components */
/* Context management functions */
int context_add_entry(int conversation_id, const char *role, const char *content);
int context_get_conversation(int conversation_id, struct llm_json_buffer *json_buf);
int context_get_entry_count(int conversation_id);
int context_clear_conversation(int conversation_id);
int context_prune_old_conversations(unsigned long age_threshold_ms);
void context_cleanup_all(void);

/* Memory management functions */
int context_register_memory(int conversation_id, size_t size);
void context_unregister_memory(int conversation_id, size_t size);
void context_get_memory_stats(size_t *total_used, size_t *max_total,
                              int *conversation_count, int *max_conversations);
int context_set_memory_limits(size_t max_total, size_t max_per_conversation,
                              size_t max_conversations);
void context_cleanup_memory_tracking(void);
struct context_entry *context_allocate_entry(int conversation_id);
void context_free_entry(int conversation_id, struct context_entry *entry);

/* JSON utility functions */
int json_buffer_init(struct llm_json_buffer *buf, size_t size);
void json_buffer_free(struct llm_json_buffer *buf);
int append_json_string(struct llm_json_buffer *buf, const char *str);
int append_json_value(struct llm_json_buffer *buf, const char *value);
int append_json_number(struct llm_json_buffer *buf, int number);
int append_json_float(struct llm_json_buffer *buf, int value_x100);
int append_json_boolean(struct llm_json_buffer *buf, bool value);
int extract_response_content(const char *json, char *output, size_t output_size);
int parse_token_count(const char *json, int *prompt_tokens, int *completion_tokens, int *total_tokens);

/* Scheduler functions */
void scheduler_init(struct scheduler_state *state);
int select_provider(struct llm_request *req, struct scheduler_state *state);
void update_provider_metrics(int provider, int result, s64 latency_ms, int tokens);
void handle_rate_limit(int provider, struct scheduler_state *state, unsigned long reset_ms);
void adjust_scheduler_weights(struct scheduler_state *state);
void scheduler_reset_metrics(struct scheduler_state *state);
void set_scheduler_state(struct scheduler_state *state);
void fifo_cleanup(struct fifo_queue *fifo);

/* Network and TLS functions */
int setup_tls(struct socket *sock);
void cleanup_tls(struct socket *sock);
int establish_connection(struct socket **sock, const char *host_ip, int port, bool use_tls);
int network_send_request(const char *host_ip, int port, const char *http_path,
                         const char *api_key, const char *auth_header, bool use_tls,
                         unsigned long timeout_ms, struct llm_json_buffer *buf,
                         struct llm_response *resp);

/* Provider API functions */
int llm_send_openai(const char *api_key, struct llm_request *req, struct llm_response *resp);
int llm_send_anthropic(const char *api_key, struct llm_request *req, struct llm_response *resp);
int llm_send_google_gemini(const char *api_key, struct llm_request *req, struct llm_response *resp);
const char *get_default_model(int provider);
bool is_model_supported(int provider, const char *model_name);
void remove_scheduler_state(void);

struct scheduler_state *get_scheduler_state(void);
struct conversation_context *find_conversation(int conversation_id);
struct conversation_context *find_conversation_internal(int conversation_id);
/* Memory management subsystem functions */
int memory_management_init(void);
void memory_management_cleanup(void);
bool memory_management_initialized(void);

/* JSON manager subsystem functions */
int json_manager_init(void);
void json_manager_cleanup(void);

/* Context management subsystem functions */
int context_management_init(void);
void context_management_cleanup(void);
void context_get_stats(int *total_conversations, int *total_entries,
                      int *entries_added_count, int *entries_pruned_count);

/* Network subsystem functions */
int network_init(void);
void network_cleanup(void);

/* TLS subsystem functions */
int tls_init(void);
void tls_cleanup(void);
/* JSON manager additional functions */
bool json_manager_initialized_check(void);
int json_buffer_resize(struct llm_json_buffer *buf, size_t new_size);
bool validate_json(const char *json);
int extract_response_content_improved(const char *json, char *output, size_t output_size);
void json_get_stats(int *buffers_created_count, int *buffers_resized_count,
                    int *parse_attempts_count, int *parse_successes_count);
ssize_t json_stats_show(struct device *dev, struct device_attribute *attr, char *buf);
/* Memory statistics functions */
ssize_t memory_stats_show(struct device *dev, struct device_attribute *attr, char *buf);
ssize_t memory_limits_show(struct device *dev, struct device_attribute *attr, char *buf);
ssize_t memory_limits_store(struct device *dev, struct device_attribute *attr,
                          const char *buf, size_t count);
ssize_t context_stats_show(struct device *dev, struct device_attribute *attr, char *buf);
bool context_management_initialized(void);

/* Add to orchestrator_main.h */
int tls_send(struct socket *sock, void *data, size_t len);
int tls_recv(struct socket *sock, void *data, size_t len, int flags);
#endif /* ORCHESTRATOR_MAIN_H */
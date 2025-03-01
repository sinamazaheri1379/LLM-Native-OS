#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/random.h>
#include <linux/jiffies.h>
#include <linux/slab.h>
#include <linux/ktime.h>
#include <linux/sched.h>
#include <linux/delay.h>
#include "llm_orchestrator.h"


/* Global scheduler state */
static struct scheduler_state *global_scheduler_state = NULL;
static DEFINE_SPINLOCK(global_scheduler_lock);

/* Get the global scheduler state */
struct scheduler_state *get_scheduler_state(void)
{
    return global_scheduler_state;
}

/* Set the global scheduler state */
void set_scheduler_state(struct scheduler_state *state)
{
    unsigned long flags;

    spin_lock_irqsave(&global_scheduler_lock, flags);
    global_scheduler_state = state;
    spin_unlock_irqrestore(&global_scheduler_lock, flags);
}


/*
 * Initialize scheduler state with default values
 */
void scheduler_init(struct scheduler_state *state)
{
    int i;
    
    if (!state)
        return;
        
    spin_lock_init(&state->lock);
    atomic_set(&state->current_algorithm, SCHEDULER_ROUND_ROBIN);
    
    /* Initialize weights for weighted scheduling */
    state->weights[PROVIDER_OPENAI] = 40;      /* 40% */
    state->weights[PROVIDER_ANTHROPIC] = 30;   /* 30% */
    state->weights[PROVIDER_GOOGLE_GEMINI] = 30; /* 30% */
    
    /* Initialize priorities for priority scheduling */
    state->priorities[PROVIDER_OPENAI] = 2;      /* Highest priority */
    state->priorities[PROVIDER_ANTHROPIC] = 1;   /* Medium priority */
    state->priorities[PROVIDER_GOOGLE_GEMINI] = 0; /* Lowest priority */
    
    /* Initialize metrics for each provider */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        struct provider_metrics *m = &state->metrics[i];
        spin_lock_init(&m->lock);
        atomic_set(&m->total_requests, 0);
        atomic_set(&m->successful_requests, 0);
        atomic_set(&m->failed_requests, 0);
        atomic_set(&m->timeouts, 0);
        atomic_set(&m->rate_limited, 0);
        atomic64_set(&m->total_latency_ms, 0);
        m->min_latency_ms = ULONG_MAX;
        m->max_latency_ms = 0;
        atomic64_set(&m->last_success_jiffies, 0);
        atomic_set(&m->current_status, PROVIDER_STATUS_OK);
        
        /* Initialize token usage tracking */
        atomic_set(&m->total_tokens, 0);
        atomic_set(&m->prompt_tokens, 0);
        atomic_set(&m->completion_tokens, 0);
        
        /* Initialize quota tracking */
        atomic_set(&m->remaining_quota, -1); /* -1 means unlimited */
        m->quota_reset_time = ktime_set(0, 0);
    }
    
    /* Initialize FIFO queue */
    fifo_init(&state->fifo);
    
    /* Initialize auto-adjustment parameters */
    state->auto_adjust = true;
    state->adjust_interval = HZ * 60 * 30; /* 30 minutes */
    state->last_adjustment = ktime_get();
}

/*
 * Initialize FIFO queue
 */
void fifo_init(struct fifo_queue *queue)
{
    if (!queue)
        return;
        
    INIT_LIST_HEAD(&queue->entries);
    queue->count = 0;
    spin_lock_init(&queue->lock);
}

/*
 * Clean up FIFO queue
 */
void fifo_cleanup(struct fifo_queue *queue)
{
    struct fifo_entry *entry, *tmp;
    unsigned long flags;
    
    if (!queue)
        return;
        
    spin_lock_irqsave(&queue->lock, flags);
    
    list_for_each_entry_safe(entry, tmp, &queue->entries, list) {
        list_del(&entry->list);
        kfree(entry);
    }
    
    queue->count = 0;
    
    spin_unlock_irqrestore(&queue->lock, flags);
}

/*
 * Enqueue a provider in the FIFO queue
 * Returns 0 on success, negative error on failure
 */
int fifo_enqueue_provider(struct fifo_queue *queue, int provider)
{
    struct fifo_entry *entry;
    unsigned long flags;
    int ret = 0;

    if (!queue || provider < 0 || provider >= PROVIDER_COUNT)
        return -EINVAL;

    spin_lock_irqsave(&queue->lock, flags);

    /* Check if queue is full */
    if (queue->count >= MAX_FIFO_QUEUE_SIZE) {
        ret = -ENOSPC;
        spin_unlock_irqrestore(&queue->lock, flags);
        return ret;
    }

    /* Allocate new entry */
    entry = kmalloc(sizeof(*entry), GFP_ATOMIC);
    if (!entry) {
        ret = -ENOMEM;
        spin_unlock_irqrestore(&queue->lock, flags);
        return ret;
    }

    entry->provider = provider;
    entry->enqueue_time = ktime_get();

    /* Add to end of queue */
    list_add_tail(&entry->list, &queue->entries);
    queue->count++;

    spin_unlock_irqrestore(&queue->lock, flags);

    return 0;
}

/*
 * Dequeue a provider from the FIFO queue
 * Returns provider ID on success, negative error on failure
 */
int fifo_dequeue_provider(struct fifo_queue *queue)
{
    struct fifo_entry *entry;
    unsigned long flags;
    int provider;
    
    if (!queue)
        return -EINVAL;
        
    spin_lock_irqsave(&queue->lock, flags);
    
    /* Check if queue is empty */
    if (list_empty(&queue->entries)) {
        spin_unlock_irqrestore(&queue->lock, flags);
        return -ENOENT;
    }
    
    /* Get first entry */
    entry = list_first_entry(&queue->entries, struct fifo_entry, list);
    provider = entry->provider;
    
    /* Remove from queue */
    list_del(&entry->list);
    queue->count--;
    
    spin_unlock_irqrestore(&queue->lock, flags);
    
    /* Print diagnostic info */
    pr_debug("FIFO dequeued provider %d, queue wait time: %lld ms\n",
             provider, ktime_to_ms(ktime_sub(ktime_get(), entry->enqueue_time)));
    
    /* Free entry */
    kfree(entry);
    
    return provider;
}

/*
 * Reset metrics for all providers
 */
void scheduler_reset_metrics(struct scheduler_state *state)
{
    int i;
    unsigned long flags;
    
    if (!state)
        return;
        
    spin_lock_irqsave(&state->lock, flags);
    for (i = 0; i < PROVIDER_COUNT; i++) {
        struct provider_metrics *m = &state->metrics[i];
        unsigned long m_flags;
        
        spin_lock_irqsave(&m->lock, m_flags);
        
        atomic_set(&m->total_requests, 0);
        atomic_set(&m->successful_requests, 0);
        atomic_set(&m->failed_requests, 0);
        atomic_set(&m->timeouts, 0);
        atomic_set(&m->rate_limited, 0);
        atomic64_set(&m->total_latency_ms, 0);
        m->min_latency_ms = ULONG_MAX;
        m->max_latency_ms = 0;
        /* We don't reset last_success_jiffies or current_status */
        
        /* Reset token usage tracking */
        atomic_set(&m->total_tokens, 0);
        atomic_set(&m->prompt_tokens, 0);
        atomic_set(&m->completion_tokens, 0);
        
        spin_unlock_irqrestore(&m->lock, m_flags);
    }
    
    /* Reset FIFO queue */
    fifo_cleanup(&state->fifo);
    fifo_init(&state->fifo);
    
    spin_unlock_irqrestore(&state->lock, flags);
    
    pr_info("Scheduler metrics have been reset\n");
}

/*
 * Update metrics for a provider after a request
 */
void update_provider_metrics(int provider, int status, unsigned long latency_ms, int tokens_used)
{
    /* We should pass state explicitly instead of accessing task_struct */
    struct scheduler_state *state;
    struct provider_metrics *m;
    unsigned long flags;

    /* Get scheduler state from a global or pass it as parameter */
    state = get_scheduler_state(); /* This function needs to be implemented */

    if (!state || provider < 0 || provider >= PROVIDER_COUNT)
        return;

    m = &state->metrics[provider];

    spin_lock_irqsave(&m->lock, flags);

    /* Update request counters */
    atomic_inc(&m->total_requests);

    /* Update based on status */
    if (status == 0) {
        atomic_inc(&m->successful_requests);
        atomic64_set(&m->last_success_jiffies, jiffies);
        atomic_set(&m->current_status, PROVIDER_STATUS_OK);
        atomic64_add(latency_ms, &m->total_latency_ms);

        /* Update latency stats */
        if (latency_ms < m->min_latency_ms)
            m->min_latency_ms = latency_ms;
        if (latency_ms > m->max_latency_ms)
            m->max_latency_ms = latency_ms;

        /* Update token usage */
        if (tokens_used > 0) {
            atomic_add(tokens_used, &m->total_tokens);
        }
    } else if (status == -ETIMEDOUT) {
        atomic_inc(&m->timeouts);
        atomic_set(&m->current_status, PROVIDER_STATUS_TIMEOUT);
    } else if (status == -LLM_ERR_RATE_LIMIT) {
        atomic_inc(&m->rate_limited);
        atomic_set(&m->current_status, PROVIDER_STATUS_RATE_LIMITED);
    } else {
        atomic_inc(&m->failed_requests);
        atomic_set(&m->current_status, PROVIDER_STATUS_ERROR);
    }

    spin_unlock_irqrestore(&m->lock, flags);

    /* Check if we should adjust weights */
    if (state->auto_adjust &&
        ktime_after(ktime_get(), ktime_add_ms(state->last_adjustment,
                                              jiffies_to_msecs(state->adjust_interval)))) {
        adjust_scheduler_weights(state);
        state->last_adjustment = ktime_get();
    }
}

/*
 * Dynamically adjust weights based on provider performance
 */
void adjust_scheduler_weights(struct scheduler_state *state)
{
    int i;
    unsigned long flags;
    unsigned long avg_latencies[PROVIDER_COUNT];
    int success_rates[PROVIDER_COUNT];
    unsigned long new_weights[PROVIDER_COUNT];
    unsigned long total_weight = 0;

    if (!state)
        return;

    /* Calculate metrics for each provider */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        struct provider_metrics *m = &state->metrics[i];
        int total = atomic_read(&m->total_requests);
        int success = atomic_read(&m->successful_requests);

        if (total > 0) {
            success_rates[i] = (success * 100) / total;
        } else {
            success_rates[i] = 0;
        }

        if (success > 0) {
            avg_latencies[i] = div_u64(atomic64_read(&m->total_latency_ms), success);
        } else {
            avg_latencies[i] = ULONG_MAX;
        }
    }

    /* Adjust weights based on success rate and latency */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (success_rates[i] == 0 || avg_latencies[i] == ULONG_MAX) {
            new_weights[i] = 10; /* Minimum weight */
        } else {
            /* Use safe operations to avoid overflow */
            u64 weight_calc = (u64)success_rates[i] * 1000;
            if (avg_latencies[i] > 0) {
                weight_calc = div64_ul(weight_calc, avg_latencies[i] + 1);
            }

            /* Cap to reasonable values */
            if (weight_calc > ULONG_MAX)
                weight_calc = ULONG_MAX;

            /* Ensure minimum weight */
            new_weights[i] = (unsigned long)weight_calc;
            if (new_weights[i] < 10)
                new_weights[i] = 10;
        }
        total_weight += new_weights[i];
    }

    /* Normalize weights to total 100 */
    if (total_weight > 0) {
        spin_lock_irqsave(&state->lock, flags);

        for (i = 0; i < PROVIDER_COUNT; i++) {
            state->weights[i] = (int)(((u64)new_weights[i] * 100) / total_weight);

            /* Ensure minimum 5% weight */
            if (state->weights[i] < 5)
                state->weights[i] = 5;
        }

        /* Adjust to ensure weights sum to 100 */
        total_weight = 0;
        for (i = 0; i < PROVIDER_COUNT; i++) {
            total_weight += state->weights[i];
        }

        if (total_weight != 100) {
            int diff = 100 - (int)total_weight;
            state->weights[0] += diff; /* Adjust first provider to make sum 100 */
        }

        spin_unlock_irqrestore(&state->lock, flags);

        pr_info("Scheduler weights adjusted: OpenAI=%d%%, Anthropic=%d%%, Gemini=%d%%\n",
                state->weights[PROVIDER_OPENAI],
                state->weights[PROVIDER_ANTHROPIC],
                state->weights[PROVIDER_GOOGLE_GEMINI]);
    }
}

/*
 * Round-robin scheduler - simple and fair rotation
 */
static int round_robin_scheduler(struct scheduler_state *state)
{
    static atomic_t next_provider = ATOMIC_INIT(0);
    int provider;
    
    provider = atomic_inc_return(&next_provider) % PROVIDER_COUNT;
    
    return provider;
}

/*
 * FIFO scheduler - uses a first-in-first-out queue
 */
static int fifo_scheduler(struct scheduler_state *state)
{
    int provider;
    
    /* First try to dequeue */
    provider = fifo_dequeue_provider(&state->fifo);
    
    /* If queue is empty, use round-robin and enqueue the others */
    if (provider < 0) {
        provider = round_robin_scheduler(state);
        
        /* Enqueue the other providers (in order) */
        for (int i = 0; i < PROVIDER_COUNT; i++) {
            if (i != provider) {
                fifo_enqueue_provider(&state->fifo, i);
            }
        }
    }
    
    return provider;
}

/*
 * Weighted scheduler - distributes requests according to weights
 */
static int weighted_scheduler(struct scheduler_state *state)
{
    int random_value;
    int sum = 0;
    int i;
    int provider_status[PROVIDER_COUNT];
    unsigned long flags;

    /* Get random value between 0 and 99 */
    get_random_bytes(&random_value, sizeof(random_value));
    random_value = (random_value & 0x7FFFFFFF) % 100;

    /* Get status of all providers with proper locking */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        spin_lock_irqsave(&state->metrics[i].lock, flags);
        provider_status[i] = atomic_read(&state->metrics[i].current_status);
        spin_unlock_irqrestore(&state->metrics[i].lock, flags);
    }

    /* Select provider based on weighted distribution */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (provider_status[i] != PROVIDER_STATUS_OK) {
            continue; /* Skip providers with non-OK status */
        }

        sum += state->weights[i];
        if (random_value < sum)
            return i;
    }

    /* If all providers have errors, try again without status check */
    sum = 0;
    for (i = 0; i < PROVIDER_COUNT; i++) {
        sum += state->weights[i];
        if (random_value < sum)
            return i;
    }

    /* Fallback to last provider */
    return PROVIDER_COUNT - 1;
}

/*
 * Priority scheduler - uses provider priorities
 */
static int priority_scheduler(struct scheduler_state *state)
{
    int best_provider = 0;
    int highest_priority = -1;
    int i;
    int provider_status[PROVIDER_COUNT];
    unsigned long flags;

    /* Get status of all providers with proper locking */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        spin_lock_irqsave(&state->metrics[i].lock, flags);
        provider_status[i] = atomic_read(&state->metrics[i].current_status);
        spin_unlock_irqrestore(&state->metrics[i].lock, flags);
    }

    /* Select provider based on priority */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        /* Skip providers with error status */
        if (provider_status[i] != PROVIDER_STATUS_OK)
            continue;
            
        if (state->priorities[i] > highest_priority) {
            highest_priority = state->priorities[i];
            best_provider = i;
        }
    }
    
    /* If all providers have errors, return the highest priority one anyway */
    if (highest_priority == -1) {
        for (i = 0; i < PROVIDER_COUNT; i++) {
            if (state->priorities[i] > highest_priority) {
                highest_priority = state->priorities[i];
                best_provider = i;
            }
        }
    }
    
    return best_provider;
}

/*
 * Performance-based scheduler - selects provider with lowest average latency
 */
static int performance_scheduler(struct scheduler_state *state)
{
    int best_provider = 0;
    unsigned long lowest_avg_latency = ULONG_MAX;
    int i;
    unsigned long avg_latency;
    int provider_status[PROVIDER_COUNT];
    int successful_requests[PROVIDER_COUNT];
    unsigned long total_latencies[PROVIDER_COUNT];
    unsigned long flags;

    /* Get metrics for all providers with proper locking */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        spin_lock_irqsave(&state->metrics[i].lock, flags);
        provider_status[i] = atomic_read(&state->metrics[i].current_status);
        successful_requests[i] = atomic_read(&state->metrics[i].successful_requests);
        total_latencies[i] = atomic64_read(&state->metrics[i].total_latency_ms);
        spin_unlock_irqrestore(&state->metrics[i].lock, flags);
    }

    /* Select provider based on average latency */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        /* Skip providers with no successful requests */
        if (successful_requests[i] == 0)
            continue;

        /* Skip providers with error status */
        if (provider_status[i] != PROVIDER_STATUS_OK)
            continue;

        avg_latency = total_latencies[i] / successful_requests[i];
        
        if (avg_latency < lowest_avg_latency) {
            lowest_avg_latency = avg_latency;
            best_provider = i;
        }
    }
    
    /* If no provider has successful requests, use round robin */
    if (lowest_avg_latency == ULONG_MAX)
        return round_robin_scheduler(state);
        
    return best_provider;
}

/*
 * Cost-aware scheduler - balances between providers to optimize cost
 * This is a simple implementation that assumes cost is correlated with weights
 */
static int cost_aware_scheduler(struct scheduler_state *state)
{
    /* Use weighted scheduler but with inverse weights */
    int random_value;
    int total_weights = 0;
    int inverse_weights[PROVIDER_COUNT];
    int sum = 0;
    int i;
    
    /* Calculate inverse weights */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (state->weights[i] > 0)
            inverse_weights[i] = 100 / state->weights[i];
        else
            inverse_weights[i] = 100; /* Avoid division by zero */
        total_weights += inverse_weights[i];
    }
    
    /* Normalize inverse weights to sum to 100 */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        inverse_weights[i] = (inverse_weights[i] * 100) / total_weights;
    }
    
    /* Get random value between 0 and 99 */
    get_random_bytes(&random_value, sizeof(random_value));
    random_value = (random_value & 0x7FFFFFFF) % 100;
    
    /* Select provider based on inverse weighted distribution */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        sum += inverse_weights[i];
        if (random_value < sum)
            return i;
    }
    
    /* Fallback to last provider */
    return PROVIDER_COUNT - 1;
}

/*
 * Fallback scheduler - tries providers in order of preference
 * until one succeeds
 */
static int fallback_scheduler(struct scheduler_state *state)
{
    int i;
    int priority_order[PROVIDER_COUNT] = {
            PROVIDER_OPENAI,         /* First try OpenAI */
            PROVIDER_ANTHROPIC,      /* Then try Anthropic */
            PROVIDER_GOOGLE_GEMINI   /* Finally try Google Gemini */
    };

    /* Try each provider in priority order */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        int provider = priority_order[i];

        /* Check if this provider is available */
        if (check_rate_limit(provider, state) == 0) {
            return provider;
        }
    }

    /* If all are rate limited, return the first one anyway */
    return priority_order[0];
}

/*
 * Check if a provider is rate limited
 * Returns 0 if not rate limited, negative error code if rate limited
 */
int check_rate_limit(int provider, struct scheduler_state *state)
{
    struct provider_metrics *m;
    unsigned long flags;
    int status;
    ktime_t now, reset_time;

    if (!state || provider < 0 || provider >= PROVIDER_COUNT)
        return -EINVAL;

    m = &state->metrics[provider];

    spin_lock_irqsave(&m->lock, flags);

    status = atomic_read(&m->current_status);
    if (status == PROVIDER_STATUS_RATE_LIMITED) {
        now = ktime_get();
        reset_time = m->quota_reset_time;

        if (ktime_before(now, reset_time)) {
            /* Still rate limited */
            spin_unlock_irqrestore(&m->lock, flags);
            return -LLM_ERR_RATE_LIMIT;
        } else {
            /* Rate limit expired, reset status */
            atomic_set(&m->current_status, PROVIDER_STATUS_OK);
        }
    }

    spin_unlock_irqrestore(&m->lock, flags);

    return 0;
}

/*
 * Handle rate limiting for a provider
 */
void handle_rate_limit(int provider, struct scheduler_state *state, unsigned long reset_time_ms)
{
    struct provider_metrics *m;
    
    if (!state || provider < 0 || provider >= PROVIDER_COUNT)
        return;
        
    m = &state->metrics[provider];
    
    atomic_inc(&m->rate_limited);
    atomic_set(&m->current_status, PROVIDER_STATUS_RATE_LIMITED);
    
    /* Set the time when the rate limit will expire */
    m->quota_reset_time = ktime_add_ms(ktime_get(), reset_time_ms);
    
    pr_warn("Provider %d rate limited, will reset in %lu ms\n", provider, reset_time_ms);
}

/*
 * Main function to select a provider based on the chosen algorithm
 */
int select_provider(struct llm_request *req, struct scheduler_state *state)
{
    int provider = -1;
    unsigned long flags;
    int current_algorithm;
    
    if (!req || !state)
        return -EINVAL;
        
    /* If user specified a preference, use that */
    if (req->provider_preference >= 0 && req->provider_preference < PROVIDER_COUNT) {
        /* Check if preferred provider is rate limited */
        if (check_rate_limit(req->provider_preference, state) == 0) {
            return req->provider_preference;
        } else {
            pr_warn("Preferred provider %d is rate limited, falling back to scheduler\n", 
                   req->provider_preference);
        }
    }
    
    /* Use the algorithm specified in the request, or the default */
    if (req->scheduler_algorithm >= 0 && req->scheduler_algorithm <= SCHEDULER_FIFO) {
        atomic_set(&state->current_algorithm, req->scheduler_algorithm);
    }
    
    current_algorithm = atomic_read(&state->current_algorithm);
    
    /* Try to select a provider that's not rate limited */
    for (int attempt = 0; attempt < 3; attempt++) {
        /* Select provider based on current algorithm */
        switch (current_algorithm) {
            case SCHEDULER_ROUND_ROBIN:
                provider = round_robin_scheduler(state);
                break;
            case SCHEDULER_WEIGHTED:
                provider = weighted_scheduler(state);
                break;
            case SCHEDULER_PRIORITY:
                provider = priority_scheduler(state);
                break;
            case SCHEDULER_PERFORMANCE:
                provider = performance_scheduler(state);
                break;
            case SCHEDULER_COST_AWARE:
                provider = cost_aware_scheduler(state);
                break;
            case SCHEDULER_FALLBACK:
                provider = fallback_scheduler(state);
                break;
            case SCHEDULER_FIFO:
                provider = fifo_scheduler(state);
                break;
            default:
                provider = round_robin_scheduler(state);
                break;
        }
        
        /* Check if selected provider is rate limited */
        if (check_rate_limit(provider, state) == 0) {
            break;
        }
        
        /* If rate limited, try a different algorithm for next attempt */
        pr_debug("Provider %d is rate limited, trying different algorithm\n", provider);
        current_algorithm = (current_algorithm + 1) % (SCHEDULER_FIFO + 1);
        
        /* Small delay to prevent tight loop */
        if (attempt < 2) {
            usleep_range(10000, 20000); /* 10-20ms */
        }
    }
    
    /* Log the selected provider and algorithm */
    pr_debug("Selected provider %d using algorithm %d\n", provider, current_algorithm);
    
    return provider;
}

/*
 * Model management helper functions
 */

/*
 * Get default model for a provider
 */
const char *get_default_model(int provider)
{
    switch(provider) {
        case PROVIDER_OPENAI:
            return "gpt-4o"; /* Most capable model as default */
        case PROVIDER_ANTHROPIC:
            return "claude-3-7-sonnet-20250219";
        case PROVIDER_GOOGLE_GEMINI:
            return "gemini-1.5-pro";
        default:
            return "";
    }
}

/*
 * Check if a model is supported by a provider
 */
bool is_model_supported(int provider, const char *model_name)
{
    if (!model_name || model_name[0] == '\0')
        return false;
        
    switch(provider) {
        case PROVIDER_OPENAI:
            /* OpenAI models */
            return (strncmp(model_name, "gpt-", 4) == 0);
            
        case PROVIDER_ANTHROPIC:
            /* Anthropic models */
            return (strncmp(model_name, "claude-", 7) == 0);
            
        case PROVIDER_GOOGLE_GEMINI:
            /* Google Gemini models */
            return (strncmp(model_name, "gemini-", 7) == 0);
            
        default:
            return false;
    }
}

/* Module exports */
EXPORT_SYMBOL(scheduler_init);
EXPORT_SYMBOL(scheduler_reset_metrics);
EXPORT_SYMBOL(update_provider_metrics);
EXPORT_SYMBOL(select_provider);
EXPORT_SYMBOL(fifo_init);
EXPORT_SYMBOL(fifo_cleanup);
EXPORT_SYMBOL(fifo_enqueue_provider);
EXPORT_SYMBOL(fifo_dequeue_provider);
EXPORT_SYMBOL(adjust_scheduler_weights);
EXPORT_SYMBOL(check_rate_limit);
EXPORT_SYMBOL(handle_rate_limit);
EXPORT_SYMBOL(get_default_model);
EXPORT_SYMBOL(is_model_supported);

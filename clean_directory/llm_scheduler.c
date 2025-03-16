#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>
#include <linux/atomic.h>
#include <linux/random.h>
#include <linux/slab.h>
#include "orchestrator_main.h"

/* Constants for scheduler logic */
#define WEIGHT_TOTAL_PERCENT     100
#define MIN_PROVIDER_WEIGHT      5
#define RATE_LIMIT_PENALTY       20
#define DEFAULT_TOKEN_WEIGHT     100
#define TOKEN_WEIGHT_FACTOR      1000000
#define METRICS_ADJUST_INTERVAL  10

/* Provider model information */
static const char *openai_default_model = "gpt-4o";
static const char *anthropic_default_model = "claude-3-opus-20240229";
static const char *gemini_default_model = "gemini-1.5-pro";

static const char *openai_supported_models[] = {
        "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", NULL
};

static const char *anthropic_supported_models[] = {
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", NULL
};

static const char *gemini_supported_models[] = {
        "gemini-1.5-pro", "gemini-1.0-pro", NULL
};

/* Provider metrics and weight locks */
static DEFINE_SPINLOCK(metrics_lock);
static DEFINE_SPINLOCK(weights_lock);

/*
 * Get default model for provider
 * Returns the default model string or NULL if provider is invalid
 */
const char *get_default_model(int provider)
{
    if (provider < 0 || provider >= PROVIDER_COUNT)
        return NULL;

    switch (provider) {
        case PROVIDER_OPENAI:
            return openai_default_model;
        case PROVIDER_ANTHROPIC:
            return anthropic_default_model;
        case PROVIDER_GOOGLE_GEMINI:
            return gemini_default_model;
        default:
            return NULL;
    }
}

/*
 * Check if a model is supported by a provider
 * Returns true if model is valid and supported, false otherwise
 */
bool is_model_supported(int provider, const char *model_name)
{
    const char **models;
    int i;

    if (!model_name || !model_name[0])
        return false;

    if (provider < 0 || provider >= PROVIDER_COUNT)
        return false;

    switch (provider) {
        case PROVIDER_OPENAI:
            models = openai_supported_models;
            break;
        case PROVIDER_ANTHROPIC:
            models = anthropic_supported_models;
            break;
        case PROVIDER_GOOGLE_GEMINI:
            models = gemini_supported_models;
            break;
        default:
            return false;
    }

    for (i = 0; models[i] != NULL; i++) {
        if (strcmp(model_name, models[i]) == 0)
            return true;
    }

    return false;
}

/*
 * Initialize scheduler state
 * Handles state validation and proper initialization of all components
 */
void scheduler_init(struct scheduler_state *state)
{
    int i;
    unsigned long flags;

    if (!state) {
        pr_err("scheduler_init: Invalid state pointer\n");
        return;
    }

    /* Initialize algorithm and scheduling parameters */
    atomic_set(&state->current_algorithm, SCHEDULER_ROUND_ROBIN);
    state->next_provider = 0;
    state->auto_adjust = true;

    /* Initialize weights with proper locking */
    spin_lock_irqsave(&weights_lock, flags);

    /* Initial equal weights */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        state->weights[i] = WEIGHT_TOTAL_PERCENT / PROVIDER_COUNT;
        state->provider_priority[i] = i;
    }

    /* Ensure weights add up to 100% */
    state->weights[PROVIDER_COUNT - 1] = WEIGHT_TOTAL_PERCENT;
    for (i = 0; i < PROVIDER_COUNT - 1; i++) {
        state->weights[PROVIDER_COUNT - 1] -= state->weights[i];
    }

    /* Ensure minimum weight for all providers */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (state->weights[i] < MIN_PROVIDER_WEIGHT)
            state->weights[i] = MIN_PROVIDER_WEIGHT;
    }

    spin_unlock_irqrestore(&weights_lock, flags);

    /* Initialize metrics with proper locking */
    spin_lock_irqsave(&metrics_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        atomic_set(&state->metrics[i].current_status, 1); /* Available */
        atomic_set(&state->metrics[i].total_requests, 0);
        atomic_set(&state->metrics[i].successful_requests, 0);
        atomic_set(&state->metrics[i].failed_requests, 0);
        atomic_set(&state->metrics[i].timeouts, 0);
        atomic_set(&state->metrics[i].rate_limited, 0);
        atomic64_set(&state->metrics[i].total_latency_ms, 0);
        state->metrics[i].min_latency_ms = ULONG_MAX;
        state->metrics[i].max_latency_ms = 0;
        atomic_set(&state->metrics[i].total_tokens, 0);
        state->metrics[i].rate_limit_reset_time = 0;
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    /* Initialize FIFO queue */
    spin_lock_init(&state->fifo.lock);
    state->fifo.head = 0;
    state->fifo.tail = 0;
    state->fifo.count = 0;

    pr_info("LLM scheduler initialized\n");
}

/*
 * Check if a provider is available (not rate limited)
 * Returns true if provider is available, false otherwise
 */
static bool is_provider_available(int provider, struct scheduler_state *state)
{
    bool available = false;
    unsigned long flags;

    if (!state) {
        pr_err("is_provider_available: Invalid state pointer\n");
        return false;
    }

    if (provider < 0 || provider >= PROVIDER_COUNT)
        return false;

    spin_lock_irqsave(&metrics_lock, flags);

    if (atomic_read(&state->metrics[provider].current_status) != 0) {
        if (state->metrics[provider].rate_limit_reset_time > 0) {
            ktime_t now = ktime_get();
            if (ktime_after(now, state->metrics[provider].rate_limit_reset_time)) {
                /* Reset limit has passed */
                state->metrics[provider].rate_limit_reset_time = 0;
                atomic_set(&state->metrics[provider].current_status, 1);
                available = true;
            }
        } else {
            available = true;
        }
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    return available;
}

/*
 * Select provider using Round Robin algorithm
 * Returns provider index to use
 */
static int select_round_robin(struct llm_request *req, struct scheduler_state *state)
{
    int starting_provider, provider;
    int i;
    unsigned long flags;

    if (!state) {
        pr_err("select_round_robin: Invalid state pointer\n");
        return 0;
    }

    spin_lock_irqsave(&metrics_lock, flags);
    starting_provider = state->next_provider;
    spin_unlock_irqrestore(&metrics_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        provider = (starting_provider + i) % PROVIDER_COUNT;
        if (is_provider_available(provider, state)) {
            spin_lock_irqsave(&metrics_lock, flags);
            state->next_provider = (provider + 1) % PROVIDER_COUNT;
            spin_unlock_irqrestore(&metrics_lock, flags);
            return provider;
        }
    }

    /* If all providers are unavailable, use the next one anyway and let caller handle error */
    spin_lock_irqsave(&metrics_lock, flags);
    provider = state->next_provider;
    state->next_provider = (provider + 1) % PROVIDER_COUNT;
    spin_unlock_irqrestore(&metrics_lock, flags);

    return provider;
}

/*
 * Select provider using Weighted algorithm
 * Returns provider index to use based on configured weights
 */
static int select_weighted(struct llm_request *req, struct scheduler_state *state)
{
    int random_val, cumulative_weight = 0;
    int i, selected = 0;
    int available_providers[PROVIDER_COUNT];
    int available_weights[PROVIDER_COUNT];
    int available_count = 0, total_available_weight = 0;
    unsigned long flags;

    if (!state) {
        pr_err("select_weighted: Invalid state pointer\n");
        return 0;
    }

    /* Find available providers with proper locking */
    spin_lock_irqsave(&weights_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (is_provider_available(i, state)) {
            available_providers[available_count] = i;
            available_weights[available_count] = state->weights[i];
            total_available_weight += state->weights[i];
            available_count++;
        }
    }

    spin_unlock_irqrestore(&weights_lock, flags);

    /* If no providers available, return the first one and let caller handle error */
    if (available_count == 0)
        return 0;

    /* If only one provider available, return it */
    if (available_count == 1)
        return available_providers[0];

    /* Choose randomly based on weights */
    get_random_bytes(&random_val, sizeof(random_val));
    random_val = abs(random_val) % max(total_available_weight, 1);

    for (i = 0; i < available_count; i++) {
        cumulative_weight += available_weights[i];
        if (random_val < cumulative_weight) {
            selected = available_providers[i];
            break;
        }
    }

    /* Ensure a valid provider is selected */
    if (i == available_count && available_count > 0) {
        selected = available_providers[0];
    }

    return selected;
}

/*
 * Select provider based on priority
 * Returns provider index based on configured priority order
 */
static int select_priority(struct llm_request *req, struct scheduler_state *state)
{
    int i;
    int provider;
    unsigned long flags;

    if (!state) {
        pr_err("select_priority: Invalid state pointer\n");
        return 0;
    }

    spin_lock_irqsave(&weights_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        provider = state->provider_priority[i];
        if (is_provider_available(provider, state)) {
            spin_unlock_irqrestore(&weights_lock, flags);
            return provider;
        }
    }

    /* If all providers unavailable, return highest priority one anyway */
    provider = state->provider_priority[0];

    spin_unlock_irqrestore(&weights_lock, flags);

    return provider;
}

/*
 * Select provider based on performance (lowest latency)
 * Returns provider index with best average latency
 */
static int select_performance(struct llm_request *req, struct scheduler_state *state)
{
    int best_provider = 0;
    unsigned long best_latency = ULONG_MAX;
    int i;
    unsigned long flags;

    if (!state) {
        pr_err("select_performance: Invalid state pointer\n");
        return 0;
    }

    spin_lock_irqsave(&metrics_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (is_provider_available(i, state)) {
            int successful = atomic_read(&state->metrics[i].successful_requests);
            unsigned long avg_latency;

            if (successful > 0) {
                avg_latency = div_u64(atomic64_read(&state->metrics[i].total_latency_ms),
                                      max(successful, 1));
                if (avg_latency < best_latency) {
                    best_latency = avg_latency;
                    best_provider = i;
                }
            } else if (best_latency == ULONG_MAX) {
                /* No data for any provider yet, use the first available one */
                best_provider = i;
            }
        }
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    return best_provider;
}

/*
 * Select provider using FIFO queue
 * Implements a proper FIFO scheduling algorithm
 */
static int select_fifo(struct llm_request *req, struct scheduler_state *state)
{
    int provider = 0;
    unsigned long flags;
    bool found = false;

    if (!state) {
        pr_err("select_fifo: Invalid state pointer\n");
        return 0;
    }

    spin_lock_irqsave(&state->fifo.lock, flags);

    /* Check if queue has entries */
    if (state->fifo.count > 0) {
        /* Get the next provider from the queue */
        provider = state->fifo.providers[state->fifo.head];
        state->fifo.head = (state->fifo.head + 1) % MAX_FIFO_QUEUE_SIZE;
        state->fifo.count--;
        found = true;

        /* Re-add the provider to the queue for round-robin behavior */
        if (state->fifo.count < MAX_FIFO_QUEUE_SIZE) {
            state->fifo.providers[state->fifo.tail] = provider;
            state->fifo.tail = (state->fifo.tail + 1) % MAX_FIFO_QUEUE_SIZE;
            state->fifo.count++;
        }
    }

    spin_unlock_irqrestore(&state->fifo.lock, flags);

    /* If queue is empty, use weighted selection as fallback */
    if (!found) {
        int i;

        /* Initialize the queue with all available providers */
        spin_lock_irqsave(&state->fifo.lock, flags);

        for (i = 0; i < PROVIDER_COUNT; i++) {
            if (is_provider_available(i, state) && state->fifo.count < MAX_FIFO_QUEUE_SIZE) {
                state->fifo.providers[state->fifo.tail] = i;
                state->fifo.tail = (state->fifo.tail + 1) % MAX_FIFO_QUEUE_SIZE;
                state->fifo.count++;
            }
        }

        /* If any providers were added, get the first one */
        if (state->fifo.count > 0) {
            provider = state->fifo.providers[state->fifo.head];
            state->fifo.head = (state->fifo.head + 1) % MAX_FIFO_QUEUE_SIZE;
            state->fifo.count--;
            found = true;

            /* Re-add the provider for round-robin */
            if (state->fifo.count < MAX_FIFO_QUEUE_SIZE) {
                state->fifo.providers[state->fifo.tail] = provider;
                state->fifo.tail = (state->fifo.tail + 1) % MAX_FIFO_QUEUE_SIZE;
                state->fifo.count++;
            }
        }

        spin_unlock_irqrestore(&state->fifo.lock, flags);

        /* If still no providers, fall back to weighted selection */
        if (!found) {
            provider = select_weighted(req, state);
        }
    }

    return provider;
}

/*
 * Select provider based on cost awareness
 * Returns provider index based on token usage (lower usage gets higher priority)
 */
static int select_cost_aware(struct llm_request *req, struct scheduler_state *state)
{
    int token_weights[PROVIDER_COUNT];
    int total_token_weight = 0;
    int random_val, cumulative_weight = 0;
    int i, selected = 0;
    unsigned long flags;

    if (!state) {
        pr_err("select_cost_aware: Invalid state pointer\n");
        return 0;
    }

    /* Calculate inverse weight based on token usage with proper locking */
    spin_lock_irqsave(&metrics_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (is_provider_available(i, state)) {
            int tokens = atomic_read(&state->metrics[i].total_tokens);
            /* Higher token count = lower weight, prevent division by zero */
            token_weights[i] = tokens > 0 ? TOKEN_WEIGHT_FACTOR / tokens : DEFAULT_TOKEN_WEIGHT;
            total_token_weight += token_weights[i];
        } else {
            token_weights[i] = 0;
        }
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    /* If no providers available, return first one */
    if (total_token_weight == 0)
        return 0;

    /* Choose randomly based on inverse token weights */
    get_random_bytes(&random_val, sizeof(random_val));
    random_val = abs(random_val) % max(total_token_weight, 1);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (token_weights[i] > 0) {
            cumulative_weight += token_weights[i];
            if (random_val < cumulative_weight) {
                selected = i;
                break;
            }
        }
    }

    /* Ensure a valid selection */
    if (i == PROVIDER_COUNT) {
        /* No provider selected, find first available one */
        for (i = 0; i < PROVIDER_COUNT; i++) {
            if (is_provider_available(i, state)) {
                selected = i;
                break;
            }
        }
    }

    return selected;
}

/*
 * Implements the fallback scheduling algorithm
 * Tries all providers in order of priority
 */
static int select_fallback(struct llm_request *req, struct scheduler_state *state)
{
    int i;

    if (!state) {
        pr_err("select_fallback: Invalid state pointer\n");
        return 0;
    }

    /* Try each provider in order */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (is_provider_available(i, state)) {
            return i;
        }
    }

    /* If all are unavailable, return the first one */
    return 0;
}

/*
 * Select provider based on request and scheduler state
 * Main entry point for provider selection
 */
int select_provider(struct llm_request *req, struct scheduler_state *state)
{
    int algorithm;
    int provider;

    if (!req || !state) {
        pr_err("select_provider: Invalid request or state pointer\n");
        return 0;
    }

    algorithm = req->scheduler_algorithm;

    if (algorithm == -1)
        algorithm = atomic_read(&state->current_algorithm);

    /* Validate algorithm */
    if (algorithm < 0 || algorithm > SCHEDULER_MAX_ALGORITHM) {
        pr_warn("select_provider: Invalid algorithm %d, using weighted\n", algorithm);
        algorithm = SCHEDULER_WEIGHTED;
    }

    switch (algorithm) {
        case SCHEDULER_ROUND_ROBIN:
            provider = select_round_robin(req, state);
            break;
        case SCHEDULER_WEIGHTED:
            provider = select_weighted(req, state);
            break;
        case SCHEDULER_PRIORITY:
            provider = select_priority(req, state);
            break;
        case SCHEDULER_PERFORMANCE:
            provider = select_performance(req, state);
            break;
        case SCHEDULER_COST_AWARE:
            provider = select_cost_aware(req, state);
            break;
        case SCHEDULER_FIFO:
            provider = select_fifo(req, state);
            break;
        case SCHEDULER_FALLBACK:
            provider = select_fallback(req, state);
            break;
        default:
            provider = select_weighted(req, state);
            break;
    }

    /* Ensure provider is valid */
    if (provider < 0 || provider >= PROVIDER_COUNT) {
        pr_warn("select_provider: Invalid provider %d, using 0\n", provider);
        provider = 0;
    }

    return provider;
}

/*
 * Update provider metrics after a request
 * Safely update all metrics with proper locking
 */
void update_provider_metrics(int provider, int result, s64 latency_ms, int tokens)
{
    struct scheduler_state *state;
    unsigned long flags;
    bool should_adjust = false;

    /* Get state from current task */
    state = get_scheduler_state();

    if (!state || provider < 0 || provider >= PROVIDER_COUNT) {
        pr_err("update_provider_metrics: Invalid state or provider\n");
        return;
    }

    /* Update metrics with proper locking */
    spin_lock_irqsave(&metrics_lock, flags);

    atomic_inc(&state->metrics[provider].total_requests);

    if (result == 0) {
        /* Successful request */
        atomic_inc(&state->metrics[provider].successful_requests);
        atomic64_add(latency_ms, &state->metrics[provider].total_latency_ms);

        if (latency_ms < state->metrics[provider].min_latency_ms)
            state->metrics[provider].min_latency_ms = latency_ms;

        if (latency_ms > state->metrics[provider].max_latency_ms)
            state->metrics[provider].max_latency_ms = latency_ms;

        if (tokens > 0)
            atomic_add(tokens, &state->metrics[provider].total_tokens);
    } else {
        /* Failed request */
        atomic_inc(&state->metrics[provider].failed_requests);

        if (result == -ETIMEDOUT)
            atomic_inc(&state->metrics[provider].timeouts);
        else if (result == -LLM_ERR_RATE_LIMIT)
            atomic_inc(&state->metrics[provider].rate_limited);
    }

    /* Check if weights should be adjusted */
    if (state->auto_adjust &&
        (atomic_read(&state->metrics[provider].total_requests) % METRICS_ADJUST_INTERVAL == 0)) {
        should_adjust = true;
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    /* Auto-adjust weights if needed */
    if (should_adjust)
        adjust_scheduler_weights(state);
}

/*
 * Handle rate limiting for a provider
 * Safely update provider status with proper locking
 */
void handle_rate_limit(int provider, struct scheduler_state *state, unsigned long reset_ms)
{
    unsigned long flags;
    ktime_t reset_time;

    if (!state || provider < 0 || provider >= PROVIDER_COUNT) {
        pr_err("handle_rate_limit: Invalid state or provider\n");
        return;
    }

    /* Set provider as rate limited with proper locking */
    spin_lock_irqsave(&metrics_lock, flags);

    /* Set provider as rate limited */
    atomic_set(&state->metrics[provider].current_status, 0);

    /* Set reset time */
    reset_time = ktime_add_ms(ktime_get(), reset_ms);
    state->metrics[provider].rate_limit_reset_time = reset_time;

    spin_unlock_irqrestore(&metrics_lock, flags);

    pr_info("Provider %d rate limited, will reset in %lu ms\n", provider, reset_ms);
}

/*
 * Normalize weights to ensure they sum to 100% and meet minimum values
 * Helper function for adjust_scheduler_weights
 */
static void normalize_weights(struct scheduler_state *state)
{
    int i;
    int total = 0;
    int remaining, deficit;
    int adjustable_count = 0;

    if (!state)
        return;

    /* First pass: apply minimum weight and calculate total */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (state->weights[i] < MIN_PROVIDER_WEIGHT)
            state->weights[i] = MIN_PROVIDER_WEIGHT;
        total += state->weights[i];
    }

    /* If total is already correct, nothing to do */
    if (total == WEIGHT_TOTAL_PERCENT)
        return;

    /* Count providers that can be adjusted */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (state->weights[i] > MIN_PROVIDER_WEIGHT)
            adjustable_count++;
    }

    if (adjustable_count == 0) {
        /* All providers at minimum, reset to equal weights */
        for (i = 0; i < PROVIDER_COUNT; i++) {
            state->weights[i] = WEIGHT_TOTAL_PERCENT / PROVIDER_COUNT;
        }

        /* Distribute remainder evenly */
        remaining = WEIGHT_TOTAL_PERCENT % PROVIDER_COUNT;
        for (i = 0; i < remaining; i++) {
            state->weights[i]++;
        }

        return;
    }

    /* Calculate deficit/surplus */
    deficit = WEIGHT_TOTAL_PERCENT - total;

    if (deficit < 0) {
        /* Total too high - reduce weights proportionally */
        int total_reducible = total - (MIN_PROVIDER_WEIGHT * PROVIDER_COUNT);
        for (i = 0; i < PROVIDER_COUNT; i++) {
            if (state->weights[i] > MIN_PROVIDER_WEIGHT) {
                int reducible = state->weights[i] - MIN_PROVIDER_WEIGHT;
                int reduction = (reducible * -deficit) / total_reducible;
                state->weights[i] -= reduction;
            }
        }
    } else if (deficit > 0) {
        /* Total too low - increase weights proportionally */
        for (i = 0; i < PROVIDER_COUNT; i++) {
            if (state->weights[i] > MIN_PROVIDER_WEIGHT) {
                state->weights[i] += deficit / adjustable_count;
                deficit -= deficit / adjustable_count;
                adjustable_count--;

                if (adjustable_count == 0 && deficit > 0) {
                    state->weights[i] += deficit;
                    break;
                }
            }
        }
    }

    /* Final check to ensure we're at exactly 100% */
    total = 0;
    for (i = 0; i < PROVIDER_COUNT; i++) {
        total += state->weights[i];
    }

    if (total != WEIGHT_TOTAL_PERCENT) {
        /* Adjust the last weight to make total exactly 100% */
        state->weights[PROVIDER_COUNT - 1] += (WEIGHT_TOTAL_PERCENT - total);
    }
}

/*
 * Adjust scheduler weights based on performance metrics
 * Dynamically tunes the system based on provider performance
 */
void adjust_scheduler_weights(struct scheduler_state *state)
{
    int success_rates[PROVIDER_COUNT];
    int total_success_rate = 0;
    int i;
    unsigned long flags_metrics, flags_weights;

    if (!state) {
        pr_err("adjust_scheduler_weights: Invalid state pointer\n");
        return;
    }

    /* Calculate success rates with metrics lock */
    spin_lock_irqsave(&metrics_lock, flags_metrics);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        int total = atomic_read(&state->metrics[i].total_requests);
        int successful = atomic_read(&state->metrics[i].successful_requests);

        if (total > 0) {
            /* Base success rate is percentage of successful requests */
            success_rates[i] = (successful * 100) / total;

            /* Penalize for timeouts and rate limits */
            int timeouts = atomic_read(&state->metrics[i].timeouts);
            int rate_limits = atomic_read(&state->metrics[i].rate_limited);

            if (timeouts > 0 || rate_limits > 0) {
                int penalty = ((timeouts + rate_limits) * RATE_LIMIT_PENALTY) / total;
                success_rates[i] = max(success_rates[i] - penalty, MIN_PROVIDER_WEIGHT);
            }

            total_success_rate += success_rates[i];
        } else {
            /* No data yet, assign default weight */
            success_rates[i] = WEIGHT_TOTAL_PERCENT / PROVIDER_COUNT;
            total_success_rate += success_rates[i];
        }
    }

    spin_unlock_irqrestore(&metrics_lock, flags_metrics);

    /* Acquire weights lock for updating weights */
    spin_lock_irqsave(&weights_lock, flags_weights);

    /* If no requests processed yet, use default weights */
    if (total_success_rate == 0) {
        for (i = 0; i < PROVIDER_COUNT; i++) {
            state->weights[i] = WEIGHT_TOTAL_PERCENT / PROVIDER_COUNT;
        }

        /* Ensure weights are properly normalized */
        normalize_weights(state);
    } else {
        /* Adjust weights based on success rates */
        for (i = 0; i < PROVIDER_COUNT; i++) {
            state->weights[i] = (success_rates[i] * WEIGHT_TOTAL_PERCENT) /
                                max(total_success_rate, 1);
        }

        /* Ensure weights are properly normalized */
        normalize_weights(state);
    }

    pr_debug("Adjusted weights: OpenAI=%d%%, Anthropic=%d%%, Gemini=%d%%\n",
             state->weights[PROVIDER_OPENAI],
             state->weights[PROVIDER_ANTHROPIC],
             state->weights[PROVIDER_GOOGLE_GEMINI]);

    spin_unlock_irqrestore(&weights_lock, flags_weights);
}

/*
 * Reset all metrics
 * Safely clears all metrics with proper locking
 */
void scheduler_reset_metrics(struct scheduler_state *state)
{
    int i;
    unsigned long flags;

    if (!state) {
        pr_err("scheduler_reset_metrics: Invalid state pointer\n");
        return;
    }

    spin_lock_irqsave(&metrics_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        atomic_set(&state->metrics[i].total_requests, 0);
        atomic_set(&state->metrics[i].successful_requests, 0);
        atomic_set(&state->metrics[i].failed_requests, 0);
        atomic_set(&state->metrics[i].timeouts, 0);
        atomic_set(&state->metrics[i].rate_limited, 0);
        atomic64_set(&state->metrics[i].total_latency_ms, 0);
        state->metrics[i].min_latency_ms = ULONG_MAX;
        state->metrics[i].max_latency_ms = 0;
        atomic_set(&state->metrics[i].total_tokens, 0);
        atomic_set(&state->metrics[i].current_status, 1); /* Reset to available */
        state->metrics[i].rate_limit_reset_time = 0;
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    pr_info("LLM scheduler metrics reset\n");
}

/*
 * Initialize the FIFO queue
 * Properly sets up a new queue or resets an existing one
 */
void fifo_init(struct fifo_queue *fifo)
{
    unsigned long flags;

    if (!fifo) {
        pr_err("fifo_init: Invalid fifo pointer\n");
        return;
    }

    spin_lock_irqsave(&fifo->lock, flags);

    memset(fifo->providers, 0, sizeof(fifo->providers));
    fifo->head = 0;
    fifo->tail = 0;
    fifo->count = 0;

    spin_unlock_irqrestore(&fifo->lock, flags);
}

/*
 * Add a provider to the FIFO queue
 * Returns 0 on success, -ENOSPC if queue is full
 */
int fifo_add_provider(struct fifo_queue *fifo, int provider)
{
    unsigned long flags;
    int ret = 0;

    if (!fifo) {
        pr_err("fifo_add_provider: Invalid fifo pointer\n");
        return -EINVAL;
    }

    if (provider < 0 || provider >= PROVIDER_COUNT) {
        pr_err("fifo_add_provider: Invalid provider %d\n", provider);
        return -EINVAL;
    }

    spin_lock_irqsave(&fifo->lock, flags);

    if (fifo->count >= MAX_FIFO_QUEUE_SIZE) {
        ret = -ENOSPC;
    } else {
        fifo->providers[fifo->tail] = provider;
        fifo->tail = (fifo->tail + 1) % MAX_FIFO_QUEUE_SIZE;
        fifo->count++;
    }

    spin_unlock_irqrestore(&fifo->lock, flags);

    return ret;
}

/*
 * Clean up FIFO queue
 * Safely resets the queue with proper locking
 */
void fifo_cleanup(struct fifo_queue *fifo)
{
    unsigned long flags;

    if (!fifo) {
        pr_err("fifo_cleanup: Invalid fifo pointer\n");
        return;
    }

    spin_lock_irqsave(&fifo->lock, flags);

    memset(fifo->providers, 0, sizeof(fifo->providers));
    fifo->head = 0;
    fifo->tail = 0;
    fifo->count = 0;

    spin_unlock_irqrestore(&fifo->lock, flags);

    pr_debug("FIFO queue cleaned up\n");
}

/* Module exports */
EXPORT_SYMBOL(scheduler_init);
EXPORT_SYMBOL(select_provider);
EXPORT_SYMBOL(update_provider_metrics);
EXPORT_SYMBOL(handle_rate_limit);
EXPORT_SYMBOL(adjust_scheduler_weights);
EXPORT_SYMBOL(scheduler_reset_metrics);
EXPORT_SYMBOL(set_scheduler_state);
EXPORT_SYMBOL(get_scheduler_state);
EXPORT_SYMBOL(fifo_init);
EXPORT_SYMBOL(fifo_add_provider);
EXPORT_SYMBOL(fifo_cleanup);
EXPORT_SYMBOL(get_default_model);
EXPORT_SYMBOL(is_model_supported);
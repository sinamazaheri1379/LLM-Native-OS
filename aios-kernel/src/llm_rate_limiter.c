//
// Created by sina-mazaheri on 12/17/24.
//

#include "../include/llm_rate_limiter.h"
int rate_limiter_init(struct rate_limiter *limiter)
{
    atomic_set(&limiter->request_count, 0);
    limiter->window_start = jiffies;
    limiter->max_requests = 60;
    limiter->window_ms = 60000;
    spin_lock_init(&limiter->lock);
    return 0;
}

bool rate_limiter_allow(struct rate_limiter *limiter)
{
    unsigned long now = jiffies;
    bool allowed = false;

    spin_lock(&limiter->lock);

    if (time_after(now, limiter->window_start + msecs_to_jiffies(limiter->window_ms))) {
        limiter->window_start = now;
        atomic_set(&limiter->request_count, 0);
    }

    if (atomic_read(&limiter->request_count) < limiter->max_requests) {
        atomic_inc(&limiter->request_count);
        allowed = true;
    }

    spin_unlock(&limiter->lock);
    return allowed;
}
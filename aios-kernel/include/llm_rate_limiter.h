//
// Created by sina-mazaheri on 12/17/24.
//

#ifndef LLM_RATE_LIMITER_H
#define LLM_RATE_LIMITER_H

#include <linux/atomic.h>
#include <linux/time.h>

struct rate_limiter {
    atomic_t request_count;
    unsigned long window_start;
    int max_requests;
    int window_ms;
    spinlock_t lock;
};

int rate_limiter_init(struct rate_limiter *limiter);
bool rate_limiter_allow(struct rate_limiter *limiter);
void rate_limiter_cleanup(struct rate_limiter *limiter);

#endif /* LLM_RATE_LIMITER_H */

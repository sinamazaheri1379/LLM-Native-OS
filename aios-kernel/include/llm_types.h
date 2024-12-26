//
// Created by sina-mazaheri on 12/17/24.
//

#ifndef LLM_TYPES_H
#define LLM_TYPES_H

#include <linux/types.h>

/* Error codes */
#define LLM_SUCCESS          0
#define LLM_ERR_NETWORK    -1
#define LLM_ERR_AUTH       -2
#define LLM_ERR_PARAM      -3
#define LLM_ERR_MEMORY     -4
#define LLM_ERR_TIMEOUT    -5

/* Common structures */
struct llm_stats {
    atomic_t requests_sent;
    atomic_t requests_failed;
    atomic_t responses_received;
    atomic_t rate_limited;
};

#endif /* LLM_TYPES_H */

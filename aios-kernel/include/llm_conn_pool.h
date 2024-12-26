//
// Created by sina-mazaheri on 12/17/24.
//

#ifndef LLM_CONN_POOL_H
#define LLM_CONN_POOL_H

#include <linux/net.h>
#include <linux/mutex.h>

#define MAX_POOL_CONNECTIONS 4

struct conn_pool {
    struct socket *connections[MAX_POOL_CONNECTIONS];
    bool in_use[MAX_POOL_CONNECTIONS];
    struct mutex lock;
    atomic_t active_count;
};

int conn_pool_init(struct conn_pool *pool);
struct socket *conn_pool_get(struct conn_pool *pool);
void conn_pool_return(struct conn_pool *pool, struct socket *sock);
void conn_pool_cleanup(struct conn_pool *pool);

#endif /* LLM_CONN_POOL_H */

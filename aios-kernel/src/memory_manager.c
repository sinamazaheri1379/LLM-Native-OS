//
// Created by sina-mazaheri on 12/17/24.
//
#include "memory_manager.h"

static struct kmem_cache *llm_msg_cache;
static struct kmem_cache *llm_conn_cache;

/* Initialize memory tracker */
static inline void llm_mem_init(void) {
    atomic_set(&mem_tracker.alloc_count, 0);
    atomic_set(&mem_tracker.free_count, 0);
    atomic64_set(&mem_tracker.total_bytes, 0);
    spin_lock_init(&mem_tracker.track_lock);
    INIT_LIST_HEAD(&mem_tracker.alloc_list);
}

/* Secure allocation with tracking */
static inline void *llm_malloc(size_t size, const char *func, int line) {
    void *ptr;
    struct llm_mem_block *block;
    unsigned long flags;

    if (size == 0)
        return NULL;

    /* Allocate memory block tracker */
    block = kmalloc(sizeof(*block), GFP_KERNEL);
    if (!block)
        return NULL;

    /* Allocate requested memory */
    ptr = kzalloc(size, GFP_KERNEL);
    if (!ptr) {
        kfree(block);
        return NULL;
    }

    /* Initialize tracking info */
    block->ptr = ptr;
    block->size = size;
    block->func = func;
    block->line = line;

    /* Update statistics */
    spin_lock_irqsave(&mem_tracker.track_lock, flags);
    atomic_inc(&mem_tracker.alloc_count);
    atomic64_add(size, &mem_tracker.total_bytes);
    list_add(&block->list, &mem_tracker.alloc_list);
    spin_unlock_irqrestore(&mem_tracker.track_lock, flags);

    return ptr;
}

/* Secure free with tracking */
static inline void llm_free(void *ptr) {
    struct llm_mem_block *block, *tmp;
    unsigned long flags;

    if (!ptr)
        return;

    spin_lock_irqsave(&mem_tracker.track_lock, flags);
    list_for_each_entry_safe(block, tmp, &mem_tracker.alloc_list, list) {
        if (block->ptr == ptr) {
            /* Update statistics */
            atomic_inc(&mem_tracker.free_count);
            atomic64_sub(block->size, &mem_tracker.total_bytes);

            /* Remove from tracking list */
            list_del(&block->list);

            /* Secure cleanup */
            memzero_explicit(ptr, block->size);
            kfree(ptr);
            kfree(block);
            break;
        }
    }
    spin_unlock_irqrestore(&mem_tracker.track_lock, flags);
}

/* Memory leak detection */
static void llm_check_leaks(void) {
    struct llm_mem_block *block;
    unsigned long flags;
    int leak_count = 0;

    spin_lock_irqsave(&mem_tracker.track_lock, flags);

    list_for_each_entry(block, &mem_tracker.alloc_list, list) {
        pr_err("LLM: Memory leak detected: %zu bytes allocated in %s:%d\n",
               block->size, block->func, block->line);
        leak_count++;
    }

    if (leak_count > 0) {
        pr_err("LLM: Total memory leaks: %d\n", leak_count);
        pr_err("LLM: Allocations: %d, Frees: %d, Total bytes: %lld\n",
               atomic_read(&mem_tracker.alloc_count),
               atomic_read(&mem_tracker.free_count),
               atomic64_read(&mem_tracker.total_bytes));
    }

    spin_unlock_irqrestore(&mem_tracker.track_lock, flags);
}


static int __init llm_cache_init(void) {
    llm_msg_cache = kmem_cache_create("llm_message",
                                     sizeof(struct llm_message),
                                     0, SLAB_HWCACHE_ALIGN | SLAB_PANIC,
                                     NULL);

    llm_conn_cache = kmem_cache_create("llm_connection",
                                      sizeof(struct llm_connection),
                                      0, SLAB_HWCACHE_ALIGN | SLAB_PANIC,
                                      NULL);
    return 0;
}

static void llm_cache_exit(void) {
    kmem_cache_destroy(llm_msg_cache);
    kmem_cache_destroy(llm_conn_cache);
}

/* Secure allocation with poisoning */
static inline void *llm_malloc(size_t size, gfp_t flags) {
    void *ptr = kmalloc(size, flags);
    if (ptr && !(flags & __GFP_ZERO))
        memset(ptr, 0xAA, size);
    return ptr;
}

/* Safe message allocation */
struct llm_message *llm_message_alloc(const char *role, const char *content) {
    struct llm_message *msg;
    size_t content_len;

    if (!role || !content)
        return NULL;

    msg = kmem_cache_alloc(llm_msg_cache, GFP_KERNEL);
    if (!msg)
        return NULL;

    content_len = strlen(content) + 1;
    if (content_len > MAX_PROMPT_LENGTH) {
        kmem_cache_free(llm_msg_cache, msg);
        return NULL;
    }

    msg->content = llm_malloc(content_len, GFP_KERNEL);
    if (!msg->content) {
        kmem_cache_free(llm_msg_cache, msg);
        return NULL;
    }

    strscpy(msg->role, role, MAX_ROLE_LENGTH);
    strscpy(msg->content, content, content_len);
    msg->content_length = content_len - 1;
    INIT_LIST_HEAD(&msg->list);

    return msg;
}
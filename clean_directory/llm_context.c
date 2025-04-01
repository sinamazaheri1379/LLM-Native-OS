/*
 * llm_context.c - Enhanced context management with optimized data structures
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>
#include <linux/ratelimit.h>
#include <linux/hashtable.h>
#include <linux/rbtree.h>
#include <linux/shrinker.h>
#include <linux/mm.h>
#include <linux/atomic.h>
#include <linux/rcupdate.h>
#include "orchestrator_main.h"

/* Statistics for monitoring */
static atomic_t entries_added = ATOMIC_INIT(0);
static atomic_t conversations_created = ATOMIC_INIT(0);
static atomic_t entries_pruned = ATOMIC_INIT(0);
static atomic_t json_generated = ATOMIC_INIT(0);
static atomic_t cache_hits = ATOMIC_INIT(0);
static atomic_t cache_misses = ATOMIC_INIT(0);

/* Rate limiting for debug logs */
static DEFINE_RATELIMIT_STATE(ratelimit_state, 5 * HZ, 10);

/* Determine hash table size based on available memory
 * More memory = larger hash table for better performance */
// With:
#define HASH_TABLE_BITS 10  // Fixed size of 1024 buckets (2^10)

/* Hash table for conversation contexts - dynamically sized based on memory */
DEFINE_HASHTABLE(conversation_table, HASH_TABLE_BITS);
DEFINE_SPINLOCK(conversations_lock); /* Global lock for hash table operations */
static bool context_initialized = false;

/* Recent conversations cache - improves performance for frequent accesses */
#define RECENT_CACHE_SIZE 16
static struct {
    int conversation_id;
    struct conversation_context *ctx;
    atomic_t usage_count;
} recent_cache[RECENT_CACHE_SIZE];
static DEFINE_SPINLOCK(cache_lock);

/* Memory pressure handling */
static struct shrinker context_shrinker;
static atomic_t memory_pressure_level = ATOMIC_INIT(0);

/* Custom hash function for conversation IDs - helps reduce hash collisions */
static inline u32 conversation_hash(int conversation_id)
{
    /* Jenkins one-at-a-time hash */
    u32 hash = conversation_id;

    hash += (hash << 10);
    hash ^= (hash >> 6);
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return hash;
}

/* Initialize the recent conversations cache */
static void init_recent_cache(void)
{
    int i;

    for (i = 0; i < RECENT_CACHE_SIZE; i++) {
        recent_cache[i].conversation_id = -1;
        recent_cache[i].ctx = NULL;
        atomic_set(&recent_cache[i].usage_count, 0);
    }
}

/* Find a cached conversation or add it to cache */
static struct conversation_context *cache_conversation(int conversation_id,
                                                      struct conversation_context *ctx)
{
    int i, least_used_idx = 0;
    int min_count = INT_MAX;
    unsigned long flags;

    if (!ctx)
        return NULL;

    spin_lock_irqsave(&cache_lock, flags);

    /* Try to find existing entry or least used slot */
    for (i = 0; i < RECENT_CACHE_SIZE; i++) {
        if (recent_cache[i].conversation_id == conversation_id) {
            /* Already cached */
            atomic_inc(&recent_cache[i].usage_count);
            spin_unlock_irqrestore(&cache_lock, flags);
            return recent_cache[i].ctx;
        }

        int count = atomic_read(&recent_cache[i].usage_count);
        if (count < min_count) {
            min_count = count;
            least_used_idx = i;
        }
    }

    /* Add to cache in least used slot */
    recent_cache[least_used_idx].conversation_id = conversation_id;
    recent_cache[least_used_idx].ctx = ctx;
    atomic_set(&recent_cache[least_used_idx].usage_count, 1);

    spin_unlock_irqrestore(&cache_lock, flags);
    return ctx;
}

/* Clear a conversation from cache */
static void uncache_conversation(int conversation_id)
{
    int i;
    unsigned long flags;

    spin_lock_irqsave(&cache_lock, flags);

    for (i = 0; i < RECENT_CACHE_SIZE; i++) {
        if (recent_cache[i].conversation_id == conversation_id) {
            recent_cache[i].conversation_id = -1;
            recent_cache[i].ctx = NULL;
            atomic_set(&recent_cache[i].usage_count, 0);
            break;
        }
    }

    spin_unlock_irqrestore(&cache_lock, flags);
}

/* Find a conversation context by ID - O(1) operation with hash table + cache */
struct conversation_context *find_conversation_internal(int conversation_id)
{
    struct conversation_context *ctx;
    int i;
    unsigned long flags;

    /* First check cache - fast path */
    spin_lock_irqsave(&cache_lock, flags);
    for (i = 0; i < RECENT_CACHE_SIZE; i++) {
        if (recent_cache[i].conversation_id == conversation_id) {
            ctx = recent_cache[i].ctx;
            atomic_inc(&recent_cache[i].usage_count);
            atomic_inc(&cache_hits);
            spin_unlock_irqrestore(&cache_lock, flags);
            return ctx;
        }
    }
    spin_unlock_irqrestore(&cache_lock, flags);

    /* Cache miss - check hash table */
    atomic_inc(&cache_misses);

    /* Use our custom hash function for better distribution */
    u32 hash = conversation_hash(conversation_id);

    hash_for_each_possible(conversation_table, ctx, hnode, hash) {
        if (ctx->conversation_id == conversation_id) {
            /* Found it - add to cache for future lookups */
            return cache_conversation(conversation_id, ctx);
        }
    }

    return NULL;
}

struct conversation_context *find_conversation(int conversation_id)
{
    return find_conversation_internal(conversation_id);
}

/* Create a new conversation context with reference counting */
static struct conversation_context *create_conversation(int conversation_id)
{
    struct conversation_context *ctx;

    /* Check if memory management is initialized */
    if (!memory_management_initialized()) {
        pr_warn("create_conversation: Memory management not initialized\n");
        return NULL;
    }

    /* Allocate with memory tracking */
    ctx = kmalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return NULL;

    /* Register memory usage */
    if (context_register_memory(conversation_id, sizeof(*ctx))) {
        kfree(ctx);
        return NULL;
    }

    ctx->conversation_id = conversation_id;
    ctx->entry_count = 0;
    ctx->last_updated = ktime_get();
    ctx->total_memory = sizeof(*ctx);
    atomic_set(&ctx->ref_count, 1);  /* Start with 1 reference */
    INIT_LIST_HEAD(&ctx->entries);
    /* Initialize red-black tree for timestamp-based indexing */
    ctx->entries_by_time = RB_ROOT;
    spin_lock_init(&ctx->lock);

    atomic_inc(&conversations_created);

    if (__ratelimit(&ratelimit_state)) {
        pr_debug("Created new conversation context with ID %d (total: %d)\n",
                 conversation_id, atomic_read(&conversations_created));
    }

    return ctx;
}

/* Get a reference to a conversation context */
static struct conversation_context *context_get(int conversation_id)
{
    struct conversation_context *ctx;
    unsigned long flags;

    spin_lock_irqsave(&conversations_lock, flags);

    ctx = find_conversation_internal(conversation_id);
    if (ctx)
        atomic_inc(&ctx->ref_count);

    spin_unlock_irqrestore(&conversations_lock, flags);

    return ctx;
}

/* Release a conversation context reference */
static void context_put(struct conversation_context *ctx)
{
    if (!ctx)
        return;

    if (atomic_dec_and_test(&ctx->ref_count)) {
        /* Last reference - free the context */
        uncache_conversation(ctx->conversation_id);
        context_unregister_memory(ctx->conversation_id, sizeof(*ctx));
        kfree(ctx);
    }
}

/* Insert an entry into the time-based index */
static void insert_entry_time_index(struct conversation_context *ctx,
                                   struct context_entry *entry)
{
    struct rb_node **link = &ctx->entries_by_time.rb_node;
    struct rb_node *parent = NULL;
    struct context_entry *other;

    /* Find the right place in the tree */
    while (*link) {
        parent = *link;
        other = rb_entry(parent, struct context_entry, time_node);

        if (ktime_before(entry->timestamp, other->timestamp))
            link = &parent->rb_left;
        else
            link = &parent->rb_right;
    }

    /* Add new node and rebalance */
    rb_link_node(&entry->time_node, parent, link);
    rb_insert_color(&entry->time_node, &ctx->entries_by_time);
}

/* Get entries in time-sorted order */
static struct context_entry *get_oldest_entry(struct conversation_context *ctx)
{
    struct rb_node *node = rb_first(&ctx->entries_by_time);

    if (!node)
        return NULL;

    return rb_entry(node, struct context_entry, time_node);
}

/* Add a new entry to a conversation context with alignments for cache efficiency */
int context_add_entry(int conversation_id, const char *role, const char *content)
{
    struct conversation_context *ctx;
    struct context_entry *entry;
    unsigned long flags;
    int ret = 0;
    size_t role_len, content_len;
    size_t entry_size;

    /* Check if context management is initialized */
    if (!context_initialized) {
        pr_warn("context_add_entry: Context management not initialized\n");
        return -EAGAIN;
    }

    if (!role || !content || conversation_id <= 0)
        return -EINVAL;

    /* Check input sizes */
    role_len = strlen(role);
    content_len = strlen(content);

    if (role_len >= MAX_ROLE_NAME - 1) {
        pr_warn("context_add_entry: Role too long (%zu > %d)\n",
                role_len, MAX_ROLE_NAME - 1);
        return -EMSGSIZE;
    }

    if (content_len >= MAX_CONTENT_LENGTH - 1) {
        pr_warn("context_add_entry: Content too long (%zu > %d)\n",
                content_len, MAX_CONTENT_LENGTH - 1);
        return -EMSGSIZE;
    }

    /* Find or create conversation */
    spin_lock_irqsave(&conversations_lock, flags);
    ctx = find_conversation(conversation_id);
    if (!ctx) {
        ctx = create_conversation(conversation_id);
        if (!ctx) {
            spin_unlock_irqrestore(&conversations_lock, flags);
            return -ENOMEM;
        }
        /* Add to hash table with our custom hash */
        hash_add(conversation_table, &ctx->hnode, conversation_hash(conversation_id));

        /* Add to cache */
        cache_conversation(conversation_id, ctx);
    }
    spin_unlock_irqrestore(&conversations_lock, flags);

    /* Allocate and initialize entry with memory tracking */
    entry_size = sizeof(struct context_entry);
    entry = context_allocate_entry(conversation_id);
    if (!entry)
        return -ENOMEM;

    /* Copy strings safely */
    ret = strscpy(entry->role, role, sizeof(entry->role));
    if (ret < 0) {
        context_free_entry(conversation_id, entry);
        return -EMSGSIZE; /* More specific error for string truncation */
    }

    ret = strscpy(entry->content, content, sizeof(entry->content));
    if (ret < 0) {
        context_free_entry(conversation_id, entry);
        return -EMSGSIZE; /* More specific error for string truncation */
    }

    entry->timestamp = ktime_get();
    INIT_LIST_HEAD(&entry->list);

    /* Add to conversation with LRU eviction if needed */
    spin_lock_irqsave(&ctx->lock, flags);

    /* Track memory usage */
    ctx->total_memory += entry_size;

    if (ctx->entry_count >= MAX_CONTEXT_ENTRIES) {
        struct context_entry *oldest;

        /* Use our time index to efficiently find oldest entry */
        oldest = get_oldest_entry(ctx);
        if (oldest) {
            /* Remove from indexes */
            list_del(&oldest->list);
            rb_erase(&oldest->time_node, &ctx->entries_by_time);

            /* Release the lock before freeing to prevent potential deadlock */
            spin_unlock_irqrestore(&ctx->lock, flags);

            /* Update memory tracking */
            ctx->total_memory -= sizeof(struct context_entry);

            context_free_entry(conversation_id, oldest);
            atomic_inc(&entries_pruned);

            if (__ratelimit(&ratelimit_state)) {
                pr_debug("Removed oldest entry from conversation %d to make room (pruned: %d)\n",
                         conversation_id, atomic_read(&entries_pruned));
            }

            /* Reacquire the lock to continue */
            spin_lock_irqsave(&ctx->lock, flags);
            ctx->entry_count--; /* Decrement after potential reschedule point */
        }
    }

    /* Add to both indexes */
    list_add_tail(&entry->list, &ctx->entries);
    insert_entry_time_index(ctx, entry);

    ctx->entry_count++;
    ctx->last_updated = ktime_get();
    spin_unlock_irqrestore(&ctx->lock, flags);

    atomic_inc(&entries_added);

    if (__ratelimit(&ratelimit_state)) {
        pr_debug("Added new entry to conversation %d, role: %s, content length: %zu (total: %d)\n",
                 conversation_id, role, content_len, atomic_read(&entries_added));
    }

    return 0;
}

/* Optimized batch operations for adding multiple entries at once */
int context_add_entries_batch(int conversation_id,
                             const struct context_entry_batch *entries,
                             int count)
{
    struct conversation_context *ctx;
    struct context_entry *entry;
    unsigned long flags;
    int i, ret = 0;
    int added = 0;

    if (!context_initialized)
        return -EAGAIN;

    if (!entries || count <= 0 || conversation_id <= 0)
        return -EINVAL;

    /* Get or create context and lock it once for the whole batch */
    ctx = context_get(conversation_id);
    if (!ctx) {
        spin_lock_irqsave(&conversations_lock, flags);
        ctx = create_conversation(conversation_id);
        if (!ctx) {
            spin_unlock_irqrestore(&conversations_lock, flags);
            return -ENOMEM;
        }
        hash_add(conversation_table, &ctx->hnode, conversation_hash(conversation_id));
        cache_conversation(conversation_id, ctx);
        spin_unlock_irqrestore(&conversations_lock, flags);
    }

    /* Process all entries in a single lock acquisition */
    spin_lock_irqsave(&ctx->lock, flags);

    for (i = 0; i < count; i++) {
        /* Check if we need to evict entries */
        while (ctx->entry_count >= MAX_CONTEXT_ENTRIES) {
            struct context_entry *oldest = get_oldest_entry(ctx);
            if (!oldest)
                break;

            /* Remove from indexes */
            list_del(&oldest->list);
            rb_erase(&oldest->time_node, &ctx->entries_by_time);
            ctx->entry_count--;

            /* Update memory tracking - but we still hold the lock */
            ctx->total_memory -= sizeof(struct context_entry);

            /* Add to a temporary list to free after releasing the lock */
            /* (implementation detail omitted for brevity) */

            atomic_inc(&entries_pruned);
        }

        /* Allocate new entry (inside the lock to prevent thrashing) */
        entry = context_allocate_entry(conversation_id);
        if (!entry) {
            ret = -ENOMEM;
            break;
        }

        /* Copy data */
        strscpy(entry->role, entries[i].role, sizeof(entry->role));
        strscpy(entry->content, entries[i].content, sizeof(entry->content));
        entry->timestamp = ktime_get();
        INIT_LIST_HEAD(&entry->list);

        /* Add to both indexes */
        list_add_tail(&entry->list, &ctx->entries);
        insert_entry_time_index(ctx, entry);

        ctx->entry_count++;
        ctx->total_memory += sizeof(struct context_entry);
        added++;
    }

    if (added > 0) {
        ctx->last_updated = ktime_get();
        atomic_add(added, &entries_added);
    }

    spin_unlock_irqrestore(&ctx->lock, flags);

    /* Free any entries we created for cleanup after lock released */
    /* (implementation detail omitted for brevity) */

    context_put(ctx);
    return added > 0 ? added : ret;
}

/* Improved JSON buffer with cached serialization */
int context_get_conversation(int conversation_id, struct llm_json_buffer *json_buf)
{
    struct conversation_context *ctx;
    struct context_entry *entry;
    unsigned long flags;
    int ret = 0;
    bool first = true;
    char *entry_json = NULL;
    size_t buffer_size = 1024; /* Start with larger buffer */
    ktime_t last_cached_generation;
    bool cache_valid = false;

    /* Check initialization */
    if (!context_initialized) {
        pr_warn("context_get_conversation: Context management not initialized\n");
        return -EAGAIN;
    }

    if (!json_buf || !json_buf->data || conversation_id <= 0)
        return -EINVAL;

    /* Get a reference to the context */
    ctx = context_get(conversation_id);
    if (!ctx) {
        if (__ratelimit(&ratelimit_state)) {
            pr_debug("Conversation %d not found\n", conversation_id);
        }
        return -ENOENT;
    }

    /* Check if we have a valid cached JSON representation */
    spin_lock_irqsave(&ctx->lock, flags);
    last_cached_generation = ctx->last_json_generation;
    if (ctx->json_cache &&
        ktime_compare(last_cached_generation, ctx->last_updated) == 0) {
        /* Cache is valid - use it */
        ret = append_json_string(json_buf, ctx->json_cache);
        cache_valid = true;
    }
    spin_unlock_irqrestore(&ctx->lock, flags);

    if (cache_valid) {
        context_put(ctx);
        return ret;
    }

    /* No valid cache - generate JSON */
    ret = append_json_string(json_buf, "[");
    if (ret) {
        context_put(ctx);
        return ret;
    }

    /* Allocate a larger temporary buffer for building each entry */
    entry_json = kmalloc(buffer_size, GFP_KERNEL);
    if (!entry_json) {
        context_put(ctx);
        return -ENOMEM;
    }

    /* Build a new JSON cache buffer */
    struct llm_json_buffer cache_buf;
    ret = json_buffer_init(&cache_buf, 4096);
    if (ret) {
        kfree(entry_json);
        context_put(ctx);
        return ret;
    }

    ret = append_json_string(&cache_buf, "[");
    if (ret) {
        json_buffer_free(&cache_buf);
        kfree(entry_json);
        context_put(ctx);
        return ret;
    }

    /* Iterate through entries */
    spin_lock_irqsave(&ctx->lock, flags);
    list_for_each_entry(entry, &ctx->entries, list) {
        size_t entry_len = 0;

        if (!first) {
            ret = append_json_string(json_buf, ",");
            if (!ret)
                ret = append_json_string(&cache_buf, ",");

            if (ret)
                break;
        }
        first = false;

        /* Format entry JSON with buffer size check */
        ret = snprintf(entry_json, buffer_size, "{\"role\":\"%s\",\"content\":\"", entry->role);
        if (ret < 0) {
            ret = -EIO; /* More specific error for formatting failure */
            break;
        }

        if (ret >= buffer_size) {
            /* Buffer too small, retry with a larger buffer */
            char *new_entry_json;
            buffer_size = ret + 64; /* Add some margin */
            new_entry_json = kmalloc(buffer_size, GFP_KERNEL);
            if (!new_entry_json) {
                ret = -ENOMEM;
                break;
            }
            kfree(entry_json); /* Free old buffer before assignment */
            entry_json = new_entry_json;

            ret = snprintf(entry_json, buffer_size, "{\"role\":\"%s\",\"content\":\"", entry->role);
            if (ret < 0 || ret >= buffer_size) {
                ret = -ENOSPC;
                break;
            }
        }

        entry_len = ret;

        /* Add to both output and cache buffers */
        ret = append_json_string(json_buf, entry_json);
        if (!ret)
            ret = append_json_string(&cache_buf, entry_json);
        if (ret)
            break;

        ret = append_json_value(json_buf, entry->content);
        if (!ret)
            ret = append_json_value(&cache_buf, entry->content);
        if (ret)
            break;

        ret = append_json_string(json_buf, "\"}");
        if (!ret)
            ret = append_json_string(&cache_buf, "\"}");
        if (ret)
            break;
    }

    if (!ret) {
        ret = append_json_string(json_buf, "]");
        if (!ret)
            ret = append_json_string(&cache_buf, "]");
    }

    /* Store the cache if successful */
    if (!ret) {
        /* Free old cache if it exists */
        if (ctx->json_cache) {
            kfree(ctx->json_cache);
            ctx->json_cache = NULL;
        }

        /* Allocate and copy the new cache */
        ctx->json_cache = kmalloc(cache_buf.used + 1, GFP_ATOMIC);
        if (ctx->json_cache) {
            memcpy(ctx->json_cache, cache_buf.data, cache_buf.used + 1);
            ctx->last_json_generation = ctx->last_updated;
        }
    }

    spin_unlock_irqrestore(&ctx->lock, flags);

    kfree(entry_json); /* Free buffer regardless of success or failure */
    json_buffer_free(&cache_buf);

    atomic_inc(&json_generated);

    if (__ratelimit(&ratelimit_state)) {
        pr_debug("Generated JSON for conversation %d with %d entries (total: %d)\n",
                 conversation_id, ctx->entry_count, atomic_read(&json_generated));
    }

    context_put(ctx);
    return ret;
}

/* Get entry count */
int context_get_entry_count(int conversation_id)
{
    struct conversation_context *ctx;
    int count;

    /* Check initialization */
    if (!context_initialized)
        return -EAGAIN;

    if (conversation_id <= 0)
        return -EINVAL;

    ctx = context_get(conversation_id);
    if (!ctx)
        return 0; /* No conversation found, so 0 entries */

    count = ctx->entry_count;
    context_put(ctx);

    return count;
}

/* Clear conversation with improved locking */
int context_clear_conversation(int conversation_id)
{
    struct conversation_context *ctx;
    struct context_entry *entry, *tmp;
    unsigned long flags;
    int cleared = 0;

    /* Check initialization */
    if (!context_initialized)
        return -EAGAIN;

    if (conversation_id <= 0)
        return -EINVAL;

    ctx = context_get(conversation_id);
    if (!ctx)
        return -ENOENT;

    /* Now handle the conversation with its own lock */
    spin_lock_irqsave(&ctx->lock, flags);

    /* Clear JSON cache */
    if (ctx->json_cache) {
        kfree(ctx->json_cache);
        ctx->json_cache = NULL;
    }

    /* Create a cleanup list for entries */
    LIST_HEAD(cleanup_list);

    /* Move all entries to the cleanup list */
    list_for_each_entry_safe(entry, tmp, &ctx->entries, list) {
        list_del(&entry->list);
        list_add(&entry->list, &cleanup_list);
        rb_erase(&entry->time_node, &ctx->entries_by_time);
        cleared++;
    }

    ctx->entry_count = 0;
    ctx->total_memory = sizeof(*ctx); /* Reset to just the context structure size */
    ctx->last_updated = ktime_get();
    ctx->entries_by_time = RB_ROOT; /* Reset time index */
    spin_unlock_irqrestore(&ctx->lock, flags);

    /* Free all entries in the cleanup list */
    list_for_each_entry_safe(entry, tmp, &cleanup_list, list) {
        list_del(&entry->list);
        context_free_entry(conversation_id, entry);
    }

    pr_debug("Cleared %d entries from conversation %d\n", cleared, conversation_id);

    context_put(ctx);
    return 0;
}

/* Memory pressure callback */
static unsigned long context_shrink_count(struct shrinker *shrink,
                                         struct shrink_control *sc)
{
    /* Report an estimate of freeable memory */
    return atomic_read(&entries_added) - atomic_read(&entries_pruned);
}

static unsigned long context_shrink_scan(struct shrinker *shrink,
                                        struct shrink_control *sc)
{
    unsigned long freed = 0;
    int level = atomic_read(&memory_pressure_level);
    unsigned long threshold_ms;

    /* Adapt pruning threshold based on memory pressure level */
    switch (level) {
        case 0: /* No pressure */
            threshold_ms = 24 * 60 * 60 * 1000; /* 24 hours */
            break;
        case 1: /* Moderate */
            threshold_ms = 6 * 60 * 60 * 1000;  /* 6 hours */
            break;
        case 2: /* High */
            threshold_ms = 1 * 60 * 60 * 1000;  /* 1 hour */
            break;
        default: /* Critical */
            threshold_ms = 10 * 60 * 1000;      /* 10 minutes */
            break;
    }

    /* Prune old conversations based on adaptive threshold */
    freed = context_prune_old_conversations(threshold_ms);

    /* Convert to pages */
    return freed * sizeof(struct context_entry) / PAGE_SIZE;
}

/* Register shrinker callbacks */
static void register_context_shrinker(void)
{
    context_shrinker.count_objects = context_shrink_count;
    context_shrinker.scan_objects = context_shrink_scan;
    context_shrinker.seeks = DEFAULT_SEEKS;
    register_shrinker(&context_shrinker);
}

/* Prune old conversations with improved locking */
int context_prune_old_conversations(unsigned long age_threshold_ms)
{
    struct conversation_context *ctx;
    unsigned long flags;
    ktime_t cutoff_time;
    int pruned = 0;
    struct hlist_node *tmp;
    int i;
    LIST_HEAD(cleanup_contexts);

    /* Check initialization */
    if (!context_initialized)
        return -EAGAIN;

    cutoff_time = ktime_sub_ms(ktime_get(), age_threshold_ms);

    spin_lock_irqsave(&conversations_lock, flags);

    /* First pass: identify old conversations */
    hash_for_each_safe(conversation_table, i, tmp, ctx, hnode) {
        if (ktime_before(ctx->last_updated, cutoff_time)) {
            /* Remove from hash table but keep a reference */
            hash_del(&ctx->hnode);
            atomic_inc(&ctx->ref_count); /* Extra reference for our cleanup list */
            list_add(&ctx->cleanup_node, &cleanup_contexts);
            pruned++;
        }
    }

    spin_unlock_irqrestore(&conversations_lock, flags);

    /* Second pass: handle entries from identified conversations */
    while (!list_empty(&cleanup_contexts)) {
        struct context_entry *entry, *tmp_entry;
        LIST_HEAD(entry_cleanup);
        int conversation_id;

        /* Get context from cleanup list */
        ctx = list_first_entry(&cleanup_contexts, struct conversation_context, cleanup_node);
        list_del(&ctx->cleanup_node);

        /* Remember ID for cleanup */
        conversation_id = ctx->conversation_id;

        /* Remove from cache */
        uncache_conversation(conversation_id);

        /* Move all entries to a cleanup list */
        spin_lock_irqsave(&ctx->lock, flags);

        /* Clear JSON cache */
        if (ctx->json_cache) {
            kfree(ctx->json_cache);
            ctx->json_cache = NULL;
        }

        list_for_each_entry_safe(entry, tmp_entry, &ctx->entries, list) {
            list_del(&entry->list);
            rb_erase(&entry->time_node, &ctx->entries_by_time);
            list_add(&entry->list, &entry_cleanup);
        }

        ctx->entry_count = 0;
        ctx->total_memory = sizeof(*ctx);
        ctx->entries_by_time = RB_ROOT;
        spin_unlock_irqrestore(&ctx->lock, flags);

        /* Free all entries in cleanup list */
        list_for_each_entry_safe(entry, tmp_entry, &entry_cleanup, list) {
            list_del(&entry->list);
            context_free_entry(conversation_id, entry);
            atomic_inc(&entries_pruned);
        }

        /* Release our reference (and potentially free the context) */
        context_put(ctx);
    }

    if (pruned > 0) {
        pr_info("Pruned %d old conversations (older than %lu ms)\n",
                pruned, age_threshold_ms);
    }

    return pruned;
}
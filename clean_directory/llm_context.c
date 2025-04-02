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
#include <linux/limits.h>
#include <linux/jiffies.h>
#include "orchestrator_main.h"

/* Constants for safe buffer management */
#define MAX_JSON_CACHE_SIZE (1024 * 1024)  /* 1MB max cache size */
#define INITIAL_ENTRY_BUFFER_SIZE 1024     /* Initial buffer for formatting entry JSON */
#define MAX_ENTRY_BUFFER_SIZE (32 * 1024)  /* Maximum buffer size for entry formatting */
#define CONTEXT_VERSION_INITIAL 1          /* Initial version for cache validity */
#define MAX_ENTRIES_TO_PROCESS 32          /* Max entries to process in a batch */

/* Statistics for monitoring */
static atomic_t entries_added = ATOMIC_INIT(0);
static atomic_t conversations_created = ATOMIC_INIT(0);
static atomic_t entries_pruned = ATOMIC_INIT(0);
static atomic_t json_generated = ATOMIC_INIT(0);
static atomic_t cache_hits = ATOMIC_INIT(0);
static atomic_t cache_misses = ATOMIC_INIT(0);
static atomic_t alloc_failures = ATOMIC_INIT(0);

/* Rate limiting for debug logs */
static DEFINE_RATELIMIT_STATE(ratelimit_state, 5 * HZ, 10);

/* Determine hash table size based on available memory */
#define HASH_TABLE_BITS 10  /* Fixed size of 1024 buckets (2^10) */

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

/* Function declarations for static helpers */
static void init_recent_cache(void);
static struct conversation_context *cache_conversation(int conversation_id,
                                                      struct conversation_context *ctx);
static void uncache_conversation(int conversation_id);
static struct conversation_context *create_conversation(int conversation_id);
static struct conversation_context *context_get(int conversation_id);
static void context_put(struct conversation_context *ctx);
static void insert_entry_time_index(struct conversation_context *ctx,
                                   struct context_entry *entry);
static struct context_entry *get_oldest_entry(struct conversation_context *ctx);
static unsigned long context_shrink_count(struct shrinker *shrink,
                                         struct shrink_control *sc);
static unsigned long context_shrink_scan(struct shrinker *shrink,
                                        struct shrink_control *sc);
static void register_context_shrinker(void);
static int format_json_entry(struct llm_json_buffer *json_buf,
                            struct llm_json_buffer *cache_buf,
                            const struct context_entry *entry,
                            char *entry_json, size_t buffer_size);
static void conversation_free_callback(struct rcu_head *head);

/* Structure to hold entry data for JSON generation */
struct entry_data {
    char role[MAX_ROLE_NAME];
    char content[MAX_CONTENT_LENGTH];
};

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

    /* Use RCU lock for better concurrency - readers can proceed without locks */
    rcu_read_lock();

    hash_for_each_possible_rcu(conversation_table, ctx, hnode, hash) {
        if (ctx->conversation_id == conversation_id) {
            /* Found it - take a reference before releasing RCU lock */
            if (atomic_inc_not_zero(&ctx->ref_count)) {
                rcu_read_unlock();
                /* Add to cache for future lookups */
                return cache_conversation(conversation_id, ctx);
            }
        }
    }

    rcu_read_unlock();
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
    if (!ctx) {
        atomic_inc(&alloc_failures);
        pr_err("create_conversation: Failed to allocate context for ID %d\n", conversation_id);
        return NULL;
    }

    /* Register memory usage */
    if (context_register_memory(conversation_id, sizeof(*ctx))) {
        pr_warn("create_conversation: Failed to register memory for ID %d\n", conversation_id);
        kfree(ctx);
        return NULL;
    }

    /* Initialize structure fields */
    ctx->conversation_id = conversation_id;
    ctx->entry_count = 0;
    ctx->last_updated = ktime_get();
    ctx->last_json_generation = ktime_set(0, 0); /* Initialize to zero */
    ctx->cache_version = CONTEXT_VERSION_INITIAL;
    ctx->total_memory = sizeof(*ctx);
    ctx->json_cache = NULL;
    atomic_set(&ctx->ref_count, 1);  /* Start with 1 reference */
    INIT_LIST_HEAD(&ctx->entries);
    INIT_LIST_HEAD(&ctx->cleanup_node);
    ctx->entries_by_time = RB_ROOT; /* Initialize red-black tree root */
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

    ctx = find_conversation_internal(conversation_id);
    /* find_conversation_internal already took a reference if it found ctx */
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

        /* Free JSON cache if it exists */
        if (ctx->json_cache) {
            kfree(ctx->json_cache);
            ctx->json_cache = NULL;
        }

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
    struct context_entry *to_free = NULL;

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

    /* Allocate and initialize entry BEFORE trying to get the context to avoid blocking */
    entry_size = sizeof(struct context_entry);
    entry = context_allocate_entry(conversation_id);
    if (!entry) {
        atomic_inc(&alloc_failures);
        return -ENOMEM;
    }

    /* Copy strings safely while we're not holding any locks */
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

    /* Find or create conversation */
    ctx = context_get(conversation_id);
    if (!ctx) {
        unsigned long flags;
        /* Need to create a new conversation */
        ctx = create_conversation(conversation_id);
        if (!ctx) {
            context_free_entry(conversation_id, entry);
            return -ENOMEM;
        }

        /* Add to hash table with RCU protection */
        spin_lock_irqsave(&conversations_lock, flags);
        hash_add_rcu(conversation_table, &ctx->hnode, conversation_hash(conversation_id));
        spin_unlock_irqrestore(&conversations_lock, flags);

        /* Add to cache */
        cache_conversation(conversation_id, ctx);
    }

    /* Add to conversation with LRU eviction if needed */
    spin_lock_irqsave(&ctx->lock, flags);

    /* Track memory usage */
    ctx->total_memory += entry_size;

    if (ctx->entry_count >= MAX_CONTEXT_ENTRIES) {
        /* Use our time index to efficiently find oldest entry */
        to_free = get_oldest_entry(ctx);
        if (to_free) {
            /* Remove from indexes while holding the lock */
            list_del(&to_free->list);
            rb_erase(&to_free->time_node, &ctx->entries_by_time);

            /* Update memory tracking inside the lock */
            ctx->total_memory -= entry_size;
            ctx->entry_count--; /* Decrement count inside the lock */
        }
    }

    /* Add to both indexes */
    list_add_tail(&entry->list, &ctx->entries);
    insert_entry_time_index(ctx, entry);

    /* Invalidate JSON cache */
    if (ctx->json_cache) {
        kfree(ctx->json_cache);
        ctx->json_cache = NULL;
    }
    /* Increment cache version to invalidate any external cache references */
    ctx->cache_version++;

    ctx->entry_count++;
    ctx->last_updated = ktime_get();
    spin_unlock_irqrestore(&ctx->lock, flags);

    /* Now free the evicted entry outside the lock if needed */
    if (to_free) {
        context_free_entry(conversation_id, to_free);
        atomic_inc(&entries_pruned);

        if (__ratelimit(&ratelimit_state)) {
            pr_debug("Removed oldest entry from conversation %d to make room (pruned: %d)\n",
                     conversation_id, atomic_read(&entries_pruned));
        }
    }

    atomic_inc(&entries_added);

    if (__ratelimit(&ratelimit_state)) {
        pr_debug("Added new entry to conversation %d, role: %s, content length: %zu (total: %d)\n",
                 conversation_id, role, content_len, atomic_read(&entries_added));
    }

    context_put(ctx);
    return 0;
}

/* Optimized batch operations for adding multiple entries at once */
int context_add_entries_batch(int conversation_id,
                             const struct context_entry_batch *entries,
                             int count)
{
    struct conversation_context *ctx;
    struct context_entry **entry_ptrs = NULL; /* Array of pointers */
    unsigned long flags;
    int i, ret = 0;
    int added = 0;
    LIST_HEAD(cleanup_entries);
    int valid_entries = 0;

    if (!context_initialized)
        return -EAGAIN;

    if (!entries || count <= 0 || conversation_id <= 0)
        return -EINVAL;

    /* Limit batch size for safety */
    if (count > MAX_ENTRIES_TO_PROCESS)
        count = MAX_ENTRIES_TO_PROCESS;

    /* Allocate array of entry pointers */
    entry_ptrs = kmalloc_array(count, sizeof(struct context_entry *), GFP_KERNEL);
    if (!entry_ptrs) {
        atomic_inc(&alloc_failures);
        return -ENOMEM;
    }

    /* Pre-allocate all entries before taking any locks */
    for (i = 0; i < count; i++) {
        struct context_entry *entry = context_allocate_entry(conversation_id);
        if (!entry) {
            /* Remember how many we allocated successfully */
            break;
        }

        /* Copy data while we don't hold any locks */
        ret = strscpy(entry->role, entries[i].role, sizeof(entry->role));
        if (ret < 0) {
            context_free_entry(conversation_id, entry);
            continue;
        }

        ret = strscpy(entry->content, entries[i].content, sizeof(entry->content));
        if (ret < 0) {
            context_free_entry(conversation_id, entry);
            continue;
        }

        entry->timestamp = ktime_get();
        INIT_LIST_HEAD(&entry->list);

        /* Save valid entry pointer */
        entry_ptrs[valid_entries++] = entry;
    }

    /* If we couldn't allocate any entries, return error */
    if (valid_entries == 0) {
        kfree(entry_ptrs);
        return -ENOMEM;
    }

    /* Get or create context and lock it once for the whole batch */
    ctx = context_get(conversation_id);
    if (!ctx) {
        unsigned long flags;
        /* Create new conversation */
        ctx = create_conversation(conversation_id);
        if (!ctx) {
            /* Free all allocated entries */
            for (i = 0; i < valid_entries; i++) {
                context_free_entry(conversation_id, entry_ptrs[i]);
            }
            kfree(entry_ptrs);
            return -ENOMEM;
        }

        /* Add to hash table with RCU protection */
        spin_lock_irqsave(&conversations_lock, flags);
        hash_add_rcu(conversation_table, &ctx->hnode, conversation_hash(conversation_id));
        spin_unlock_irqrestore(&conversations_lock, flags);

        cache_conversation(conversation_id, ctx);
    }

    /* Process all entries in a single lock acquisition */
    spin_lock_irqsave(&ctx->lock, flags);

    /* Invalidate JSON cache */
    if (ctx->json_cache) {
        kfree(ctx->json_cache);
        ctx->json_cache = NULL;
    }
    /* Increment version to invalidate any external references */
    ctx->cache_version++;

    /* First handle evictions if needed */
    while (ctx->entry_count >= MAX_CONTEXT_ENTRIES &&
           ctx->entry_count + valid_entries > MAX_CONTEXT_ENTRIES) {

        struct context_entry *oldest = get_oldest_entry(ctx);
        if (!oldest)
            break;

        /* Remove from indexes */
        list_del(&oldest->list);
        rb_erase(&oldest->time_node, &ctx->entries_by_time);
        ctx->entry_count--;

        /* Update memory tracking inside the lock */
        ctx->total_memory -= sizeof(struct context_entry);

        /* Add to cleanup list for freeing after releasing the lock */
        list_add(&oldest->list, &cleanup_entries);
    }

    /* Now add all our pre-allocated entries */
    for (i = 0; i < valid_entries; i++) {
        struct context_entry *entry = entry_ptrs[i];

        /* Add to both indexes */
        list_add_tail(&entry->list, &ctx->entries);
        insert_entry_time_index(ctx, entry);

        ctx->entry_count++;
        ctx->total_memory += sizeof(struct context_entry);
        added++;
    }

    if (added > 0) {
        ctx->last_updated = ktime_get();
    }

    spin_unlock_irqrestore(&ctx->lock, flags);

    /* Free any entries in the cleanup list */
    struct context_entry *tmp, *entry_to_free;
    list_for_each_entry_safe(entry_to_free, tmp, &cleanup_entries, list) {
        list_del(&entry_to_free->list);
        context_free_entry(conversation_id, entry_to_free);
        atomic_inc(&entries_pruned);
    }

    if (added > 0) {
        atomic_add(added, &entries_added);
    }

    /* Free the entry pointer array */
    kfree(entry_ptrs);

    context_put(ctx);
    return added > 0 ? added : ret;
}

/* Helper function to safely format an entry as JSON */
static int format_json_entry(struct llm_json_buffer *json_buf,
                            struct llm_json_buffer *cache_buf,
                            const struct context_entry *entry,
                            char *entry_json, size_t buffer_size)
{
    int ret;

    /* Format entry JSON with buffer size check */
    ret = snprintf(entry_json, buffer_size, "{\"role\":\"%s\",\"content\":\"", entry->role);
    if (ret < 0) {
        return -EIO; /* More specific error for formatting failure */
    }

    if (ret >= buffer_size) {
        return -ENOSPC; /* Buffer too small */
    }

    /* Add to both output and cache buffers */
    ret = append_json_string(json_buf, entry_json);
    if (ret)
        return ret;

    ret = append_json_string(cache_buf, entry_json);
    if (ret)
        return ret;

    ret = append_json_value(json_buf, entry->content);
    if (ret)
        return ret;

    ret = append_json_value(cache_buf, entry->content);
    if (ret)
        return ret;

    ret = append_json_string(json_buf, "\"}");
    if (ret)
        return ret;

    ret = append_json_string(cache_buf, "\"}");

    return ret;
}

/* Improved JSON buffer with cached serialization - using two-phase approach to reduce lock time */
int context_get_conversation(int conversation_id, struct llm_json_buffer *json_buf)
{
    struct conversation_context *ctx;
    unsigned long flags;
    int ret = 0;
    u32 cache_version;
    bool cache_valid = false;
    struct llm_json_buffer cache_buf;
    char *entry_json = NULL;
    bool cache_buf_initialized = false;
    struct entry_data *entries_data = NULL;
    int entry_count = 0;
    int i;

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
    cache_version = ctx->cache_version;
    if (ctx->json_cache &&
        ktime_compare(ctx->last_json_generation, ctx->last_updated) == 0) {
        /* Cache is valid - use it and add version protection */
        ret = append_json_string(json_buf, ctx->json_cache);
        cache_valid = true;
    }
    spin_unlock_irqrestore(&ctx->lock, flags);

    if (cache_valid) {
        context_put(ctx);
        return ret;
    }

    /* No valid cache - start JSON generation */
    ret = append_json_string(json_buf, "[");
    if (ret) {
        context_put(ctx);
        return ret;
    }

    /* Initialize cache buffer outside lock */
    ret = json_buffer_init(&cache_buf, 4096);
    if (ret) {
        context_put(ctx);
        return ret;
    }
    cache_buf_initialized = true;

    ret = append_json_string(&cache_buf, "[");
    if (ret) {
        goto cleanup;
    }

    /* Allocate a temporary buffer for building each entry */
    entry_json = kmalloc(INITIAL_ENTRY_BUFFER_SIZE, GFP_KERNEL);
    if (!entry_json) {
        ret = -ENOMEM;
        atomic_inc(&alloc_failures);
        goto cleanup;
    }

    /* First pass: Get a snapshot of entries while holding the lock */
    spin_lock_irqsave(&ctx->lock, flags);

    /* If version changed, the cache is still valid */
    if (ctx->json_cache && ctx->cache_version == cache_version) {
        /* Someone else generated the cache while we were preparing */
        ret = append_json_string(json_buf, ctx->json_cache);
        spin_unlock_irqrestore(&ctx->lock, flags);
        goto cleanup;
    }

    /* Count the entries first */
    entry_count = ctx->entry_count;

    /* Allocate array for entry data - but don't do memory operations under spinlock */
    spin_unlock_irqrestore(&ctx->lock, flags);

    /* Allocate array with reasonable size cap */
    if (entry_count > MAX_ENTRIES_TO_PROCESS)
        entry_count = MAX_ENTRIES_TO_PROCESS;

    entries_data = kmalloc_array(entry_count, sizeof(struct entry_data), GFP_KERNEL);
    if (!entries_data) {
        ret = -ENOMEM;
        atomic_inc(&alloc_failures);
        goto cleanup;
    }

    /* Second pass: Get entry data with minimal lock time */
    spin_lock_irqsave(&ctx->lock, flags);

    /* Check cache again */
    if (ctx->json_cache && ctx->cache_version == cache_version) {
        /* Someone else generated the cache while we were preparing */
        ret = append_json_string(json_buf, ctx->json_cache);
        spin_unlock_irqrestore(&ctx->lock, flags);
        goto cleanup;
    }

    /* Copy entry data to our array */
    {
        struct context_entry *entry;
        int idx = 0;

        list_for_each_entry(entry, &ctx->entries, list) {
            if (idx >= entry_count)
                break;

            /* Copy only the necessary data */
            strscpy(entries_data[idx].role, entry->role, MAX_ROLE_NAME);
            strscpy(entries_data[idx].content, entry->content, MAX_CONTENT_LENGTH);
            idx++;
        }

        /* Update actual count */
        entry_count = idx;
    }

    /* We're done with the lock */
    spin_unlock_irqrestore(&ctx->lock, flags);

    /* Third pass: Generate JSON from our snapshot - outside the lock */
    for (i = 0; i < entry_count; i++) {
        /* Add comma between entries */
        if (i > 0) {
            ret = append_json_string(json_buf, ",");
            if (!ret)
                ret = append_json_string(&cache_buf, ",");
            if (ret)
                goto cleanup;
        }

        /* Format the JSON without holding any lock */
        ret = snprintf(entry_json, INITIAL_ENTRY_BUFFER_SIZE,
                     "{\"role\":\"%s\",\"content\":\"", entries_data[i].role);
        if (ret < 0) {
            ret = -EIO; /* More specific error for formatting failure */
            goto cleanup;
        }

        if (ret >= INITIAL_ENTRY_BUFFER_SIZE) {
            /* Buffer too small, retry with a larger buffer */
            char *new_entry_json;
            size_t buffer_size = ret + 64; /* Add some margin */

            /* Enforce maximum buffer size */
            if (buffer_size > MAX_ENTRY_BUFFER_SIZE) {
                buffer_size = MAX_ENTRY_BUFFER_SIZE;
            }

            new_entry_json = kmalloc(buffer_size, GFP_KERNEL);
            if (!new_entry_json) {
                ret = -ENOMEM;
                goto cleanup;
            }
            kfree(entry_json); /* Free old buffer before assignment */
            entry_json = new_entry_json;

            ret = snprintf(entry_json, buffer_size,
                         "{\"role\":\"%s\",\"content\":\"", entries_data[i].role);
            if (ret < 0 || ret >= buffer_size) {
                ret = -ENOSPC;
                goto cleanup;
            }
        }

        /* Add to both output and cache buffers */
        ret = append_json_string(json_buf, entry_json);
        if (ret)
            goto cleanup;

        ret = append_json_string(&cache_buf, entry_json);
        if (ret)
            goto cleanup;

        ret = append_json_value(json_buf, entries_data[i].content);
        if (ret)
            goto cleanup;

        ret = append_json_value(&cache_buf, entries_data[i].content);
        if (ret)
            goto cleanup;

        ret = append_json_string(json_buf, "\"}");
        if (ret)
            goto cleanup;

        ret = append_json_string(&cache_buf, "\"}");
        if (ret)
            goto cleanup;
    }

    /* Finish JSON arrays */
    ret = append_json_string(json_buf, "]");
    if (ret)
        goto cleanup;

    ret = append_json_string(&cache_buf, "]");
    if (ret)
        goto cleanup;

    /* Store the cache if successful - with minimal lock time */
    if (cache_buf.used <= MAX_JSON_CACHE_SIZE) {
        char *new_cache = kmalloc(cache_buf.used + 1, GFP_KERNEL);
        if (new_cache) {
            memcpy(new_cache, cache_buf.data, cache_buf.used + 1);

            /* Update cache with lock */
            spin_lock_irqsave(&ctx->lock, flags);

            /* Free old cache */
            if (ctx->json_cache) {
                kfree(ctx->json_cache);
            }

            /* Set new cache */
            ctx->json_cache = new_cache;
            ctx->last_json_generation = ctx->last_updated;

            spin_unlock_irqrestore(&ctx->lock, flags);
        }
    }

    atomic_inc(&json_generated);

    if (__ratelimit(&ratelimit_state)) {
        pr_debug("Generated JSON for conversation %d with %d entries (total: %d)\n",
                 conversation_id, entry_count, atomic_read(&json_generated));
    }

cleanup:
    if (entries_data)
        kfree(entries_data);
    if (entry_json)
        kfree(entry_json);
    if (cache_buf_initialized)
        json_buffer_free(&cache_buf);
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
    LIST_HEAD(cleanup_list);

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
    /* Increment version to invalidate any external references */
    ctx->cache_version++;

    /* Move all entries to the cleanup list */
    list_for_each_entry_safe(entry, tmp, &ctx->entries, list) {
        list_del(&entry->list);
        rb_erase(&entry->time_node, &ctx->entries_by_time);
        list_add(&entry->list, &cleanup_list);
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

/* Use RCU safe version for conversation deletion */
static void conversation_free_callback(struct rcu_head *head)
{
    struct conversation_context *ctx = container_of(head, struct conversation_context, rcu);
    context_put(ctx); /* Release the reference we took earlier */
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

    /* Use RCU-safe traversal to identify old conversations */
    rcu_read_lock();

    hash_for_each_rcu(conversation_table, i, ctx, hnode) {
        if (ktime_before(ctx->last_updated, cutoff_time)) {
            /* Found an old conversation - take a reference */
            if (atomic_inc_not_zero(&ctx->ref_count)) {
                list_add(&ctx->cleanup_node, &cleanup_contexts);
                pruned++;
            }
        }
    }

    rcu_read_unlock();

    /* Now process each identified conversation */
    list_for_each_entry(ctx, &cleanup_contexts, cleanup_node) {
        /* Remove from hash table with proper synchronization */
        spin_lock_irqsave(&conversations_lock, flags);
        hash_del_rcu(&ctx->hnode);
        spin_unlock_irqrestore(&conversations_lock, flags);

        /* Remove from cache */
        uncache_conversation(ctx->conversation_id);

        /* Clear all entries */
        context_clear_conversation(ctx->conversation_id);

        /* Use RCU to safely free the conversation after grace period */
        call_rcu(&ctx->rcu, conversation_free_callback);
    }

    if (pruned > 0) {
        pr_info("Pruned %d old conversations (older than %lu ms)\n",
                pruned, age_threshold_ms);
    }

    return pruned;
}

/* Get cache statistics */
void context_get_cache_stats(int *hits, int *misses, int *hit_ratio_percent)
{
    int h, m;

    h = atomic_read(&cache_hits);
    m = atomic_read(&cache_misses);

    if (hits)
        *hits = h;

    if (misses)
        *misses = m;

    if (hit_ratio_percent) {
        int total = h + m;
        *hit_ratio_percent = total > 0 ? (h * 100) / total : 0;
    }
}

/* Set memory pressure level */
void context_set_memory_pressure(int level)
{
    if (level >= 0 && level <= 3) {
        atomic_set(&memory_pressure_level, level);

        /* If high pressure, trigger immediate pruning */
        if (level >= 2) {
            context_prune_old_conversations(level == 3 ?
                                           (10 * 60 * 1000) :  /* 10 minutes */
                                           (60 * 60 * 1000));  /* 1 hour */
        }
    }
}

/* Get allocation statistics */
void context_get_alloc_stats(int *failures, int *entries_added_count, int *entries_pruned_count)
{
    if (failures)
        *failures = atomic_read(&alloc_failures);

    if (entries_added_count)
        *entries_added_count = atomic_read(&entries_added);

    if (entries_pruned_count)
        *entries_pruned_count = atomic_read(&entries_pruned);
}

/* Initialize context management subsystem */
int context_management_init(void)
{
    /* Initialize hash table and locks already defined statically */

    /* Initialize recent cache */
    init_recent_cache();

    /* Register memory pressure handler */
    register_context_shrinker();

    /* Mark as initialized */
    context_initialized = true;

    pr_info("Context management subsystem initialized: hash_bits=%d, cache_size=%d\n",
            HASH_TABLE_BITS, RECENT_CACHE_SIZE);

    return 0;
}

/* Clean up all conversations */
void context_cleanup_all(void)
{
    struct conversation_context *ctx;
    struct hlist_node *tmp;
    int i;
    unsigned long flags;
    LIST_HEAD(cleanup_list);

    if (!context_initialized)
        return;

    /* First pass - get all conversations */
    spin_lock_irqsave(&conversations_lock, flags);

    hash_for_each_safe(conversation_table, i, tmp, ctx, hnode) {
        hash_del_rcu(&ctx->hnode);
        atomic_inc(&ctx->ref_count); /* Prevent premature cleanup */
        list_add(&ctx->cleanup_node, &cleanup_list);
    }

    spin_unlock_irqrestore(&conversations_lock, flags);

    /* Second pass - clean up each conversation */
    while (!list_empty(&cleanup_list)) {
        ctx = list_first_entry(&cleanup_list, struct conversation_context, cleanup_node);
        list_del(&ctx->cleanup_node);

        /* Clear from cache */
        uncache_conversation(ctx->conversation_id);

        /* Clear conversation content */
        context_clear_conversation(ctx->conversation_id);

        /* Release our reference */
        context_put(ctx);
        context_put(ctx); /* Second put for the reference we took above */
    }

    /* Wait for RCU grace period to ensure all callbacks complete */
    synchronize_rcu();

    /* Unregister shrinker */
    unregister_shrinker(&context_shrinker);

    context_initialized = false;

    pr_info("Context management cleaned up\n");
}

/* Clean up context management subsystem */
void context_management_cleanup(void)
{
    /* Clean up all conversations */
    context_cleanup_all();
}

/* Get context statistics */
void context_get_stats(int *total_conversations, int *total_entries,
                      int *entries_added_count, int *entries_pruned_count)
{
    if (total_conversations)
        *total_conversations = atomic_read(&conversations_created);

    if (total_entries) {
        *total_entries = atomic_read(&entries_added) - atomic_read(&entries_pruned);
        if (*total_entries < 0)
            *total_entries = 0;
    }

    if (entries_added_count)
        *entries_added_count = atomic_read(&entries_added);

    if (entries_pruned_count)
        *entries_pruned_count = atomic_read(&entries_pruned);
}

/* Export symbols for functions that need to be accessible from other modules */
EXPORT_SYMBOL(context_add_entry);
EXPORT_SYMBOL(context_get_conversation);
EXPORT_SYMBOL(context_get_entry_count);
EXPORT_SYMBOL(context_clear_conversation);
EXPORT_SYMBOL(context_prune_old_conversations);
EXPORT_SYMBOL(context_cleanup_all);
EXPORT_SYMBOL(find_conversation);
EXPORT_SYMBOL(find_conversation_internal);
EXPORT_SYMBOL(context_management_init);
EXPORT_SYMBOL(context_management_cleanup);
EXPORT_SYMBOL(context_get_stats);
EXPORT_SYMBOL(context_add_entries_batch);
EXPORT_SYMBOL(context_set_memory_pressure);
EXPORT_SYMBOL(context_get_cache_stats);
EXPORT_SYMBOL(context_get_alloc_stats);
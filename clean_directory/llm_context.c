#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>
#include <linux/ratelimit.h>
#include "orchestrator_main.h"

/*
 * Locking hierarchy:
 * 1. conversations_lock (global)
 * 2. ctx->lock (per-conversation)
 * Always acquire locks in this order to prevent deadlocks.
 */

/* Statistics for monitoring */
static atomic_t entries_added = ATOMIC_INIT(0);
static atomic_t conversations_created = ATOMIC_INIT(0);
static atomic_t entries_pruned = ATOMIC_INIT(0);
static atomic_t json_generated = ATOMIC_INIT(0);

/* Rate limiting for debug logs */
static DEFINE_RATELIMIT_STATE(ratelimit_state, 5 * HZ, 10);

/* Global list of conversation contexts */
static LIST_HEAD(conversation_list);
static DEFINE_SPINLOCK(conversations_lock);
static bool context_initialized = false;

/* Find a conversation context by ID */
static struct conversation_context *find_conversation(int conversation_id)
{
    struct conversation_context *ctx;

    list_for_each_entry(ctx, &conversation_list, list) {
        if (ctx->conversation_id == conversation_id)
            return ctx;
    }

    return NULL;
}

/* Create a new conversation context */
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
    INIT_LIST_HEAD(&ctx->entries);
    spin_lock_init(&ctx->lock);

    atomic_inc(&conversations_created);

    if (__ratelimit(&ratelimit_state)) {
        pr_debug("Created new conversation context with ID %d (total: %d)\n",
                 conversation_id, atomic_read(&conversations_created));
    }

    return ctx;
}

/* Add a new entry to a conversation context */
int context_add_entry(int conversation_id, const char *role, const char *content)
{
    struct conversation_context *ctx;
    struct context_entry *entry;
    unsigned long flags;
    int ret = 0;
    size_t role_len, content_len;

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
        list_add(&ctx->list, &conversation_list);
    }
    spin_unlock_irqrestore(&conversations_lock, flags);

    /* Allocate and initialize entry */
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
    if (ctx->entry_count >= MAX_CONTEXT_ENTRIES) {
        struct context_entry *oldest;
        oldest = list_first_entry(&ctx->entries, struct context_entry, list);
        list_del(&oldest->list);
        /* Release the lock before freeing to prevent potential deadlock */
        spin_unlock_irqrestore(&ctx->lock, flags);

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

    list_add_tail(&entry->list, &ctx->entries);
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

/* Fix 1: Correct memory leak in context_get_conversation() in llm_context.c */
int context_get_conversation(int conversation_id, struct llm_json_buffer *json_buf)
{
    struct conversation_context *ctx;
    struct context_entry *entry;
    unsigned long flags;
    int ret = 0;
    bool first = true;
    char *entry_json = NULL;
    size_t buffer_size = 1024; /* Start with larger buffer */

    /* Check initialization */
    if (!context_initialized) {
        pr_warn("context_get_conversation: Context management not initialized\n");
        return -EAGAIN;
    }

    if (!json_buf || !json_buf->data || conversation_id <= 0)
        return -EINVAL;

    spin_lock_irqsave(&conversations_lock, flags);
    ctx = find_conversation(conversation_id);
    if (!ctx) {
        spin_unlock_irqrestore(&conversations_lock, flags);
        if (__ratelimit(&ratelimit_state)) {
            pr_debug("Conversation %d not found\n", conversation_id);
        }
        return -ENOENT;
    }
    spin_unlock_irqrestore(&conversations_lock, flags);

    ret = append_json_string(json_buf, "[");
    if (ret)
        return ret;

    /* Allocate a larger temporary buffer for building each entry */
    entry_json = kmalloc(buffer_size, GFP_KERNEL);
    if (!entry_json)
        return -ENOMEM;

    /* Iterate through entries */
    spin_lock_irqsave(&ctx->lock, flags);
    list_for_each_entry(entry, &ctx->entries, list) {
        size_t entry_len = 0;

        if (!first) {
            ret = append_json_string(json_buf, ",");
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

        ret = append_json_string(json_buf, entry_json);
        if (ret)
            break;

        ret = append_json_value(json_buf, entry->content);
        if (ret)
            break;

        ret = append_json_string(json_buf, "\"}");
        if (ret)
            break;
    }
    spin_unlock_irqrestore(&ctx->lock, flags);

    kfree(entry_json); /* Free buffer regardless of success or failure */

    if (!ret)
        ret = append_json_string(json_buf, "]");

    atomic_inc(&json_generated);

    if (__ratelimit(&ratelimit_state)) {
        pr_debug("Generated JSON for conversation %d with %d entries (total: %d)\n",
                 conversation_id, ctx->entry_count, atomic_read(&json_generated));
    }

    return ret;
}


/* Get entry count */
int context_get_entry_count(int conversation_id)
{
    struct conversation_context *ctx;
    unsigned long flags;
    int count;

    /* Check initialization */
    if (!context_initialized)
        return -EAGAIN;

    if (conversation_id <= 0)
        return -EINVAL;

    spin_lock_irqsave(&conversations_lock, flags);
    ctx = find_conversation(conversation_id);
    if (!ctx) {
        spin_unlock_irqrestore(&conversations_lock, flags);
        return 0; /* No conversation found, so 0 entries */
    }
    count = ctx->entry_count;
    spin_unlock_irqrestore(&conversations_lock, flags);

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

    spin_lock_irqsave(&conversations_lock, flags);
    ctx = find_conversation(conversation_id);
    if (!ctx) {
        spin_unlock_irqrestore(&conversations_lock, flags);
        return -ENOENT;
    }

    /*
     * To avoid nested spinlocks, we just get a reference to the conversation
     * and release the global lock before acquiring the per-conversation lock
     */
    spin_unlock_irqrestore(&conversations_lock, flags);

    /* Now handle the conversation with its own lock */
    spin_lock_irqsave(&ctx->lock, flags);
    list_for_each_entry_safe(entry, tmp, &ctx->entries, list) {
        list_del(&entry->list);
        /* Release lock before potential sleep in free */
        spin_unlock_irqrestore(&ctx->lock, flags);

        context_free_entry(conversation_id, entry);
        cleared++;

        /* Reacquire lock to continue iteration */
        spin_lock_irqsave(&ctx->lock, flags);
    }

    ctx->entry_count = 0;
    ctx->last_updated = ktime_get();
    spin_unlock_irqrestore(&ctx->lock, flags);

    pr_debug("Cleared %d entries from conversation %d\n", cleared, conversation_id);

    return 0;
}

/* Prune old conversations with improved locking */
int context_prune_old_conversations(unsigned long age_threshold_ms)
{
    struct conversation_context *ctx, *tmp_ctx;
    unsigned long flags;
    ktime_t cutoff_time;
    int pruned = 0;

    /* Check initialization */
    if (!context_initialized)
        return -EAGAIN;

    cutoff_time = ktime_sub_ms(ktime_get(), age_threshold_ms);

    spin_lock_irqsave(&conversations_lock, flags);

    /* First pass: identify old conversations without modifying the list */
    list_for_each_entry(ctx, &conversation_list, list) {
        if (ktime_before(ctx->last_updated, cutoff_time))
            pruned++;
    }

    if (pruned == 0) {
        /* Early exit if no conversations need pruning */
        spin_unlock_irqrestore(&conversations_lock, flags);
        return 0;
    }

    /* Second pass: remove old conversations */
    pruned = 0;
    list_for_each_entry_safe(ctx, tmp_ctx, &conversation_list, list) {
        if (ktime_before(ctx->last_updated, cutoff_time)) {
            struct context_entry *entry, *tmp_entry;
            int conversation_id = ctx->conversation_id;

            /* Remove from global list first */
            list_del(&ctx->list);

            /* Release global lock before handling entries */
            spin_unlock_irqrestore(&conversations_lock, flags);

            /* Handle entries with conversation lock */
            spin_lock_irqsave(&ctx->lock, flags);
            list_for_each_entry_safe(entry, tmp_entry, &ctx->entries, list) {
                list_del(&entry->list);
                /* Release lock before freeing to avoid nested locks */
                spin_unlock_irqrestore(&ctx->lock, flags);

                context_free_entry(conversation_id, entry);

                /* Reacquire lock to continue */
                spin_lock_irqsave(&ctx->lock, flags);
            }
            spin_unlock_irqrestore(&ctx->lock, flags);

            /* Free conversation after all entries are processed */
            context_unregister_memory(conversation_id, sizeof(*ctx));
            kfree(ctx);
            pruned++;

            /* Reacquire global lock to continue iteration */
            spin_lock_irqsave(&conversations_lock, flags);
        }
    }

    spin_unlock_irqrestore(&conversations_lock, flags);

    if (pruned > 0) {
        pr_info("Pruned %d old conversations (older than %lu ms)\n",
                pruned, age_threshold_ms);
    }

    return pruned;
}

/* Fix 1: Correct lock hierarchy violation in context_cleanup_all() in llm_context.c */
void context_cleanup_all(void)
{
    struct conversation_context *ctx, *tmp_ctx;
    unsigned long flags;
    int count = 0;
    struct list_head temp_list;

    /* Even if not initialized, try to clean up */
    context_initialized = false;

    INIT_LIST_HEAD(&temp_list);

    /* First, unlink all contexts from the global list with proper locking */
    spin_lock_irqsave(&conversations_lock, flags);
    list_for_each_entry(ctx, &conversation_list, list) {
        count++;
    }
    list_splice_init(&conversation_list, &temp_list);
    spin_unlock_irqrestore(&conversations_lock, flags);

    /* Now process each context without holding the global lock */
    list_for_each_entry_safe(ctx, tmp_ctx, &temp_list, list) {
        struct context_entry *entry, *tmp_entry;
        int conversation_id = ctx->conversation_id;
        unsigned long ctx_flags;

        /* Remove from temp list */
        list_del(&ctx->list);

        /* Process entries with conversation lock */
        spin_lock_irqsave(&ctx->lock, ctx_flags);
        list_for_each_entry_safe(entry, tmp_entry, &ctx->entries, list) {
            list_del(&entry->list);
            /* Release lock before freeing */
            spin_unlock_irqrestore(&ctx->lock, ctx_flags);

            context_free_entry(conversation_id, entry);

            /* Reacquire lock to continue */
            spin_lock_irqsave(&ctx->lock, ctx_flags);
        }
        spin_unlock_irqrestore(&ctx->lock, ctx_flags);

        /* Free conversation */
        context_unregister_memory(conversation_id, sizeof(*ctx));
        kfree(ctx);
    }

    pr_info("Cleaned up %d conversations\n", count);
}


/* Get statistics for monitoring */
void context_get_stats(int *total_conversations, int *total_entries,
                       int *entries_added_count, int *entries_pruned_count)
{
    if (total_conversations)
        *total_conversations = atomic_read(&conversations_created);

    if (total_entries)
        *total_entries = atomic_read(&entries_added) - atomic_read(&entries_pruned);

    if (entries_added_count)
        *entries_added_count = atomic_read(&entries_added);

    if (entries_pruned_count)
        *entries_pruned_count = atomic_read(&entries_pruned);
}

/* Show statistics via sysfs */
ssize_t context_stats_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    int total_conversations, total_entries, added, pruned, json_count;

    total_conversations = atomic_read(&conversations_created);
    added = atomic_read(&entries_added);
    pruned = atomic_read(&entries_pruned);
    total_entries = added - pruned;
    json_count = atomic_read(&json_generated);

    return scnprintf(buf, PAGE_SIZE,
                     "Conversation Context Statistics:\n"
                     "  Conversations Created: %d\n"
                     "  Entries Added: %d\n"
                     "  Entries Pruned: %d\n"
                     "  Current Entries: %d\n"
                     "  JSON Generations: %d\n"
                     "  System Initialized: %s\n",
                     total_conversations,
                     added,
                     pruned,
                     total_entries,
                     json_count,
                     context_initialized ? "Yes" : "No");
}

/* Initialize context management system */
int context_management_init(void)
{
    if (context_initialized) {
        pr_warn("context_management_init: Already initialized\n");
        return 0;
    }

    /* Reset statistics */
    atomic_set(&entries_added, 0);
    atomic_set(&conversations_created, 0);
    atomic_set(&entries_pruned, 0);
    atomic_set(&json_generated, 0);

    /* Initialize ratelimit state */
    ratelimit_state.interval = 5 * HZ;  /* 5 seconds */
    ratelimit_state.burst = 10;

    context_initialized = true;
    pr_info("Conversation context management initialized\n");
    return 0;
}

/* Clean up context management system */
void context_management_cleanup(void)
{
    if (!context_initialized) {
        pr_warn("context_management_cleanup: Not initialized\n");
        return;
    }

    /* Mark as uninitialized before cleanup to prevent new entries */
    context_initialized = false;

    /* Clean up all conversations */
    context_cleanup_all();

    pr_info("Conversation context management cleaned up\n");
}

/* Check if context management is initialized */
bool context_management_initialized(void)
{
    return context_initialized;
}

EXPORT_SYMBOL(context_add_entry);
EXPORT_SYMBOL(context_get_conversation);
EXPORT_SYMBOL(context_clear_conversation);
EXPORT_SYMBOL(context_cleanup_all);
EXPORT_SYMBOL(context_get_entry_count);
EXPORT_SYMBOL(context_prune_old_conversations);
EXPORT_SYMBOL(context_get_stats);
EXPORT_SYMBOL(context_management_init);
EXPORT_SYMBOL(context_management_cleanup);
EXPORT_SYMBOL(context_management_initialized);
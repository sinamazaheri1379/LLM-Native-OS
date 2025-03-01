#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>
#include "llm_orchestrator.h"

/* Global list of conversation contexts */
static LIST_HEAD(conversation_list);
static DEFINE_SPINLOCK(conversations_lock);

/*
 * Find a conversation context by ID
 * Must be called with conversations_lock held
 */
static struct conversation_context *find_conversation(int conversation_id)
{
    struct conversation_context *ctx;
    
    list_for_each_entry(ctx, &conversation_list, list) {
        if (ctx->conversation_id == conversation_id)
            return ctx;
    }
    
    return NULL;
}

/*
 * Create a new conversation context
 * Returns pointer on success, NULL on failure
 */
static struct conversation_context *create_conversation(int conversation_id)
{
    struct conversation_context *ctx;
    
    ctx = kmalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return NULL;
        
    ctx->conversation_id = conversation_id;
    ctx->entry_count = 0;
    ctx->last_updated = ktime_get();
    INIT_LIST_HEAD(&ctx->entries);
    spin_lock_init(&ctx->lock);
    
    pr_debug("Created new conversation context with ID %d\n", conversation_id);
    
    return ctx;
}

/*
 * Add a new entry to the conversation context
 * Returns 0 on success, negative error code on failure
 */
int context_add_entry(int conversation_id, const char *role, const char *content)
{
    struct conversation_context *ctx;
    struct context_entry *entry;
    unsigned long flags;
    int ret = 0;

    if (!role || !content || conversation_id <= 0)
        return -EINVAL;

    spin_lock_irqsave(&conversations_lock, flags);

    /* Find or create conversation context */
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

    /* Allocate new entry */
    entry = kmalloc(sizeof(*entry), GFP_KERNEL);
    if (!entry)
        return -ENOMEM;

    /* Copy role and content */
    if (strlcpy(entry->role, role, sizeof(entry->role)) >= sizeof(entry->role) ||
        strlcpy(entry->content, content, sizeof(entry->content)) >= sizeof(entry->content)) {
        kfree(entry);
        return -EINVAL; /* Strings were truncated */
    }

    entry->timestamp = ktime_get();
    INIT_LIST_HEAD(&entry->list);

    /* Add entry to conversation */
    spin_lock_irqsave(&ctx->lock, flags);

    /* If we reached the maximum number of entries, remove the oldest one */
    if (ctx->entry_count >= MAX_CONTEXT_ENTRIES) {
        struct context_entry *oldest;
        oldest = list_first_entry(&ctx->entries, struct context_entry, list);
        list_del(&oldest->list);
        kfree(oldest);
        ctx->entry_count--;
        pr_debug("Removed oldest entry from conversation %d to make room\n", conversation_id);
    }

    list_add_tail(&entry->list, &ctx->entries);
    ctx->entry_count++;
    ctx->last_updated = ktime_get();

    spin_unlock_irqrestore(&ctx->lock, flags);

    pr_debug("Added new entry to conversation %d, role: %s, content length: %zu\n",
             conversation_id, role, strlen(content));

    return ret;
}

/*
 * Generate JSON representation of a conversation context
 * Returns 0 on success, negative error code on failure
 */
int context_get_conversation(int conversation_id, struct llm_json_buffer *json_buf)
{
    struct conversation_context *ctx;
    struct context_entry *entry;
    unsigned long flags, entry_flags;
    int ret = 0;
    bool first = true;

    if (!json_buf || !json_buf->data || conversation_id <= 0)
        return -EINVAL;

    spin_lock_irqsave(&conversations_lock, flags);

    /* Find conversation context */
    ctx = find_conversation(conversation_id);
    if (!ctx) {
        spin_unlock_irqrestore(&conversations_lock, flags);
        pr_debug("Conversation %d not found\n", conversation_id);
        return -ENOENT;
    }

    spin_unlock_irqrestore(&conversations_lock, flags);

    /* Start array */
    ret = append_json_string(json_buf, "[");
    if (ret) {
        return ret;
    }

    spin_lock_irqsave(&ctx->lock, entry_flags);

    /* Add each entry */
    list_for_each_entry(entry, &ctx->entries, list) {
        char entry_json[MAX_PAYLOAD_SIZE];
        size_t entry_len = 0;

        if (!first) {
            ret = append_json_string(json_buf, ",");
            if (ret)
                break;
        }
        first = false;

        /* Pre-format entry JSON in a local buffer */
        ret = snprintf(entry_json, sizeof(entry_json),
                       "{\"role\":\"%s\",\"content\":\"", entry->role);
        if (ret < 0 || ret >= sizeof(entry_json)) {
            ret = -ENOSPC;
            break;
        }
        entry_len = ret;

        /* Ensure space for value and closing braces */
        if (entry_len + 20 >= sizeof(entry_json)) {
            ret = -ENOSPC;
            break;
        }

        spin_unlock_irqrestore(&ctx->lock, entry_flags);

        /* Append the pre-formatted string to the JSON buffer */
        ret = append_json_string(json_buf, entry_json);
        if (ret) {
            spin_lock_irqsave(&ctx->lock, entry_flags);
            break;
        }

        /* Append the escaped content value */
        ret = append_json_value(json_buf, entry->content);
        if (ret) {
            spin_lock_irqsave(&ctx->lock, entry_flags);
            break;
        }

        /* Append closing braces */
        ret = append_json_string(json_buf, "\"}");
        if (ret) {
            spin_lock_irqsave(&ctx->lock, entry_flags);
            break;
        }

        spin_lock_irqsave(&ctx->lock, entry_flags);
    }

    spin_unlock_irqrestore(&ctx->lock, entry_flags);

    /* End array */
    if (!ret) {
        ret = append_json_string(json_buf, "]");
    }

    pr_debug("Generated JSON for conversation %d with %d entries\n",
             conversation_id, ctx->entry_count);

    return ret;
}

/*
 * Get the number of entries in a conversation
 * Returns entry count on success, negative error code on failure
 */
int context_get_entry_count(int conversation_id)
{
    struct conversation_context *ctx;
    unsigned long flags;
    int count;
    
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

/*
 * Clear all entries from a conversation context
 * Returns 0 on success, negative error code on failure
 */
int context_clear_conversation(int conversation_id)
{
    struct conversation_context *ctx;
    struct context_entry *entry, *tmp;
    unsigned long flags;
    
    if (conversation_id <= 0)
        return -EINVAL;
    
    spin_lock_irqsave(&conversations_lock, flags);
    
    /* Find conversation context */
    ctx = find_conversation(conversation_id);
    if (!ctx) {
        spin_unlock_irqrestore(&conversations_lock, flags);
        return -ENOENT;
    }
    
    /* We found the context, now lock it */
    spin_lock(&ctx->lock);
    
    /* Remove all entries */
    list_for_each_entry_safe(entry, tmp, &ctx->entries, list) {
        list_del(&entry->list);
        kfree(entry);
    }
    
    ctx->entry_count = 0;
    ctx->last_updated = ktime_get();
    
    spin_unlock(&ctx->lock);
    spin_unlock_irqrestore(&conversations_lock, flags);
    
    pr_debug("Cleared all entries from conversation %d\n", conversation_id);
    
    return 0;
}

/*
 * Prune old conversations based on age threshold
 * Returns number of conversations pruned
 */
int context_prune_old_conversations(unsigned long age_threshold_ms)
{
    struct conversation_context *ctx, *tmp_ctx;
    unsigned long flags;
    ktime_t cutoff_time;
    int pruned = 0;
    
    cutoff_time = ktime_sub_ms(ktime_get(), age_threshold_ms);
    
    spin_lock_irqsave(&conversations_lock, flags);
    
    /* Find and remove old conversations */
    list_for_each_entry_safe(ctx, tmp_ctx, &conversation_list, list) {
        if (ktime_before(ctx->last_updated, cutoff_time)) {
            struct context_entry *entry, *tmp_entry;
            
            /* Remove all entries */
            spin_lock(&ctx->lock);
            list_for_each_entry_safe(entry, tmp_entry, &ctx->entries, list) {
                list_del(&entry->list);
                kfree(entry);
            }
            spin_unlock(&ctx->lock);
            
            /* Remove conversation */
            list_del(&ctx->list);
            kfree(ctx);
            
            pruned++;
        }
    }
    
    spin_unlock_irqrestore(&conversations_lock, flags);
    
    if (pruned > 0) {
        pr_info("Pruned %d old conversations (older than %lu ms)\n", 
                pruned, age_threshold_ms);
    }
    
    return pruned;
}

/*
 * Clean up all conversation contexts
 * Called when module is unloaded
 */
void context_cleanup_all(void)
{
    struct conversation_context *ctx, *tmp_ctx;
    struct context_entry *entry, *tmp_entry;
    unsigned long flags;
    int count = 0;
    
    spin_lock_irqsave(&conversations_lock, flags);
    
    /* Remove all conversations and entries */
    list_for_each_entry_safe(ctx, tmp_ctx, &conversation_list, list) {
        list_del(&ctx->list);
        
        /* Remove all entries in this conversation */
        list_for_each_entry_safe(entry, tmp_entry, &ctx->entries, list) {
            list_del(&entry->list);
            kfree(entry);
        }
        
        kfree(ctx);
        count++;
    }
    
    spin_unlock_irqrestore(&conversations_lock, flags);
    
    pr_info("Cleaned up %d conversations\n", count);
}

/*
 * Helper function to append a JSON string
 */
int append_json_string(struct llm_json_buffer *buf, const char *str)
{
    size_t len;
    
    if (!buf || !buf->data || !str)
        return -EINVAL;
        
    len = strlen(str);
    if (buf->used + len >= buf->size)
        return -ENOSPC;
        
    memcpy(buf->data + buf->used, str, len);
    buf->used += len;
    buf->data[buf->used] = '\0';
    
    return 0;
}

/*
 * Helper function to append a JSON number
 */
int append_json_number(struct llm_json_buffer *buf, int number)
{
    char value[32];
    int ret;
    
    if (!buf || !buf->data)
        return -EINVAL;
        
    ret = snprintf(value, sizeof(value), "%d", number);
    if (ret < 0 || ret >= sizeof(value))
        return -EINVAL;
        
    return append_json_string(buf, value);
}

/*
 * Helper function to append a JSON float value (stored as integer * 100)
 */
int append_json_float(struct llm_json_buffer *buf, int value_x100)
{
    char value[32];
    int ret;
    
    if (!buf || !buf->data)
        return -EINVAL;
        
    ret = snprintf(value, sizeof(value), "%.2f", (float)value_x100 / 100.0f);
    if (ret < 0 || ret >= sizeof(value))
        return -EINVAL;
        
    return append_json_string(buf, value);
}

/*
 * Helper function to append a JSON boolean
 */
int append_json_boolean(struct llm_json_buffer *buf, bool value)
{
    if (!buf || !buf->data)
        return -EINVAL;
        
    return append_json_string(buf, value ? "true" : "false");
}

/*
 * Helper function to append a JSON value with proper escaping
 */
int append_json_value(struct llm_json_buffer *buf, const char *value)
{
    size_t i, len;

    if (!buf || !buf->data || !value)
        return -EINVAL;

    len = strlen(value);

    for (i = 0; i < len; i++) {
        char c = value[i];

        /* Check if we need to escape this character */
        if (c == '"' || c == '\\' || c == '\b' || c == '\f' ||
            c == '\n' || c == '\r' || c == '\t') {

            /* Make sure we have enough space (2 chars) */
            if (buf->used + 2 >= buf->size)
                return -ENOSPC;

            buf->data[buf->used++] = '\\';

            switch (c) {
                case '"':  buf->data[buf->used++] = '"';  break;
                case '\\': buf->data[buf->used++] = '\\'; break;
                case '\b': buf->data[buf->used++] = 'b';  break;
                case '\f': buf->data[buf->used++] = 'f';  break;
                case '\n': buf->data[buf->used++] = 'n';  break;
                case '\r': buf->data[buf->used++] = 'r';  break;
                case '\t': buf->data[buf->used++] = 't';  break;
            }
        } else if ((unsigned char)c < 32) {
            /* Control characters need special handling (6 chars for \uXXXX) */
            if (buf->used + 6 >= buf->size)
                return -ENOSPC;

            buf->data[buf->used++] = '\\';
            buf->data[buf->used++] = 'u';
            buf->data[buf->used++] = '0';
            buf->data[buf->used++] = '0';

            /* Convert to hex */
            snprintf(buf->data + buf->used, 3, "%02x", (unsigned char)c);
            buf->used += 2;
        } else {
            /* Normal character */
            if (buf->used + 1 >= buf->size)
                return -ENOSPC;

            buf->data[buf->used++] = c;
        }
    }

    buf->data[buf->used] = '\0';
    return 0;
}

/*
 * Extract content from JSON response
 * This is a simple parser to extract content from various API responses
 */
int extract_response_content(const char *json, char *output, size_t output_size)
{
    const char *content_start = NULL;
    const char *content_end = NULL;
    int json_depth = 0;
    bool in_string = false;

    if (!json || !output || output_size == 0)
        return -EINVAL;

    /* Clear output buffer */
    output[0] = '\0';

    /* Try different formats based on provider patterns */

    /* OpenAI format: "content":"..." */
    content_start = strstr(json, "\"content\":\"");
    if (content_start) {
        content_start += 11;  /* Skip "content":" */
        goto process_content;
    }

    /* Anthropic format: might be same as OpenAI */

    /* Gemini format: "text":"..." */
    content_start = strstr(json, "\"text\":\"");
    if (content_start) {
        content_start += 8;  /* Skip "text":" */
        goto process_content;
    }

    /* No recognizable format found */
    return -EINVAL;

    process_content:
    /* Find end of content by properly handling nested structures */
    content_end = content_start;
    while (*content_end) {
        if (*content_end == '\\' && *(content_end + 1)) {
            content_end += 2; /* Skip escaped character */
            continue;
        }

        if (*content_end == '{' && !in_string) json_depth++;
        if (*content_end == '}' && !in_string) json_depth--;

        if (*content_end == '"') {
            /* If we're at the top level and this is the closing quote */
            if (json_depth == 0 && content_end > content_start) {
                break;
            }
            in_string = !in_string;
        }

        content_end++;
    }

    if (*content_end != '"')
        return -EINVAL; /* No closing quote found */

    {
        size_t content_len = content_end - content_start;
        size_t out_idx = 0;
        size_t i;

        /* Make sure we don't overflow output buffer */
        if (content_len >= output_size)
            content_len = output_size - 1;

        /* Copy and unescape the content */
        for (i = 0; i < content_len && out_idx < output_size - 1; i++) {
            if (content_start[i] == '\\' && i + 1 < content_len) {
                i++;
                switch (content_start[i]) {
                    case 'n': output[out_idx++] = '\n'; break;
                    case 'r': output[out_idx++] = '\r'; break;
                    case 't': output[out_idx++] = '\t'; break;
                    case 'b': output[out_idx++] = '\b'; break;
                    case 'f': output[out_idx++] = '\f'; break;
                    case '\\': output[out_idx++] = '\\'; break;
                    case '"': output[out_idx++] = '"'; break;
                    case 'u': /* Unicode escape */
                        /* This would require more complex handling for unicode */
                        if (i + 4 < content_len && out_idx < output_size - 2) {
                            /* Simple handling: just output a placeholder */
                            output[out_idx++] = '?';
                            i += 4; /* Skip the 4 hex digits */
                        } else {
                            /* Not enough characters left */
                            i = content_len; /* Exit loop */
                        }
                        break;
                    default:
                        if (out_idx < output_size - 1) {
                            output[out_idx++] = content_start[i];
                        }
                        break;
                }
            } else {
                if (out_idx < output_size - 1) {
                    output[out_idx++] = content_start[i];
                }
            }
        }

        output[out_idx] = '\0';
        return out_idx;
    }
}

/*
 * Parse token count from JSON response
 * Returns 0 on success, negative error on failure
 */
int parse_token_count(const char *json, int *prompt_tokens,
                      int *completion_tokens, int *total_tokens)
{
    const char *usage_start;
    int ret_prompt = -EINVAL;
    int ret_completion = -EINVAL;
    int ret_total = -EINVAL;

    if (!json || !prompt_tokens || !completion_tokens || !total_tokens)
        return -EINVAL;

    /* Initialize outputs */
    *prompt_tokens = 0;
    *completion_tokens = 0;
    *total_tokens = 0;

    /* Find usage section */
    usage_start = strstr(json, "\"usage\"");
    if (!usage_start) {
        return -EINVAL;
    }

    /* Parse prompt_tokens */
    {
        const char *pt = strstr(usage_start, "\"prompt_tokens\"");
        if (pt) {
            pt = strchr(pt, ':');
            if (pt) {
                ret_prompt = kstrtoint(pt + 1, 10, prompt_tokens);
                if (ret_prompt) {
                    *prompt_tokens = 0;
                }
            }
        }
    }

    /* Parse completion_tokens */
    {
        const char *ct = strstr(usage_start, "\"completion_tokens\"");
        if (ct) {
            ct = strchr(ct, ':');
            if (ct) {
                ret_completion = kstrtoint(ct + 1, 10, completion_tokens);
                if (ret_completion) {
                    *completion_tokens = 0;
                }
            }
        }
    }

    /* Parse total_tokens */
    {
        const char *tt = strstr(usage_start, "\"total_tokens\"");
        if (tt) {
            tt = strchr(tt, ':');
            if (tt) {
                ret_total = kstrtoint(tt + 1, 10, total_tokens);
                if (ret_total) {
                    *total_tokens = 0;
                }
            }
        }
    }

    /* Only return success if we found at least one valid token count */
    if (ret_prompt == 0 || ret_completion == 0 || ret_total == 0) {
        return 0;
    }

    return -EINVAL;
}

/*
 * Initialize a JSON buffer
 */
int json_buffer_init(struct llm_json_buffer *buf, size_t size)
{
    if (!buf)
        return -EINVAL;

    /* Use GFP_ATOMIC for allocations that might happen in interrupt context */
    buf->data = kmalloc(size, GFP_ATOMIC);
    if (!buf->data)
        return -ENOMEM;

    buf->size = size;
    buf->used = 0;
    buf->data[0] = '\0';

    return 0;
}

/*
 * Free a JSON buffer
 */
void json_buffer_free(struct llm_json_buffer *buf)
{
    if (!buf)
        return;
        
    if (buf->data) {
        kfree(buf->data);
        buf->data = NULL;
    }
    
    buf->size = 0;
    buf->used = 0;
}

/* Module exports */
EXPORT_SYMBOL(context_add_entry);
EXPORT_SYMBOL(context_get_conversation);
EXPORT_SYMBOL(context_clear_conversation);
EXPORT_SYMBOL(context_cleanup_all);
EXPORT_SYMBOL(context_get_entry_count);
EXPORT_SYMBOL(context_prune_old_conversations);
EXPORT_SYMBOL(append_json_string);
EXPORT_SYMBOL(append_json_value);
EXPORT_SYMBOL(append_json_number);
EXPORT_SYMBOL(append_json_float);
EXPORT_SYMBOL(append_json_boolean);
EXPORT_SYMBOL(json_buffer_init);
EXPORT_SYMBOL(json_buffer_free);
EXPORT_SYMBOL(extract_response_content);
EXPORT_SYMBOL(parse_token_count);

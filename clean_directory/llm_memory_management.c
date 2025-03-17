#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/atomic.h>
#include <linux/timer.h>
#include <linux/limits.h>
#include <linux/workqueue.h>
#include <linux/moduleparam.h>
#include "orchestrator_main.h"

/* Global memory usage tracking */
static atomic_t g_total_memory_used = ATOMIC_INIT(0);
static atomic_t g_context_count = ATOMIC_INIT(0);
static DEFINE_SPINLOCK(memory_lock);
static atomic_t g_memory_pressure_flag = ATOMIC_INIT(0);
static bool g_initialized = false;

/* Configurable limits */
static size_t g_max_total_memory = 20 * 1024 * 1024; /* 20MB default */
static size_t g_max_per_conversation = 1 * 1024 * 1024; /* 1MB default */
static size_t g_max_conversations = 100; /* Default max conversations */

module_param(g_max_total_memory, ulong, 0644);
MODULE_PARM_DESC(g_max_total_memory, "Maximum total memory in bytes");
module_param(g_max_per_conversation, ulong, 0644);
MODULE_PARM_DESC(g_max_per_conversation, "Maximum memory per conversation in bytes");
module_param(g_max_conversations, ulong, 0644);
MODULE_PARM_DESC(g_max_conversations, "Maximum number of conversations");

/* Memory pressure levels */
enum memory_pressure {
    PRESSURE_NORMAL = 0,
    PRESSURE_MODERATE = 1,
    PRESSURE_HIGH = 2,
    PRESSURE_CRITICAL = 3
};

/* Track memory usage per conversation */
struct conversation_memory {
    int conversation_id;
    atomic_t memory_used;
    struct list_head list;
    spinlock_t lock; /* Per-conversation lock for better concurrency */
};

/* List of conversation memory usage */
static LIST_HEAD(conversation_memory_list);

/* Workqueue for memory pressure handling */
static struct workqueue_struct *memory_workqueue = NULL;
static struct work_struct memory_pressure_work;

/* Forward declaration */
static void schedule_memory_pressure_work(void);
static void memory_pressure_work_handler(struct work_struct *work);

/* Get current memory pressure level safely */
static enum memory_pressure get_memory_pressure(void)
{
    size_t total_used;
    size_t max_mem;
    u64 percentage;

    total_used = atomic_read(&g_total_memory_used);
    max_mem = g_max_total_memory;

    /* Avoid division by zero */
    if (max_mem == 0)
        return PRESSURE_CRITICAL;

    /* Use u64 for safer arithmetic to prevent overflow */
    percentage = (u64)total_used * 100ULL / max_mem;

    if (percentage > 90)
        return PRESSURE_CRITICAL;
    else if (percentage > 75)
        return PRESSURE_HIGH;
    else if (percentage > 50)
        return PRESSURE_MODERATE;
    else
        return PRESSURE_NORMAL;
}

/* Find conversation memory tracking structure */
static struct conversation_memory *find_conversation_memory(int conversation_id)
{
    struct conversation_memory *mem;

    list_for_each_entry(mem, &conversation_memory_list, list) {
        if (mem->conversation_id == conversation_id)
            return mem;
    }

    return NULL;
}

/* Memory pressure work handler */
static void memory_pressure_work_handler(struct work_struct *work)
{
    enum memory_pressure pressure;

    if (!g_initialized) {
        pr_warn("memory_pressure_work: System not initialized\n");
        return;
    }

    /* Take a reference to the module while work is in progress */
    if (!try_module_get(THIS_MODULE)) {
        pr_warn("memory_pressure_work: Module being unloaded, aborting\n");
        return;
    }

    pressure = get_memory_pressure();

    if (pressure >= PRESSURE_HIGH) {
        context_prune_old_conversations(60 * 60 * 1000); /* 1 hour */
        pr_info("memory_pressure_work: High memory pressure, pruned old conversations\n");
    }

    /* Reset the flag */
    atomic_set(&g_memory_pressure_flag, 0);

    /* Release the module reference */
    module_put(THIS_MODULE);
}

/* Schedule memory pressure handling work */
static void schedule_memory_pressure_work(void)
{
    if (!g_initialized || !memory_workqueue) {
        pr_warn("schedule_memory_pressure_work: Workqueue not initialized\n");
        return;
    }

    /* Only schedule if not already scheduled */
    if (atomic_cmpxchg(&g_memory_pressure_flag, 0, 1) == 0) {
        /* Take a reference on the module before scheduling work */
        if (try_module_get(THIS_MODULE)) {
            queue_work(memory_workqueue, &memory_pressure_work);
        } else {
            /* Module is being unloaded, reset flag */
            atomic_set(&g_memory_pressure_flag, 0);
        }
    }
}

/* Initialize memory management subsystem */
int memory_management_init(void)
{
    if (g_initialized) {
        pr_warn("memory_management_init: Already initialized\n");
        return 0;
    }

    /* Create a dedicated workqueue for memory management */
    memory_workqueue = create_singlethread_workqueue("llm_memory_wq");
    if (!memory_workqueue) {
        pr_err("memory_management_init: Failed to create workqueue\n");
        return -ENOMEM;
    }

    INIT_WORK(&memory_pressure_work, memory_pressure_work_handler);
    g_initialized = true;

    pr_info("Memory management subsystem initialized\n");
    return 0;
}

/* Clean up memory management subsystem */
void memory_management_cleanup(void)
{
    if (!g_initialized) {
        pr_warn("memory_management_cleanup: Not initialized\n");
        return;
    }

    g_initialized = false;

    /* Cancel any pending work and destroy workqueue */
    if (memory_workqueue) {
        flush_workqueue(memory_workqueue);
        destroy_workqueue(memory_workqueue);
        memory_workqueue = NULL;
    }

    /* Clean up memory tracking resources */
    context_cleanup_memory_tracking();

    pr_info("Memory management subsystem cleaned up\n");
}

int context_register_memory(int conversation_id, size_t size)
{
    struct conversation_memory *mem;
    unsigned long flags;
    int ret = 0;
    enum memory_pressure pressure;
    size_t current_total, current_conv_usage;
    bool exceeds_limit = false;

    if (!g_initialized) {
        pr_err("context_register_memory: System not initialized\n");
        return -EINVAL;
    }

    if (conversation_id <= 0 || size == 0)
        return -EINVAL;

    /* Lock hierarchy: Always memory_lock first, then per-conversation lock */
    spin_lock_irqsave(&memory_lock, flags);

    /* Check global memory limit atomically with other operations */
    current_total = atomic_read(&g_total_memory_used);
    if (current_total + size > g_max_total_memory) {
        spin_unlock_irqrestore(&memory_lock, flags);
        pr_warn("context_register_memory: Global memory limit would be exceeded\n");
        return -ENOMEM;
    }

    /* Find or create conversation memory tracking */
    mem = find_conversation_memory(conversation_id);
    if (!mem) {
        /* Check if we've reached max conversations */
        if (atomic_read(&g_context_count) >= g_max_conversations) {
            spin_unlock_irqrestore(&memory_lock, flags);
            pr_warn("context_register_memory: Max conversation count reached\n");
            return -ENOMEM;
        }

        mem = kmalloc(sizeof(*mem), GFP_ATOMIC);
        if (!mem) {
            spin_unlock_irqrestore(&memory_lock, flags);
            return -ENOMEM;
        }

        mem->conversation_id = conversation_id;
        atomic_set(&mem->memory_used, 0);
        INIT_LIST_HEAD(&mem->list);
        spin_lock_init(&mem->lock);

        atomic_inc(&g_context_count);
        list_add(&mem->list, &conversation_memory_list);
    }

    /* Check per-conversation limit without holding the per-conversation lock yet */
    current_conv_usage = atomic_read(&mem->memory_used);
    if (current_conv_usage + size > g_max_per_conversation) {
        exceeds_limit = true;
    }

    if (exceeds_limit) {
        if (mem->conversation_id != conversation_id) {
            /* This is a new conversation that we just created but can't use */
            list_del(&mem->list);
            atomic_dec(&g_context_count);
            kfree(mem);
        }
        spin_unlock_irqrestore(&memory_lock, flags);
        pr_warn("context_register_memory: Per-conversation limit would be exceeded for %d\n",
                conversation_id);
        return -ENOMEM;
    }

    /* Update memory accounting atomically */
    atomic_add(size, &g_total_memory_used);
    atomic_add(size, &mem->memory_used);

    /* Check and save pressure level while still holding the lock */
    pressure = get_memory_pressure();

    spin_unlock_irqrestore(&memory_lock, flags);

    /* Schedule memory pressure handling if needed */
    if (pressure >= PRESSURE_HIGH) {
        schedule_memory_pressure_work();
    }

    return ret;
}


/* Unregister memory allocation with better safety */
void context_unregister_memory(int conversation_id, size_t size)
{
    struct conversation_memory *mem = NULL;
    unsigned long flags;
    size_t actual_size = size;
    int actual_mem_used;

    if (!g_initialized || conversation_id <= 0 || size == 0)
        return;

    /* Lock hierarchy: Always memory_lock first, then per-conversation lock if needed */
    spin_lock_irqsave(&memory_lock, flags);

    mem = find_conversation_memory(conversation_id);
    if (mem) {
        /* Check current memory usage safely */
        actual_mem_used = atomic_read(&mem->memory_used);
        if (actual_mem_used < size)
            actual_size = actual_mem_used;

        /* Update memory accounting */
        if (actual_size > 0) {
            atomic_sub(actual_size, &mem->memory_used);

            /* If conversation has no memory, remove tracking structure */
            if (atomic_read(&mem->memory_used) == 0) {
                list_del(&mem->list);
                kfree(mem);
                atomic_dec(&g_context_count);
                mem = NULL;
            }
        }
    }

    /* Update global counter safely */
    if (actual_size > 0) {
        int global_used = atomic_read(&g_total_memory_used);
        if (global_used < actual_size)
            actual_size = global_used;

        atomic_sub(actual_size, &g_total_memory_used);
    }

    spin_unlock_irqrestore(&memory_lock, flags);
}

/* Get memory usage statistics */
void context_get_memory_stats(size_t *total_used, size_t *max_total,
                              int *conversation_count, int *max_conversations)
{
    if (!g_initialized) {
        /* Return zeros if not initialized */
        if (total_used)
            *total_used = 0;
        if (max_total)
            *max_total = 0;
        if (conversation_count)
            *conversation_count = 0;
        if (max_conversations)
            *max_conversations = 0;
        return;
    }

    if (total_used)
        *total_used = atomic_read(&g_total_memory_used);

    if (max_total)
        *max_total = g_max_total_memory;

    if (conversation_count)
        *conversation_count = atomic_read(&g_context_count);

    if (max_conversations)
        *max_conversations = g_max_conversations;
}

/* Set memory limits with better validation */
int context_set_memory_limits(size_t max_total, size_t max_per_conversation,
                              size_t max_conversations)
{
    unsigned long flags;

    if (!g_initialized)
        return -EINVAL;

    /* Validate limits */
    if (max_total < 1024 * 1024) /* At least 1MB */
        return -EINVAL;

    if (max_per_conversation < 64 * 1024) /* At least 64KB */
        return -EINVAL;

    if (max_conversations < 10) /* At least 10 conversations */
        return -EINVAL;

    /* Check and update limits with proper locking */
    spin_lock_irqsave(&memory_lock, flags);

    /* Check if new limits are smaller than current usage */
    if (atomic_read(&g_total_memory_used) > max_total) {
        spin_unlock_irqrestore(&memory_lock, flags);
        pr_warn("context_set_memory_limits: New total limit smaller than current usage\n");
        return -EBUSY;
    }

    if (atomic_read(&g_context_count) > max_conversations) {
        spin_unlock_irqrestore(&memory_lock, flags);
        pr_warn("context_set_memory_limits: New conversation limit smaller than current count\n");
        return -EBUSY;
    }

    /* Set new limits with validation */
    if (max_total <= SIZE_MAX && max_per_conversation <= SIZE_MAX &&
        max_conversations <= SIZE_MAX) {
        g_max_total_memory = max_total;
        g_max_per_conversation = max_per_conversation;
        g_max_conversations = max_conversations;
        spin_unlock_irqrestore(&memory_lock, flags);
        return 0;
    }

    spin_unlock_irqrestore(&memory_lock, flags);
    return -EINVAL;
}

/* Clean up memory tracking resources */
void context_cleanup_memory_tracking(void)
{
    struct conversation_memory *mem, *tmp;
    unsigned long flags;

    if (!g_initialized)
        return;

    spin_lock_irqsave(&memory_lock, flags);

    list_for_each_entry_safe(mem, tmp, &conversation_memory_list, list) {
        list_del(&mem->list);
        kfree(mem);
    }

    atomic_set(&g_total_memory_used, 0);
    atomic_set(&g_context_count, 0);

    spin_unlock_irqrestore(&memory_lock, flags);
}

/* Allocate context entry with memory tracking */
struct context_entry *context_allocate_entry(int conversation_id)
{
    struct context_entry *entry;
    size_t entry_size = sizeof(struct context_entry);

    /* Validate input */
    if (!g_initialized || conversation_id <= 0)
        return NULL;

    /* Check and register memory allocation */
    if (context_register_memory(conversation_id, entry_size))
        return NULL;

    entry = kmalloc(entry_size, GFP_KERNEL);
    if (!entry) {
        context_unregister_memory(conversation_id, entry_size);
        return NULL;
    }

    /* Initialize and return */
    memset(entry, 0, entry_size);
    INIT_LIST_HEAD(&entry->list);

    return entry;
}

/* Free entry with memory tracking */
void context_free_entry(int conversation_id, struct context_entry *entry)
{
    if (!g_initialized || !entry || conversation_id <= 0)
        return;

    context_unregister_memory(conversation_id, sizeof(struct context_entry));
    kfree(entry);
}

/* Sysfs memory stats show function with safer formatting - avoiding floats */
ssize_t memory_stats_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    size_t total_used, max_total;
    int conversation_count, max_conversations;
    const char *pressure_level;
    u64 usage_percent;
    u64 used_mb, max_mb;
    u64 used_mb_frac, max_mb_frac;

    if (!g_initialized)
        return scnprintf(buf, PAGE_SIZE, "Memory management subsystem not initialized\n");

    context_get_memory_stats(&total_used, &max_total, &conversation_count, &max_conversations);

    /* Safer percentage calculation to avoid division by zero and use integer arithmetic */
    if (max_total > 0) {
        usage_percent = (u64)total_used * 100ULL / (u64)max_total;
    } else {
        usage_percent = 100;
    }

    /* Convert bytes to MB using integer arithmetic (1 MB = 1048576 bytes) */
    used_mb = total_used / 1048576ULL;
    used_mb_frac = ((total_used % 1048576ULL) * 100ULL) / 1048576ULL;

    max_mb = max_total / 1048576ULL;
    max_mb_frac = ((max_total % 1048576ULL) * 100ULL) / 1048576ULL;

    /* Convert pressure level to string */
    switch (get_memory_pressure()) {
        case PRESSURE_NORMAL:
            pressure_level = "Normal";
            break;
        case PRESSURE_MODERATE:
            pressure_level = "Moderate";
            break;
        case PRESSURE_HIGH:
            pressure_level = "High";
            break;
        case PRESSURE_CRITICAL:
            pressure_level = "Critical";
            break;
        default:
            pressure_level = "Unknown";
            break;
    }

    return scnprintf(buf, PAGE_SIZE,
                     "Memory Usage Statistics:\n"
                     "  Total Used: %zu bytes (%llu.%02llu MB)\n"
                     "  Maximum Allowed: %zu bytes (%llu.%02llu MB)\n"
                     "  Usage: %llu%%\n"
                     "  Active Conversations: %d / %d\n"
                     "  Pressure Level: %s\n",
                     total_used, used_mb, used_mb_frac,
                     max_total, max_mb, max_mb_frac,
                     usage_percent,
                     conversation_count, max_conversations,
                     pressure_level);
}

/* Sysfs memory limits show function - avoiding floats */
ssize_t memory_limits_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    u64 max_total_mb, max_per_conv_mb, max_total_frac, max_per_conv_frac;

    if (!g_initialized)
        return scnprintf(buf, PAGE_SIZE, "Memory management subsystem not initialized\n");

    /* Convert bytes to MB using integer arithmetic */
    max_total_mb = g_max_total_memory / 1048576ULL;
    max_total_frac = ((g_max_total_memory % 1048576ULL) * 100ULL) / 1048576ULL;

    max_per_conv_mb = g_max_per_conversation / 1048576ULL;
    max_per_conv_frac = ((g_max_per_conversation % 1048576ULL) * 100ULL) / 1048576ULL;

    return scnprintf(buf, PAGE_SIZE,
                     "Memory Limits:\n"
                     "  Total Memory: %zu bytes (%llu.%02llu MB)\n"
                     "  Per-Conversation: %zu bytes (%llu.%02llu MB)\n"
                     "  Max Conversations: %zu\n",
                     g_max_total_memory, max_total_mb, max_total_frac,
                     g_max_per_conversation, max_per_conv_mb, max_per_conv_frac,
                     g_max_conversations);
}

/* Sysfs memory limits store function with better error reporting */
ssize_t memory_limits_store(struct device *dev, struct device_attribute *attr,
                            const char *buf, size_t count)
{
    size_t max_total, max_per_conversation, max_conversations;
    int ret;

    if (!g_initialized)
        return -EINVAL;

    ret = sscanf(buf, "%zu,%zu,%zu", &max_total, &max_per_conversation, &max_conversations);
    if (ret != 3) {
        pr_err("memory_limits_store: Invalid input format. Expected 'total,per_conv,max_count'\n");
        return -EINVAL;
    }

    ret = context_set_memory_limits(max_total, max_per_conversation, max_conversations);
    if (ret) {
        switch (ret) {
            case -EINVAL:
                pr_err("memory_limits_store: Invalid memory limits specified\n");
                break;
            case -EBUSY:
                pr_err("memory_limits_store: Cannot reduce limits below current usage\n");
                break;
            default:
                pr_err("memory_limits_store: Error setting memory limits: %d\n", ret);
                break;
        }
        return ret;
    }

    return count;
}

EXPORT_SYMBOL(context_register_memory);
EXPORT_SYMBOL(context_unregister_memory);
EXPORT_SYMBOL(context_get_memory_stats);
EXPORT_SYMBOL(context_set_memory_limits);
EXPORT_SYMBOL(context_cleanup_memory_tracking);
EXPORT_SYMBOL(context_allocate_entry);
EXPORT_SYMBOL(context_free_entry);
EXPORT_SYMBOL(memory_management_init);
EXPORT_SYMBOL(memory_management_cleanup);
EXPORT_SYMBOL(memory_stats_show);
EXPORT_SYMBOL(memory_limits_show);
EXPORT_SYMBOL(memory_limits_store);
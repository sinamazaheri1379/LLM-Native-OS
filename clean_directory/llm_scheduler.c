/*
 * LLM Scheduler using Strict Fibonacci Heap (Improved Version)
 *
 * This implementation follows the strict Fibonacci heap structure with fixes
 * for potential issues in the previous version:
 * 1. More robust maintenance of first_linkable pointer
 * 2. Improved node traversal logic in extract-min
 * 3. Better handling of edge cases in loss reduction
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/spinlock.h>
#include <linux/slab.h>
#include <linux/ktime.h>
#include <linux/atomic.h>
#include "orchestrator_main.h"

/* Constants for scheduler logic */
#define WEIGHT_TOTAL_PERCENT     100
#define MIN_PROVIDER_WEIGHT      5
#define RATE_LIMIT_PENALTY       20
#define DEFAULT_TOKEN_WEIGHT     100
#define TOKEN_WEIGHT_FACTOR      1000000
#define METRICS_ADJUST_INTERVAL  10
#define MAX_RANK                 32 /* log₂(n) for large n */

/* Priority levels for scheduling */
#define PRIORITY_HIGH      0
#define PRIORITY_NORMAL    1
#define PRIORITY_LOW       2
#define PRIORITY_LEVELS    3

/* Node types */
#define NODE_ACTIVE        0
#define NODE_PASSIVE       1

/* Provider model information */
static const char *openai_default_model = "gpt-4o";
static const char *anthropic_default_model = "claude-3-7-sonnet-20250219";
static const char *gemini_default_model = "gemini-1.5-pro";

static const char *openai_supported_models[] = {
        "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", NULL
};

static const char *anthropic_supported_models[] = {
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
        "claude-3-7-sonnet-20250219", NULL
};

static const char *gemini_supported_models[] = {
        "gemini-1.5-pro", "gemini-1.0-pro", NULL
};

/* Key box structure to maintain stable references */
struct key_box {
    int priority;                 /* Priority value (smaller = higher priority) */
    ktime_t submit_time;          /* Submission time for tiebreaking */
    struct strict_fib_node *node; /* Pointer back to the node */
};

/* Strict Fibonacci heap node structure */
struct strict_fib_node {
    struct strict_fib_node *parent;      /* Parent node */
    struct strict_fib_node *child;       /* Leftmost child node */
    struct strict_fib_node *left;        /* Left sibling (circular list) */
    struct strict_fib_node *right;       /* Right sibling (circular list) */
    struct key_box *key_box;             /* Box containing priority value */
    struct llm_request *req;             /* Request data */
    int node_type;                       /* Active (0) or passive (1) */
    bool is_linkable;                    /* For passive nodes: can be linked to root */
    unsigned int rank;                   /* Number of active children */
    unsigned int loss;                   /* Number of active children lost */
    int request_id;                      /* Unique request ID */
};

/* Structure for tracking nodes that need fixing */
struct fix_list {
    struct strict_fib_node *nodes[MAX_RANK][2]; /* Nodes by rank and loss */
    int count[MAX_RANK][2];                    /* Count of nodes in each category */
};

/* Strict Fibonacci heap structure */
struct strict_fib_heap {
    struct strict_fib_node *root;        /* Root of the heap (always passive) */
    struct strict_fib_node *first_linkable; /* First passive linkable child of root */
    struct fix_list fix_list;            /* List of nodes that need fixing */
    unsigned int n;                      /* Number of nodes in heap */
    spinlock_t lock;                     /* Heap lock */
};

/* Flag used for batch marking of nodes */
struct active_flag {
    int value;  /* 0 = active, 1 = passive */
};

/* Global priority queues - one per priority level */
static struct strict_fib_heap priority_queues[PRIORITY_LEVELS];

/* Provider metrics and weight locks */
static DEFINE_SPINLOCK(metrics_lock);
static DEFINE_SPINLOCK(weights_lock);
static atomic_t request_counter = ATOMIC_INIT(0);

/*
 * Initialize a new key box
 * Returns pointer to initialized box or NULL on failure
 */
static struct key_box *key_box_new(int priority, ktime_t submit_time)
{
    struct key_box *box;

    box = kmalloc(sizeof(*box), GFP_KERNEL);
    if (!box)
        return NULL;

    box->priority = priority;
    box->submit_time = submit_time;
    box->node = NULL; /* Will be set when linked to node */

    return box;
}

/*
 * Free a key box
 */
static void key_box_free(struct key_box *box)
{
    if (box)
        kfree(box);
}

/*
 * Initialize a new Strict Fibonacci heap node
 * Returns pointer to initialized node or NULL on failure
 */
static struct strict_fib_node *strict_fib_node_new(struct llm_request *req, struct key_box *box)
{
    struct strict_fib_node *node;

    /* Allocate node */
    node = kmalloc(sizeof(*node), GFP_KERNEL);
    if (!node)
        return NULL;

    /* Make a copy of the request */
    if (req) {
        node->req = kmalloc(sizeof(struct llm_request), GFP_KERNEL);
        if (!node->req) {
            kfree(node);
            return NULL;
        }

        memcpy(node->req, req, sizeof(struct llm_request));
    } else {
        node->req = NULL;
    }

    /* Initialize node */
    node->parent = NULL;
    node->child = NULL;
    node->left = node;  /* Point to itself (singleton circular list) */
    node->right = node;
    node->key_box = box;
    node->node_type = NODE_PASSIVE; /* Start as passive node */
    node->is_linkable = true;      /* Passive node with no children is linkable */
    node->rank = 0;
    node->loss = 0;
    node->request_id = req ? atomic_inc_return(&request_counter) : 0;

    /* Link box to node */
    if (box)
        box->node = node;

    return node;
}

/*
 * Free a Strict Fibonacci heap node and its request
 */
static void strict_fib_node_free(struct strict_fib_node *node)
{
    if (!node)
        return;

    if (node->key_box) {
        node->key_box->node = NULL; /* Unlink from box */
        key_box_free(node->key_box);
        node->key_box = NULL;
    }

    if (node->req) {
        kfree(node->req);
        node->req = NULL;
    }

    kfree(node);
}

/*
 * Add node2 to the right of node1 in the circular linked list
 */
static void strict_fib_add_to_list(struct strict_fib_node *node1, struct strict_fib_node *node2)
{
    if (!node1 || !node2)
        return;

    /* Insert node2 between node1 and node1->right */
    node2->right = node1->right;
    node1->right->left = node2;
    node1->right = node2;
    node2->left = node1;
}

/*
 * Remove a node from its circular linked list
 */
static void strict_fib_remove_from_list(struct strict_fib_node *node)
{
    if (!node)
        return;

    /* Only proceed if node is in a list with others */
    if (node->left != node) {
        node->left->right = node->right;
        node->right->left = node->left;

        /* Isolate the node (point to itself) */
        node->left = node;
        node->right = node;
    }
}

/*
 * Update first_linkable pointer after operations
 * FIX #1: This is a new helper function to ensure consistent maintenance
 * of the first_linkable pointer
 */
static void update_first_linkable(struct strict_fib_heap *heap)
{
    struct strict_fib_node *child;

    if (!heap || !heap->root || !heap->root->child) {
        heap->first_linkable = NULL;
        return;
    }

    /* Reset first_linkable */
    heap->first_linkable = NULL;

    /* Find the first passive linkable child */
    child = heap->root->child;
    do {
        if (child->node_type == NODE_PASSIVE && child->is_linkable) {
            heap->first_linkable = child;
            break;
        }
        child = child->right;
    } while (child != heap->root->child);
}

/*
 * Make node2 a child of node1, respecting active/passive ordering
 */
static void strict_fib_add_child(struct strict_fib_node *node1, struct strict_fib_node *node2)
{
    if (!node1 || !node2)
        return;

    /* Remove node2 from its current list if any */
    if (node2->parent)
        strict_fib_remove_from_list(node2);

    /* Set parent relationship */
    node2->parent = node1;

    if (!node1->child) {
        /* node1 had no children */
        node1->child = node2;
        node2->left = node2;
        node2->right = node2;
    } else {
        /* Add to child list in proper position based on active/passive status */
        if (node2->node_type == NODE_ACTIVE) {
            /* Active nodes go to the left */
            struct strict_fib_node *child = node1->child;

            /* Find leftmost passive child or end of list */
            while (child->left != node1->child && child->node_type == NODE_ACTIVE)
                child = child->left;

            if (child->node_type == NODE_ACTIVE) {
                /* All children are active, add to list */
                strict_fib_add_to_list(child, node2);
            } else {
                /* Add before the first passive child */
                strict_fib_add_to_list(child->left, node2);
            }
        } else {
            /* Passive nodes go to the right */
            strict_fib_add_to_list(node1->child->left, node2);
        }
    }

    /* Update rank if adding active child */
    if (node2->node_type == NODE_ACTIVE)
        node1->rank++;

    /* Check if we need to update linkable flag for passive nodes */
    if (node2->node_type == NODE_PASSIVE && node2->child) {
        /* A passive node with children is not linkable */
        node2->is_linkable = false;
    }
}

/*
 * Add a node to the fix list at the given rank and loss
 */
static void add_to_fix_list(struct strict_fib_heap *heap, struct strict_fib_node *node)
{
    unsigned int rank, loss_idx;

    if (!heap || !node || node->node_type != NODE_ACTIVE)
        return;

    rank = node->rank;
    /* Only track nodes with loss 0 or 1 */
    loss_idx = (node->loss >= 1) ? 1 : 0;

    /* Add to fix list if rank is within range */
    if (rank < MAX_RANK) {
        heap->fix_list.nodes[rank][loss_idx] = node;
        heap->fix_list.count[rank][loss_idx]++;
    }
}

/*
 * Remove a node from the fix list
 */
static void remove_from_fix_list(struct strict_fib_heap *heap, struct strict_fib_node *node)
{
    unsigned int rank, loss_idx;

    if (!heap || !node || node->node_type != NODE_ACTIVE)
        return;

    rank = node->rank;
    /* Only track nodes with loss 0 or 1 */
    loss_idx = (node->loss >= 1) ? 1 : 0;

    /* Remove from fix list if rank is within range */
    if (rank < MAX_RANK && heap->fix_list.nodes[rank][loss_idx] == node) {
        heap->fix_list.nodes[rank][loss_idx] = NULL;
        heap->fix_list.count[rank][loss_idx]--;
    }
}

/*
 * Find nodes in the fix list with equal rank and loss=1
 * Returns true if candidates were found and performs the loss reduction
 * FIX #3: Improved to handle more edge cases and search more thoroughly
 */
static bool perform_loss_reduction(struct strict_fib_heap *heap)
{
    int i;
    struct strict_fib_node *x, *y, *parent;
    struct strict_fib_node *candidates[MAX_RANK]; /* Track all candidates by rank */
    int candidate_counts[MAX_RANK] = {0};        /* Count candidates for each rank */

    /* First pass: collect candidates with loss >= 1 */
    for (i = 0; i < MAX_RANK; i++) {
        /* Initialize candidate tracking */
        candidates[i] = NULL;
        candidate_counts[i] = 0;

        /* Case 1: single node with loss >= 2 */
        if (heap->fix_list.nodes[i][1] && heap->fix_list.nodes[i][1]->loss >= 2) {
            x = heap->fix_list.nodes[i][1];

            /* Decrease loss by 2 */
            x->loss -= 2;

            /* If x is no longer a loss reduction candidate, update fix list */
            if (x->loss <= 1) {
                remove_from_fix_list(heap, x);
                add_to_fix_list(heap, x);
            }

            return true;
        }

        /* Add all nodes with loss=1 to candidates for this rank */
        if (heap->fix_list.count[i][1] > 0) {
            /* Find all nodes with the same rank and loss=1 */
            struct strict_fib_node *temp;

            /* Start with the one in the fix list */
            candidates[i] = heap->fix_list.nodes[i][1];
            candidate_counts[i] = 1;

            /* Find other nodes with the same rank and loss=1 */
            if (candidates[i] && candidates[i]->parent && candidates[i]->parent->child) {
                temp = candidates[i]->parent->child;
                do {
                    if (temp != candidates[i] && temp->node_type == NODE_ACTIVE &&
                        temp->rank == i && temp->loss == 1) {
                        /* We've found a second candidate - we can perform reduction */
                        x = candidates[i];
                        y = temp;

                        /* Ensure x->key <= y->key */
                        if (x->key_box->priority > y->key_box->priority ||
                            (x->key_box->priority == y->key_box->priority &&
                             ktime_after(x->key_box->submit_time, y->key_box->submit_time))) {
                            /* Swap x and y */
                            struct strict_fib_node *temp_ptr = x;
                            x = y;
                            y = temp_ptr;
                        }

                        /* Get parent of y */
                        parent = y->parent;

                        /* Detach y from its parent */
                        strict_fib_remove_from_list(y);
                        parent->rank--;

                        /* Make y a child of x */
                        strict_fib_add_child(x, y);

                        /* Reset losses */
                        x->loss = 0;
                        y->loss = 0;

                        /* Update fix list */
                        remove_from_fix_list(heap, x);
                        remove_from_fix_list(heap, y);
                        add_to_fix_list(heap, x);
                        add_to_fix_list(heap, parent);

                        return true;
                    }
                    temp = temp->right;
                } while (temp != candidates[i]->parent->child);
            }
        }
    }

    /* Second pass: look for nodes across different parents but same rank */
    for (i = 0; i < MAX_RANK; i++) {
        if (heap->fix_list.count[i][1] >= 2) {
            /* Try to find two nodes from different parents */
            struct strict_fib_node *first = NULL;
            struct strict_fib_node *second = NULL;

            /* Traverse all nodes in the tree (starting from root) */
            if (heap->root && heap->root->child) {
                struct strict_fib_node *child = heap->root->child;

                /* First level search - direct children of root */
                do {
                    /* If child is active and has rank i nodes with loss=1 */
                    if (child->node_type == NODE_ACTIVE && child->child) {
                        /* Search child's children */
                        struct strict_fib_node *grandchild = child->child;
                        do {
                            if (grandchild->node_type == NODE_ACTIVE &&
                                grandchild->rank == i && grandchild->loss == 1) {
                                /* Found a candidate */
                                if (!first) {
                                    first = grandchild;
                                } else if (grandchild->parent != first->parent) {
                                    /* Found a second candidate with different parent */
                                    second = grandchild;
                                    break;
                                }
                            }
                            grandchild = grandchild->right;
                        } while (grandchild != child->child);
                    }

                    if (first && second) break; /* Found two candidates */

                    child = child->right;
                } while (child != heap->root->child);

                if (first && second) {
                    /* Perform loss reduction with these two nodes */
                    x = first;
                    y = second;

                    /* Ensure x->key <= y->key */
                    if (x->key_box->priority > y->key_box->priority ||
                        (x->key_box->priority == y->key_box->priority &&
                         ktime_after(x->key_box->submit_time, y->key_box->submit_time))) {
                        /* Swap x and y */
                        struct strict_fib_node *temp = x;
                        x = y;
                        y = temp;
                    }

                    /* Get parent of y */
                    parent = y->parent;

                    /* Detach y from its parent */
                    strict_fib_remove_from_list(y);
                    parent->rank--;

                    /* Make y a child of x */
                    strict_fib_add_child(x, y);

                    /* Reset losses */
                    x->loss = 0;
                    y->loss = 0;

                    /* Update fix list */
                    remove_from_fix_list(heap, x);
                    remove_from_fix_list(heap, y);
                    add_to_fix_list(heap, x);
                    add_to_fix_list(heap, parent);

                    return true;
                }
            }
        }
    }

    return false;
}

/*
 * Perform active root reduction on the heap
 * Returns true if reduction was performed
 */
static bool perform_active_root_reduction(struct strict_fib_heap *heap)
{
    struct strict_fib_node *active_root = NULL;
    struct strict_fib_node *child, *next;

    /* Find an active root (active node with passive parent) */
    if (heap->root && heap->root->child) {
        child = heap->root->child;
        do {
            if (child->node_type == NODE_ACTIVE) {
                active_root = child;
                break;
            }
            child = child->right;
        } while (child != heap->root->child);
    }

    if (active_root) {
        /* Make the active root passive */
        active_root->node_type = NODE_PASSIVE;
        active_root->is_linkable = active_root->child ? false : true;

        /* Update parent's rank */
        active_root->parent->rank--;

        /* All of active_root's active children become active roots */
        if (active_root->child) {
            /* FIX #2: Improved traversal logic for handling active children */
            struct strict_fib_node *active_children[64]; /* Array to store active children */
            int count = 0;
            int i;

            /* First collect all active children */
            child = active_root->child;
            do {
                next = child->right;
                if (child->node_type == NODE_ACTIVE && count < 64) {
                    active_children[count++] = child;
                }
                child = next;
            } while (child != active_root->child && active_root->child);

            /* Now process collected active children */
            for (i = 0; i < count; i++) {
                child = active_children[i];

                /* Remove from active_root */
                strict_fib_remove_from_list(child);
                if (active_root->child == child) {
                    active_root->child = (child->right != child) ? child->right : NULL;
                }
                active_root->rank--;

                /* Add as child of root */
                strict_fib_add_child(heap->root, child);
            }
        }

        /* Update first_linkable pointer if needed */
        if (active_root->is_linkable) {
            update_first_linkable(heap);
        }

        return true;
    }

    return false;
}

/*
 * Perform root degree reduction
 * Returns true if reduction was performed
 * FIX #1: Improved handling of first_linkable pointer
 */
static bool perform_root_degree_reduction(struct strict_fib_heap *heap)
{
    struct strict_fib_node *x, *y, *z;
    struct strict_fib_node *passive_children[3] = {NULL};
    int count = 0;

    /* Check if we have at least 3 passive linkable children of the root */
    if (!heap->first_linkable)
        return false;

    /* Find three passive linkable children */
    x = heap->first_linkable;
    do {
        if (x->node_type == NODE_PASSIVE && x->is_linkable && count < 3) {
            passive_children[count++] = x;
        }
        x = x->right;
    } while (count < 3 && x != heap->first_linkable && x != heap->root->child->left);

    if (count < 3)
        return false;

    /* Get the three nodes */
    x = passive_children[0];
    y = passive_children[1];
    z = passive_children[2];

    /* Sort them by key: x.key <= y.key <= z.key */
    if (x->key_box && y->key_box && (
        x->key_box->priority > y->key_box->priority ||
        (x->key_box->priority == y->key_box->priority &&
         ktime_after(x->key_box->submit_time, y->key_box->submit_time)))) {
        struct strict_fib_node *temp = x;
        x = y;
        y = temp;
    }

    if (y->key_box && z->key_box && (
        y->key_box->priority > z->key_box->priority ||
        (y->key_box->priority == z->key_box->priority &&
         ktime_after(y->key_box->submit_time, z->key_box->submit_time)))) {
        struct strict_fib_node *temp = y;
        y = z;
        z = temp;

        /* Check x and y again after swapping */
        if (x->key_box && y->key_box && (
            x->key_box->priority > y->key_box->priority ||
            (x->key_box->priority == y->key_box->priority &&
             ktime_after(x->key_box->submit_time, y->key_box->submit_time)))) {
            temp = x;
            x = y;
            y = temp;
        }
    }

    /* Detach from the root */
    strict_fib_remove_from_list(x);
    strict_fib_remove_from_list(y);
    strict_fib_remove_from_list(z);

    /* Change x and y to active nodes */
    x->node_type = NODE_ACTIVE;
    y->node_type = NODE_ACTIVE;

    /* Link z to y, y to x, and x as leftmost child of root */
    strict_fib_add_child(y, z);
    strict_fib_add_child(x, y);
    strict_fib_add_child(heap->root, x);

    /* Update first_linkable pointer */
    update_first_linkable(heap);

    return true;
}

/*
 * Initialize a Strict Fibonacci heap
 */
static void strict_fib_heap_init(struct strict_fib_heap *heap)
{
    int i, j;

    /* Create passive root node */
    heap->root = strict_fib_node_new(NULL, NULL);
    if (!heap->root) {
        pr_err("strict_fib_heap_init: Failed to allocate root node\n");
        return;
    }

    /* Root is always passive and has no request */
    heap->root->node_type = NODE_PASSIVE;
    heap->first_linkable = NULL;
    heap->n = 0;

    /* Initialize fix list */
    for (i = 0; i < MAX_RANK; i++) {
        for (j = 0; j < 2; j++) {
            heap->fix_list.nodes[i][j] = NULL;
            heap->fix_list.count[i][j] = 0;
        }
    }

    spin_lock_init(&heap->lock);
}

/*
 * Insert a node into a Strict Fibonacci heap
 * Returns 0 on success, negative error code on failure
 */
static int strict_fib_heap_insert(struct strict_fib_heap *heap, struct llm_request *req, int priority)
{
    struct strict_fib_node *node;
    struct key_box *box;
    unsigned long flags;

    if (!heap || !req)
        return -EINVAL;

    /* Create key box and node */
    box = key_box_new(priority, ktime_get());
    if (!box)
        return -ENOMEM;

    node = strict_fib_node_new(req, box);
    if (!node) {
        key_box_free(box);
        return -ENOMEM;
    }

    spin_lock_irqsave(&heap->lock, flags);

    /* Make node passive and linkable */
    node->node_type = NODE_PASSIVE;
    node->is_linkable = true;

    /* Add to root's children */
    strict_fib_add_child(heap->root, node);

    /* Update linkable pointer if needed */
    if (!heap->first_linkable) {
        heap->first_linkable = node;
    }

    heap->n++;

    spin_unlock_irqrestore(&heap->lock, flags);

    return 0;
}

/*
 * Find and extract the minimum node from a Strict Fibonacci heap
 * Returns the minimum node or NULL if heap is empty
 * FIX #2: Improved node traversal logic in extract-min
 */
static struct strict_fib_node *strict_fib_heap_extract_min(struct strict_fib_heap *heap)
{
    struct strict_fib_node *min = NULL;
    struct strict_fib_node *child;
    unsigned long flags;

    spin_lock_irqsave(&heap->lock, flags);

    if (heap->n == 0 || !heap->root || !heap->root->child) {
        spin_unlock_irqrestore(&heap->lock, flags);
        return NULL;
    }

    /* Find minimum key among root's children */
    child = heap->root->child;
    min = child;

    /* Find node with minimum key */
    do {
        if (child->key_box && min->key_box &&
            (child->key_box->priority < min->key_box->priority ||
             (child->key_box->priority == min->key_box->priority &&
              ktime_before(child->key_box->submit_time, min->key_box->submit_time)))) {
            min = child;
        }
        child = child->right;
    } while (child != heap->root->child);

    /* If min is active, make it passive */
    if (min->node_type == NODE_ACTIVE) {
        /* FIX #2: Improved handling of active children during extract-min */
        struct strict_fib_node **active_children = NULL;
        int active_count = 0;
        int i;

        /* First count active children */
        if (min->child) {
            child = min->child;
            do {
                if (child->node_type == NODE_ACTIVE) {
                    active_count++;
                }
                child = child->right;
            } while (child != min->child);

            /* Allocate array if we have active children */
            if (active_count > 0) {
                active_children = kmalloc(sizeof(struct strict_fib_node *) * active_count, GFP_ATOMIC);
                if (!active_children) {
                    /* Memory allocation failed, fall back to original logic */
                    active_count = 0;
                } else {
                    /* Fill the array */
                    i = 0;
                    child = min->child;
                    do {
                        if (child->node_type == NODE_ACTIVE && i < active_count) {
                            active_children[i++] = child;
                        }
                        child = child->right;
                    } while (child != min->child);
                }
            }
        }

        /* Change min's status */
        min->node_type = NODE_PASSIVE;
        min->parent->rank--;

        /* Process active children if we have them */
        if (active_count > 0 && active_children) {
            for (i = 0; i < active_count; i++) {
                child = active_children[i];

                /* Remove from min */
                strict_fib_remove_from_list(child);
                if (min->child == child) {
                    min->child = (child->right != child) ? child->right : NULL;
                }
                min->rank--;

                /* Add to root */
                strict_fib_add_child(heap->root, child);
            }

            /* Free the array */
            kfree(active_children);
        }
    }

    /* Gather all of min's remaining (passive) children into an array */
    if (min->child) {
        struct strict_fib_node **children = NULL;
        int child_count = 0;
        int i;

        /* Count remaining children */
        child = min->child;
        do {
            child_count++;
            child = child->right;
        } while (child != min->child);

        /* Allocate array to hold children */
        if (child_count > 0) {
            children = kmalloc(sizeof(struct strict_fib_node *) * child_count, GFP_ATOMIC);
            if (children) {
                /* Fill the array */
                i = 0;
                child = min->child;
                do {
                    children[i++] = child;
                    child = child->right;
                } while (child != min->child && i < child_count);

                /* Process children */
                for (i = 0; i < child_count; i++) {
                    child = children[i];

                    /* Remove from min */
                    strict_fib_remove_from_list(child);
                    if (min->child == child) {
                        min->child = (child->right != child) ? child->right : NULL;
                    }

                    /* Add to root */
                    strict_fib_add_child(heap->root, child);
                }

                /* Free the array */
                kfree(children);
            }
        }
    }

    /* Remove min from the root's children */
    strict_fib_remove_from_list(min);
    if (heap->root->child == min) {
        heap->root->child = (min->right != min) ? min->right : NULL;
    }
    min->parent = NULL;

    /* Decrement count */
    heap->n--;

    /* Update first_linkable pointer */
    update_first_linkable(heap);

    /* Perform transformations to maintain heap properties */
    while (perform_loss_reduction(heap) ||
           perform_active_root_reduction(heap) ||
           perform_root_degree_reduction(heap)) {
        /* Continue until no more transformations are possible */
    }

    spin_unlock_irqrestore(&heap->lock, flags);

    return min;
}



/*
 * Clean up a Strict Fibonacci heap
 */
static void strict_fib_heap_cleanup(struct strict_fib_heap *heap)
{
    struct strict_fib_node *node;

    if (!heap)
        return;

    /* Extract and free all nodes */
    while ((node = strict_fib_heap_extract_min(heap)) != NULL) {
        strict_fib_node_free(node);
    }

    /* Free the root */
    if (heap->root) {
        strict_fib_node_free(heap->root);
        heap->root = NULL;
    }
}

/*
 * Initialize scheduler state with Strict Fibonacci heaps
 */
void scheduler_init(struct scheduler_state *state)
{
    int i;
    unsigned long flags;

    if (!state) {
        pr_err("scheduler_init: Invalid state pointer\n");
        return;
    }

    /* Initialize algorithm and scheduling parameters */
    atomic_set(&state->current_algorithm, SCHEDULER_ROUND_ROBIN);
    state->next_provider = 0;
    state->auto_adjust = true;

    /* Initialize weights with proper locking */
    spin_lock_irqsave(&weights_lock, flags);

    /* Initial equal weights */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        state->weights[i] = WEIGHT_TOTAL_PERCENT / PROVIDER_COUNT;
        state->provider_priority[i] = i;
    }

    /* Ensure weights add up to 100% */
    state->weights[PROVIDER_COUNT - 1] = WEIGHT_TOTAL_PERCENT;
    for (i = 0; i < PROVIDER_COUNT - 1; i++) {
        state->weights[PROVIDER_COUNT - 1] -= state->weights[i];
    }

    /* Ensure minimum weight for all providers */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (state->weights[i] < MIN_PROVIDER_WEIGHT)
            state->weights[i] = MIN_PROVIDER_WEIGHT;
    }

    spin_unlock_irqrestore(&weights_lock, flags);

    /* Initialize metrics with proper locking */
    spin_lock_irqsave(&metrics_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        atomic_set(&state->metrics[i].current_status, 1); /* Available */
        atomic_set(&state->metrics[i].total_requests, 0);
        atomic_set(&state->metrics[i].successful_requests, 0);
        atomic_set(&state->metrics[i].failed_requests, 0);
        atomic_set(&state->metrics[i].timeouts, 0);
        atomic_set(&state->metrics[i].rate_limited, 0);
        atomic64_set(&state->metrics[i].total_latency_ms, 0);
        state->metrics[i].min_latency_ms = ULONG_MAX;
        state->metrics[i].max_latency_ms = 0;
        atomic_set(&state->metrics[i].total_tokens, 0);
        state->metrics[i].rate_limit_reset_time = 0;
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    /* Initialize FIFO queue */
    fifo_init(&state->fifo);

    /* Initialize priority queues */
    for (i = 0; i < PRIORITY_LEVELS; i++) {
        strict_fib_heap_init(&priority_queues[i]);
    }

    pr_info("LLM scheduler initialized with Strict Fibonacci heap-based priority scheduling\n");
}

/*
 * Submit a request to the priority queue
 * Returns 0 on success, negative error code on failure
 */
int scheduler_submit_request(struct llm_request *req, int priority)
{
    int ret;

    /* Validate request */
    if (!req) {
        pr_err("scheduler_submit_request: Invalid request\n");
        return -EINVAL;
    }

    /* Validate priority */
    if (priority < 0 || priority >= PRIORITY_LEVELS) {
        priority = PRIORITY_NORMAL; /* Default to normal priority */
    }

    /* Insert into appropriate priority queue */
    ret = strict_fib_heap_insert(&priority_queues[priority], req, priority);
    if (ret) {
        pr_err("scheduler_submit_request: Failed to insert into heap: %d\n", ret);
        return ret;
    }

    pr_debug("scheduler_submit_request: Request submitted with priority %d\n", priority);
    return 0;
}

/*
 * Get the next request from priority queues
 * Returns the request pointer on success, NULL if no requests
 */
struct llm_request *scheduler_get_next_request(void)
{
    struct strict_fib_node *node = NULL;
    struct llm_request *req = NULL;
    int i;

    /* Check each priority queue in order */
    for (i = 0; i < PRIORITY_LEVELS; i++) {
        node = strict_fib_heap_extract_min(&priority_queues[i]);
        if (node) {
            break;
        }
    }

    if (node) {
        req = node->req;
        /* Don't free the request, it will be returned to the caller */
        node->req = NULL;
        strict_fib_node_free(node);
    }

    return req;
}

/*
 * Get default model for provider
 * Returns the default model string or NULL if provider is invalid
 */
const char *get_default_model(int provider)
{
    if (provider < 0 || provider >= PROVIDER_COUNT)
        return NULL;

    switch (provider) {
        case PROVIDER_OPENAI:
            return openai_default_model;
        case PROVIDER_ANTHROPIC:
            return anthropic_default_model;
        case PROVIDER_GOOGLE_GEMINI:
            return gemini_default_model;
        default:
            return NULL;
    }
}

/*
 * Check if a model is supported by a provider
 * Returns true if model is valid and supported, false otherwise
 */
bool is_model_supported(int provider, const char *model_name)
{
    const char **models;
    int i;

    if (!model_name || !model_name[0])
        return false;

    if (provider < 0 || provider >= PROVIDER_COUNT)
        return false;

    switch (provider) {
        case PROVIDER_OPENAI:
            models = openai_supported_models;
            break;
        case PROVIDER_ANTHROPIC:
            models = anthropic_supported_models;
            break;
        case PROVIDER_GOOGLE_GEMINI:
            models = gemini_supported_models;
            break;
        default:
            return false;
    }

    for (i = 0; models[i] != NULL; i++) {
        if (strcmp(model_name, models[i]) == 0)
            return true;
    }

    return false;
}

/*
 * Clean up priority-based scheduling
 */
void scheduler_priority_cleanup(void)
{
    int i;

    /* Clean up priority queues */
    for (i = 0; i < PRIORITY_LEVELS; i++) {
        strict_fib_heap_cleanup(&priority_queues[i]);
    }

    pr_info("Strict Fibonacci heap-based priority scheduling cleaned up\n");
}
/* FIFO Queue Management functions */
void fifo_init(struct fifo_queue *fifo)
{
    if (!fifo) {
        pr_err("fifo_init: Invalid fifo pointer\n");
        return;
    }

    fifo->head = 0;
    fifo->tail = 0;
    fifo->count = 0;
    spin_lock_init(&fifo->lock);

    memset(fifo->providers, 0, sizeof(fifo->providers));

    pr_debug("fifo_init: FIFO queue initialized\n");
}

void fifo_cleanup(struct fifo_queue *fifo)
{
    if (!fifo) {
        pr_err("fifo_cleanup: Invalid fifo pointer\n");
        return;
    }

    fifo->head = 0;
    fifo->tail = 0;
    fifo->count = 0;

    pr_debug("fifo_cleanup: FIFO queue cleaned up\n");
}
/*
 * Check if a provider is available (not rate limited)
 * Returns true if provider is available, false otherwise
 */
static bool is_provider_available(int provider, struct scheduler_state *state)
{
    bool available = false;
    unsigned long flags;

    if (!state) {
        pr_err("is_provider_available: Invalid state pointer\n");
        return false;
    }

    if (provider < 0 || provider >= PROVIDER_COUNT)
        return false;

    spin_lock_irqsave(&metrics_lock, flags);

    if (atomic_read(&state->metrics[provider].current_status) != 0) {
        if (state->metrics[provider].rate_limit_reset_time > 0) {
            ktime_t now = ktime_get();
            if (ktime_after(now, state->metrics[provider].rate_limit_reset_time)) {
                /* Reset limit has passed */
                state->metrics[provider].rate_limit_reset_time = 0;
                atomic_set(&state->metrics[provider].current_status, 1);
                available = true;
            }
        } else {
            available = true;
        }
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    return available;
}

/*
 * Handle provider override request
 * Returns the requested provider if available, otherwise -1
 */
static int handle_provider_override(struct llm_request *req, struct scheduler_state *state)
{
    int provider;

    if (!req || !state) {
        return -1;
    }

    /* Check if provider is explicitly specified and valid */
    provider = req->provider_override;
    if (provider >= 0 && provider < PROVIDER_COUNT) {
        /* Check if specified provider is available */
        if (is_provider_available(provider, state)) {
            return provider;
        }

        pr_debug("handle_provider_override: Requested provider %d is unavailable\n", provider);
    }

    return -1; /* No override or provider unavailable */
}

/*
 * Select provider based on request and scheduler state
 * Main entry point for provider selection
 */
int select_provider(struct llm_request *req, struct scheduler_state *state)
{
    int algorithm;
    int provider;

    if (!req || !state) {
        pr_err("select_provider: Invalid request or state pointer\n");
        return 0;
    }

    /* Check for provider override */
    provider = handle_provider_override(req, state);
    if (provider >= 0) {
        pr_debug("select_provider: Using provider override %d\n", provider);
        return provider;
    }

    /* Always use round-robin algorithm for this implementation */
    algorithm = SCHEDULER_ROUND_ROBIN;

    /* Select first available provider (simplified round-robin) */
    for (provider = 0; provider < PROVIDER_COUNT; provider++) {
        if (is_provider_available(provider, state)) {
            return provider;
        }
    }

    /* If no providers available, return the first one */
    return 0;
}

/*
 * Update provider metrics after a request
 * Safely update all metrics with proper locking
 */
void update_provider_metrics(int provider, int result, s64 latency_ms, int tokens)
{
    struct scheduler_state *state;
    unsigned long flags;
    bool should_adjust = false;

    /* Get state from current task */
    state = get_scheduler_state();

    if (!state || provider < 0 || provider >= PROVIDER_COUNT) {
        pr_err("update_provider_metrics: Invalid state or provider\n");
        return;
    }

    /* Update metrics with proper locking */
    spin_lock_irqsave(&metrics_lock, flags);

    atomic_inc(&state->metrics[provider].total_requests);

    if (result == 0) {
        /* Successful request */
        atomic_inc(&state->metrics[provider].successful_requests);
        atomic64_add(latency_ms, &state->metrics[provider].total_latency_ms);

        if (latency_ms < state->metrics[provider].min_latency_ms)
            state->metrics[provider].min_latency_ms = latency_ms;

        if (latency_ms > state->metrics[provider].max_latency_ms)
            state->metrics[provider].max_latency_ms = latency_ms;

        if (tokens > 0)
            atomic_add(tokens, &state->metrics[provider].total_tokens);
    } else {
        /* Failed request */
        atomic_inc(&state->metrics[provider].failed_requests);

        if (result == -ETIMEDOUT)
            atomic_inc(&state->metrics[provider].timeouts);
        else if (result == -LLM_ERR_RATE_LIMIT)
            atomic_inc(&state->metrics[provider].rate_limited);
    }

    /* Check if weights should be adjusted */
    if (state->auto_adjust &&
        (atomic_read(&state->metrics[provider].total_requests) % METRICS_ADJUST_INTERVAL == 0)) {
        should_adjust = true;
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    /* Auto-adjust weights if needed */
    if (should_adjust)
        adjust_scheduler_weights(state);
}

/*
 * Normalize weights to ensure they sum to 100% and meet minimum values
 */
static void normalize_weights(struct scheduler_state *state)
{
    int i;
    int total = 0;
    int remaining, deficit;
    int adjustable_count = 0;

    if (!state)
        return;

    /* First pass: apply minimum weight and calculate total */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (state->weights[i] < MIN_PROVIDER_WEIGHT)
            state->weights[i] = MIN_PROVIDER_WEIGHT;
        total += state->weights[i];
    }

    /* If total is already correct, nothing to do */
    if (total == WEIGHT_TOTAL_PERCENT)
        return;

    /* Count providers that can be adjusted */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        if (state->weights[i] > MIN_PROVIDER_WEIGHT)
            adjustable_count++;
    }

    if (adjustable_count == 0) {
        /* All providers at minimum, reset to equal weights */
        for (i = 0; i < PROVIDER_COUNT; i++) {
            state->weights[i] = WEIGHT_TOTAL_PERCENT / PROVIDER_COUNT;
        }

        /* Distribute remainder evenly */
        remaining = WEIGHT_TOTAL_PERCENT % PROVIDER_COUNT;
        for (i = 0; i < remaining; i++) {
            state->weights[i]++;
        }

        return;
    }

    /* Calculate deficit/surplus */
    deficit = WEIGHT_TOTAL_PERCENT - total;

    if (deficit < 0) {
        /* Total too high - reduce weights proportionally */
        int total_reducible = total - (MIN_PROVIDER_WEIGHT * PROVIDER_COUNT);
        for (i = 0; i < PROVIDER_COUNT; i++) {
            if (state->weights[i] > MIN_PROVIDER_WEIGHT) {
                int reducible = state->weights[i] - MIN_PROVIDER_WEIGHT;
                int reduction = (reducible * -deficit) / total_reducible;
                state->weights[i] -= reduction;
            }
        }
    } else if (deficit > 0) {
        /* Total too low - increase weights proportionally */
        for (i = 0; i < PROVIDER_COUNT; i++) {
            if (state->weights[i] > MIN_PROVIDER_WEIGHT) {
                state->weights[i] += deficit / adjustable_count;
                deficit -= deficit / adjustable_count;
                adjustable_count--;

                if (adjustable_count == 0 && deficit > 0) {
                    state->weights[i] += deficit;
                    break;
                }
            }
        }
    }

    /* Final check to ensure we're at exactly 100% */
    total = 0;
    for (i = 0; i < PROVIDER_COUNT; i++) {
        total += state->weights[i];
    }

    if (total != WEIGHT_TOTAL_PERCENT) {
        /* Adjust the last weight to make total exactly 100% */
        state->weights[PROVIDER_COUNT - 1] += (WEIGHT_TOTAL_PERCENT - total);
    }
}

/*
 * Adjust scheduler weights based on performance metrics
 */
void adjust_scheduler_weights(struct scheduler_state *state)
{
    int success_rates[PROVIDER_COUNT];
    int total_success_rate = 0;
    int i;
    unsigned long flags_metrics, flags_weights;

    if (!state) {
        pr_err("adjust_scheduler_weights: Invalid state pointer\n");
        return;
    }

    /* Calculate success rates with metrics lock */
    spin_lock_irqsave(&metrics_lock, flags_metrics);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        int total = atomic_read(&state->metrics[i].total_requests);
        int successful = atomic_read(&state->metrics[i].successful_requests);

        if (total > 0) {
            /* Base success rate is percentage of successful requests */
            success_rates[i] = (successful * 100) / total;

            /* Penalize for timeouts and rate limits */
            int timeouts = atomic_read(&state->metrics[i].timeouts);
            int rate_limits = atomic_read(&state->metrics[i].rate_limited);

            if (timeouts > 0 || rate_limits > 0) {
                int penalty = ((timeouts + rate_limits) * RATE_LIMIT_PENALTY) / total;
                success_rates[i] = max(success_rates[i] - penalty, MIN_PROVIDER_WEIGHT);
            }

            total_success_rate += success_rates[i];
        } else {
            /* No data yet, assign default weight */
            success_rates[i] = WEIGHT_TOTAL_PERCENT / PROVIDER_COUNT;
            total_success_rate += success_rates[i];
        }
    }

    spin_unlock_irqrestore(&metrics_lock, flags_metrics);

    /* Acquire weights lock for updating weights */
    spin_lock_irqsave(&weights_lock, flags_weights);

    /* If no requests processed yet, use default weights */
    if (total_success_rate == 0) {
        for (i = 0; i < PROVIDER_COUNT; i++) {
            state->weights[i] = WEIGHT_TOTAL_PERCENT / PROVIDER_COUNT;
        }

        /* Ensure weights are properly normalized */
        normalize_weights(state);
    } else {
        /* Adjust weights based on success rates */
        for (i = 0; i < PROVIDER_COUNT; i++) {
            state->weights[i] = (success_rates[i] * WEIGHT_TOTAL_PERCENT) /
                                max(total_success_rate, 1);
        }

        /* Ensure weights are properly normalized */
        normalize_weights(state);
    }

    spin_unlock_irqrestore(&weights_lock, flags_weights);

    pr_debug("Adjusted weights: OpenAI=%d%%, Anthropic=%d%%, Gemini=%d%%\n",
             state->weights[PROVIDER_OPENAI],
             state->weights[PROVIDER_ANTHROPIC],
             state->weights[PROVIDER_GOOGLE_GEMINI]);
}

/*
 * Reset all metrics
 */
void scheduler_reset_metrics(struct scheduler_state *state)
{
    int i;
    unsigned long flags;

    if (!state) {
        pr_err("scheduler_reset_metrics: Invalid state pointer\n");
        return;
    }

    spin_lock_irqsave(&metrics_lock, flags);

    for (i = 0; i < PROVIDER_COUNT; i++) {
        atomic_set(&state->metrics[i].total_requests, 0);
        atomic_set(&state->metrics[i].successful_requests, 0);
        atomic_set(&state->metrics[i].failed_requests, 0);
        atomic_set(&state->metrics[i].timeouts, 0);
        atomic_set(&state->metrics[i].rate_limited, 0);
        atomic64_set(&state->metrics[i].total_latency_ms, 0);
        state->metrics[i].min_latency_ms = ULONG_MAX;
        state->metrics[i].max_latency_ms = 0;
        atomic_set(&state->metrics[i].total_tokens, 0);
        atomic_set(&state->metrics[i].current_status, 1); /* Reset to available */
        state->metrics[i].rate_limit_reset_time = 0;
    }

    spin_unlock_irqrestore(&metrics_lock, flags);

    pr_info("LLM scheduler metrics reset\n");
}

/*
 * Handle rate limiting for a provider
 */
void handle_rate_limit(int provider, struct scheduler_state *state, unsigned long reset_ms)
{
    unsigned long flags;
    ktime_t reset_time;

    if (!state || provider < 0 || provider >= PROVIDER_COUNT) {
        pr_err("handle_rate_limit: Invalid state or provider\n");
        return;
    }

    /* Set provider as rate limited with proper locking */
    spin_lock_irqsave(&metrics_lock, flags);

    /* Set provider as rate limited */
    atomic_set(&state->metrics[provider].current_status, 0);

    /* Set reset time */
    reset_time = ktime_add_ms(ktime_get(), reset_ms);
    state->metrics[provider].rate_limit_reset_time = reset_time;

    spin_unlock_irqrestore(&metrics_lock, flags);

    pr_info("Provider %d rate limited, will reset in %lu ms\n", provider, reset_ms);
}

/*
 * Add a provider to the FIFO queue
 */
int fifo_add_provider(struct fifo_queue *fifo, int provider)
{
    unsigned long flags;
    int ret = 0;

    if (!fifo) {
        pr_err("fifo_add_provider: Invalid fifo pointer\n");
        return -EINVAL;
    }

    if (provider < 0 || provider >= PROVIDER_COUNT) {
        pr_err("fifo_add_provider: Invalid provider %d\n", provider);
        return -EINVAL;
    }

    spin_lock_irqsave(&fifo->lock, flags);

    if (fifo->count >= MAX_FIFO_QUEUE_SIZE) {
        ret = -ENOSPC;
    } else {
        fifo->providers[fifo->tail] = provider;
        fifo->tail = (fifo->tail + 1) % MAX_FIFO_QUEUE_SIZE;
        fifo->count++;
    }

    spin_unlock_irqrestore(&fifo->lock, flags);

    return ret;
}
EXPORT_SYMBOL(fifo_init);
EXPORT_SYMBOL(fifo_cleanup);
EXPORT_SYMBOL(scheduler_init);
EXPORT_SYMBOL(scheduler_submit_request);
EXPORT_SYMBOL(scheduler_get_next_request);
EXPORT_SYMBOL(get_default_model);
EXPORT_SYMBOL(is_model_supported);
EXPORT_SYMBOL(scheduler_priority_cleanup);
EXPORT_SYMBOL(select_provider);
EXPORT_SYMBOL(update_provider_metrics);
EXPORT_SYMBOL(scheduler_reset_metrics);
EXPORT_SYMBOL(handle_rate_limit);
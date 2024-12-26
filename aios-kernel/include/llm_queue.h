//
// Created by sina-mazaheri on 12/17/24.
//

#ifndef LLM_QUEUE_H
#define LLM_QUEUE_H

#include <linux/list.h>
#include <linux/wait.h>

#define MAX_QUEUE_SIZE 100

struct llm_request {
    char *prompt;
    size_t length;
    struct completion *done;
    int result;
    struct list_head list;
};

struct request_queue {
    struct list_head queue;
    wait_queue_head_t wait_queue;
    struct mutex lock;
    atomic_t size;
    struct task_struct *worker;
    bool should_stop;
};

int request_queue_init(struct request_queue *queue);
int request_queue_submit(struct request_queue *queue, const char *prompt, size_t length);
void request_queue_cleanup(struct request_queue *queue);

#endif /* LLM_QUEUE_H */

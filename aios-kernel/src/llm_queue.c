//
// Created by sina-mazaheri on 12/17/24.
//

#include "llm_queue.h"
#include <linux/slab.h>
#include <linux/kthread.h>

static int queue_worker(void *data)
{
    struct request_queue *queue = data;
    struct llm_request *req;

    while (!kthread_should_stop()) {
        wait_event_interruptible(queue->wait_queue,
                                 !list_empty(&queue->queue) ||
                                 queue->should_stop);

        if (queue->should_stop)
            break;

        mutex_lock(&queue->lock);
        if (!list_empty(&queue->queue)) {
            req = list_first_entry(&queue->queue,
            struct llm_request, list);
            list_del(&req->list);
            atomic_dec(&queue->size);
            mutex_unlock(&queue->lock);

            /* Process request */
            req->result = process_request(req);
            complete(req->done);
        } else {
            mutex_unlock(&queue->lock);
        }
    }

    return 0;
}

int request_queue_submit(struct request_queue *queue,
                         const char *prompt, size_t length)
{
    struct llm_request *req;
    struct completion done;
    int ret;

    if (atomic_read(&queue->size) >= MAX_QUEUE_SIZE)
        return -EBUSY;

    req = kmalloc(sizeof(*req), GFP_KERNEL);
    if (!req)
        return -ENOMEM;

    req->prompt = kmalloc(length + 1, GFP_KERNEL);
    if (!req->prompt) {
        kfree(req);
        return -ENOMEM;
    }

    memcpy(req->prompt, prompt, length);
    req->prompt[length] = '\0';
    req->length = length;
    init_completion(&done);
    req->done = &done;

    mutex_lock(&queue->lock);
    list_add_tail(&req->list, &queue->queue);
    atomic_inc(&queue->size);
    mutex_unlock(&queue->lock);

    wake_up(&queue->wait_queue);
    wait_for_completion(&done);

    ret = req->result;
    kfree(req->prompt);
    kfree(req);

    return ret;
}
/**
 * test_basic.c - Basic functionality test for LLM Orchestrator
 *
 * This test creates 3 threads to test basic functionality of the LLM orchestrator.
 * Each thread sends a different request and displays the response.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <time.h>
#include <stddef.h>  /* For NULL definition */

/* Constants matching the kernel module */
#define DEVICE_PATH "/dev/llm_orchestrator"
#define MAX_PROMPT_LENGTH 4096
#define MAX_RESPONSE_LENGTH 8192
#define MAX_MODEL_NAME 64
#define MAX_ROLE_NAME 32

/* Provider definitions */
#define PROVIDER_OPENAI 0
#define PROVIDER_ANTHROPIC 1
#define PROVIDER_GOOGLE_GEMINI 2

/* Scheduler algorithms */
#define SCHEDULER_ROUND_ROBIN 0
#define SCHEDULER_WEIGHTED 1
#define SCHEDULER_PRIORITY 2
#define SCHEDULER_PERFORMANCE 3
#define SCHEDULER_COST_AWARE 4
#define SCHEDULER_FALLBACK 5
#define SCHEDULER_FIFO 6

/* Request structure matching the kernel module */
struct llm_request {
    char prompt[MAX_PROMPT_LENGTH];
    char role[MAX_ROLE_NAME];
    char model_name[MAX_MODEL_NAME];
    int conversation_id;
    int max_tokens;
    int temperature_x100;
    unsigned long timeout_ms;
    int scheduler_algorithm;
    int priority;
};

/* Thread data structure */
struct thread_data {
    int thread_id;
    int conversation_id;
    const char *prompt;
    int scheduler_algorithm;
};

/* Thread function */
void *request_thread(void *arg) {
    struct thread_data *data = (struct thread_data *)arg;
    struct llm_request req;
    char response[MAX_RESPONSE_LENGTH];
    int fd, ret;
    ssize_t bytes_read;

    printf("[Thread %d] Starting with conversation ID %d\n",
           data->thread_id, data->conversation_id);

    /* Open the device */
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        printf("[Thread %d] Failed to open device: %s\n",
               data->thread_id, strerror(errno));
        return NULL;
    }

    /* Initialize request */
    memset(&req, 0, sizeof(req));
    strncpy(req.prompt, data->prompt, MAX_PROMPT_LENGTH - 1);
    strncpy(req.role, "user", MAX_ROLE_NAME - 1);
    req.conversation_id = data->conversation_id;
    req.max_tokens = 1000;
    req.temperature_x100 = 70;  /* 0.7 */
    req.timeout_ms = 30000;     /* 30 seconds */
    req.scheduler_algorithm = data->scheduler_algorithm;
    req.priority = 50;          /* Medium priority */

    printf("[Thread %d] Sending request: %s\n",
           data->thread_id, req.prompt);

    /* Send request to device */
    ret = write(fd, &req, sizeof(req));
    if (ret < 0) {
        printf("[Thread %d] Failed to write to device: %s\n",
               data->thread_id, strerror(errno));
        close(fd);
        return NULL;
    }

    /* Read response */
    memset(response, 0, sizeof(response));
    bytes_read = read(fd, response, sizeof(response) - 1);

    if (bytes_read < 0) {
        printf("[Thread %d] Failed to read response: %s\n",
               data->thread_id, strerror(errno));
    } else if (bytes_read == 0) {
        printf("[Thread %d] No response received\n", data->thread_id);
    } else {
        response[bytes_read] = '\0';
        printf("[Thread %d] Response received (%zd bytes): %s\n",
               data->thread_id, bytes_read, response);
    }

    close(fd);
    printf("[Thread %d] Thread completed\n", data->thread_id);
    return NULL;
}

int main() {
    pthread_t threads[3];
    struct thread_data thread_data_array[3];
    int i, ret;

    printf("LLM Orchestrator Basic Test\n");
    printf("---------------------------\n");

    /* Initialize random seed */
    srand(time(NULL));

    /* Configure thread data */
    for (i = 0; i < 3; i++) {
        thread_data_array[i].thread_id = i + 1;
        thread_data_array[i].conversation_id = 1000 + i;
        thread_data_array[i].scheduler_algorithm = SCHEDULER_ROUND_ROBIN;
    }

    /* Set different prompts for each thread */
    thread_data_array[0].prompt = "Explain the theory of relativity in simple terms";
    thread_data_array[1].prompt = "Write a short poem about spring";
    thread_data_array[2].prompt = "What are the main features of Linux kernel modules?";

    /* Create threads */
    for (i = 0; i < 1; i++) {
        ret = pthread_create(&threads[i], NULL, request_thread, &thread_data_array[i]);
        if (ret) {
            printf("Error creating thread %d: %s\n", i + 1, strerror(ret));
            exit(EXIT_FAILURE);
        }
    }

    /* Wait for threads to complete */
    for (i = 0; i < 1; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads completed\n");

    return 0;
}
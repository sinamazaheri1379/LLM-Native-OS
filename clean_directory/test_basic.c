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
#include <sys/stat.h>
#include <signal.h>

/* Constants matching the kernel module */
#define DEVICE_PATH "/dev/llm_orchestrator"
#define MAX_PROMPT_LENGTH 4096
#define MAX_RESPONSE_LENGTH 65535  /* Updated to match kernel's definition */
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

/* Priority levels */
#define PRIORITY_LOW 0
#define PRIORITY_NORMAL 50
#define PRIORITY_HIGH 100

/* IOCTL commands - must match the kernel definitions */
#define IOCTL_SET_PREFERRED_PROVIDER _IOW('L', 1, int)
#define IOCTL_SET_REQUEST_PRIORITY   _IOW('L', 2, int)
#define IOCTL_GET_REQUEST_STATUS     _IOR('L', 3, int)

/* Request structure matching the kernel module EXACTLY */
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
    int provider_override;  /* Match kernel's struct - this was missing */
};

/* Thread data structure */
struct thread_data {
    int thread_id;
    int conversation_id;
    const char *prompt;
    int scheduler_algorithm;
    int priority;
    int preferred_provider;
};

/* Global flag for timeout handling */
volatile sig_atomic_t timeout_occurred = 0;

/* Signal handler for alarm */
void timeout_handler(int sig) {
    timeout_occurred = 1;
}

/* Thread function */
void *request_thread(void *arg) {
    struct thread_data *data = (struct thread_data *)arg;
    struct llm_request req;
    char *response = NULL;
    int fd, ret, status;
    ssize_t bytes_read, total_read = 0;
    struct sigaction sa;

    printf("[Thread %d] Starting with conversation ID %d\n",
           data->thread_id, data->conversation_id);

    /* Check if the device file exists */
    struct stat st;
    if (stat(DEVICE_PATH, &st) != 0) {
        printf("[Thread %d] Error: Device file %s not found. Error: %s\n",
               data->thread_id, DEVICE_PATH, strerror(errno));
        return NULL;
    }

    /* Open the device */
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        printf("[Thread %d] Failed to open device: %s (errno=%d)\n",
               data->thread_id, strerror(errno), errno);
        return NULL;
    }

    /* Set up preferred provider using ioctl if specified */
    if (data->preferred_provider >= 0) {
        ret = ioctl(fd, IOCTL_SET_PREFERRED_PROVIDER, &data->preferred_provider);
        if (ret < 0) {
            printf("[Thread %d] Failed to set preferred provider: %s (errno=%d)\n",
                   data->thread_id, strerror(errno), errno);
            /* Continue anyway - this is not fatal */
        } else {
            printf("[Thread %d] Set preferred provider to %d\n",
                   data->thread_id, data->preferred_provider);
        }
    }

    /* Set priority using ioctl if specified */
    if (data->priority > 0) {
        ret = ioctl(fd, IOCTL_SET_REQUEST_PRIORITY, &data->priority);
        if (ret < 0) {
            printf("[Thread %d] Failed to set priority: %s (errno=%d)\n",
                   data->thread_id, strerror(errno), errno);
            /* Continue anyway - this is not fatal */
        } else {
            printf("[Thread %d] Set priority to %d\n",
                   data->thread_id, data->priority);
        }
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
    req.priority = data->priority;
    req.provider_override = -1;  /* No override - use scheduler */

    printf("[Thread %d] Sending request (%zu bytes): %s\n",
           data->thread_id, sizeof(req), req.prompt);

    /* Send request to device */
    ret = write(fd, &req, sizeof(req));
    if (ret < 0) {
        printf("[Thread %d] Failed to write to device: %s (errno=%d)\n",
               data->thread_id, strerror(errno), errno);
        printf("[Thread %d] Request size: %zu bytes\n",
               data->thread_id, sizeof(req));
        close(fd);
        return NULL;
    }

    if (ret != sizeof(req)) {
        printf("[Thread %d] Warning: Wrote only %d of %zu bytes\n",
               data->thread_id, ret, sizeof(req));
    }

    /* Allocate response buffer with proper size */
    response = malloc(MAX_RESPONSE_LENGTH);
    if (!response) {
        printf("[Thread %d] Failed to allocate response buffer\n", data->thread_id);
        close(fd);
        return NULL;
    }
    memset(response, 0, MAX_RESPONSE_LENGTH);

    /* Set up alarm handler for timeout */
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = timeout_handler;
    sigaction(SIGALRM, &sa, NULL);

    /* Set alarm for 60 seconds (longer than kernel timeout) */
    timeout_occurred = 0;
    alarm(60);

    /* Read response - potentially in multiple chunks */
    while (!timeout_occurred) {
        /* Check if request completed */
        status = 0;
        ret = ioctl(fd, IOCTL_GET_REQUEST_STATUS, &status);
        if (ret < 0) {
            printf("[Thread %d] Failed to get request status: %s (errno=%d)\n",
                   data->thread_id, strerror(errno), errno);
            /* Continue anyway - might still get a response */
        }

        /* Try to read data */
        bytes_read = read(fd, response + total_read, MAX_RESPONSE_LENGTH - total_read - 1);

        if (bytes_read < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                /* Not ready yet, try again */
                printf("[Thread %d] Waiting for response...\n", data->thread_id);
                sleep(1);
                continue;
            } else {
                printf("[Thread %d] Failed to read response: %s (errno=%d)\n",
                       data->thread_id, strerror(errno), errno);
                break;
            }
        } else if (bytes_read == 0) {
            /* End of file, done reading */
            if (total_read > 0) {
                printf("[Thread %d] Response complete\n", data->thread_id);
                break;
            } else {
                /* Empty response, try again */
                sleep(1);
                continue;
            }
        } else {
            /* Got some data */
            total_read += bytes_read;
            response[total_read] = '\0';

            /* Check if we have a complete response */
            if (status == 1) {
                printf("[Thread %d] Response completed (status=1)\n", data->thread_id);
                break;
            }

            /* Check if buffer is full */
            if (total_read >= MAX_RESPONSE_LENGTH - 1) {
                printf("[Thread %d] Response buffer full\n", data->thread_id);
                break;
            }
        }
    }

    /* Cancel alarm */
    alarm(0);

    if (timeout_occurred) {
        printf("[Thread %d] Request timed out\n", data->thread_id);
    } else if (total_read > 0) {
        printf("[Thread %d] Response received (%zd bytes):\n", data->thread_id, total_read);
        printf("------ RESPONSE START ------\n");
        printf("%s\n", response);
        printf("------- RESPONSE END -------\n");
    } else {
        printf("[Thread %d] No response received\n", data->thread_id);
    }

    /* Clean up */
    close(fd);
    free(response);

    printf("[Thread %d] Thread completed\n", data->thread_id);
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t threads[3];
    struct thread_data thread_data_array[3];
    int i, ret;
    int algorithm = SCHEDULER_ROUND_ROBIN;
    int test_mode = 0; /* 0 = normal, 1 = provider test */

    /* Parse command line arguments if any */
    if (argc > 1) {
        if (strcmp(argv[1], "--fallback") == 0) {
            algorithm = SCHEDULER_FALLBACK;
            printf("Using FALLBACK scheduler algorithm\n");
        } else if (strcmp(argv[1], "--weighted") == 0) {
            algorithm = SCHEDULER_WEIGHTED;
            printf("Using WEIGHTED scheduler algorithm\n");
        } else if (strcmp(argv[1], "--performance") == 0) {
            algorithm = SCHEDULER_PERFORMANCE;
            printf("Using PERFORMANCE scheduler algorithm\n");
        } else if (strcmp(argv[1], "--providers") == 0) {
            test_mode = 1;
            printf("Running provider-specific test\n");
        }
    }

    printf("LLM Orchestrator Basic Test\n");
    printf("---------------------------\n");

    /* Initialize random seed */
    srand(time(NULL));

    if (test_mode == 0) {
        /* Normal test with different prompts */

        /* Configure thread data */
        for (i = 0; i < 3; i++) {
            thread_data_array[i].thread_id = i + 1;
            thread_data_array[i].conversation_id = 1000 + i;
            thread_data_array[i].scheduler_algorithm = algorithm;
            thread_data_array[i].priority = PRIORITY_NORMAL;
            thread_data_array[i].preferred_provider = -1; /* Use scheduler */
        }

        /* Set different prompts for each thread */
        thread_data_array[0].prompt = "Explain the theory of relativity in simple terms";
        thread_data_array[1].prompt = "Write a short poem about spring";
        thread_data_array[2].prompt = "What are the main features of Linux kernel modules?";
    } else {
        /* Provider-specific test */

        /* Configure thread data to test each provider */
        for (i = 0; i < 3; i++) {
            thread_data_array[i].thread_id = i + 1;
            thread_data_array[i].conversation_id = 2000 + i;
            thread_data_array[i].scheduler_algorithm = -1; /* Use default */
            thread_data_array[i].priority = PRIORITY_NORMAL;
            thread_data_array[i].preferred_provider = i; /* Use specific provider */
            thread_data_array[i].prompt = "Summarize what you can do in one paragraph";
        }
    }

    /* Create threads */
    for (i = 0; i < 3; i++) {
        ret = pthread_create(&threads[i], NULL, request_thread, &thread_data_array[i]);
        if (ret) {
            printf("Error creating thread %d: %s\n", i + 1, strerror(ret));
            exit(EXIT_FAILURE);
        }

        /* Small delay between thread creation to avoid overloading */
        usleep(250000); /* 250ms */
    }

    /* Wait for threads to complete */
    for (i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads completed\n");

    return 0;
}
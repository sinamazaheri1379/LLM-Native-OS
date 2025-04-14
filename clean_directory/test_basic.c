/**
 * test_basic.c - Multi-turn threaded test for LLM Orchestrator
 *
 * Each thread maintains its own conversation context and sends multiple
 * prompts sequentially, creating a true multi-turn, multi-threaded test.
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
#define MAX_RESPONSE_LENGTH 65535
#define MAX_MODEL_NAME 64
#define MAX_ROLE_NAME 32
#define MAX_PROMPTS_PER_THREAD 10

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
#define PRIORITY_HIGH 0
#define PRIORITY_NORMAL 1
#define PRIORITY_LOW 2

/* IOCTL commands */
#define IOCTL_SET_PREFERRED_PROVIDER _IOW('L', 1, int)
#define IOCTL_SET_REQUEST_PRIORITY   _IOW('L', 2, int)
#define IOCTL_GET_REQUEST_STATUS     _IOR('L', 3, int)

/* Request structure matching the kernel module */
struct llm_request {
    char prompt[MAX_PROMPT_LENGTH];
    char role[MAX_ROLE_NAME];
    char model_name[MAX_MODEL_NAME];
    int conversation_id;
    int request_id;       // Add this field
    int max_tokens;
    int temperature_x100;
    unsigned long timeout_ms;
    int scheduler_algorithm;
    int priority;
    int provider_override;
};

/* Enhanced thread data structure for multi-turn conversations */
struct thread_data {
    int thread_id;
    int conversation_id;
    int num_prompts;                           /* Number of prompts this thread will process */
    const char *prompts[MAX_PROMPTS_PER_THREAD]; /* Array of prompts for this thread */
    char *responses[MAX_PROMPTS_PER_THREAD];   /* Array to store responses */
    int scheduler_algorithm;
    int priority;
    int preferred_provider;
    pthread_mutex_t mutex;                     /* Mutex for thread-specific data */
};

/* Global flag for timeout handling */
volatile sig_atomic_t timeout_occurred = 0;

/* Signal handler for alarm */
void timeout_handler(int sig) {
    timeout_occurred = 1;
}

/**
 * Function to send a request and get a response
 * Returns 0 on success, negative on error
 */
int send_request_and_get_response(int fd, const char *prompt, int conversation_id,
                                  int scheduler_algorithm, int priority,
                                  int provider_override, char *response_buffer,
                                  size_t buffer_size) {
    struct llm_request req;
    struct sigaction sa;
    int ret, status;
    ssize_t bytes_read, total_read = 0;

    /* Clear the timeout flag */
    timeout_occurred = 0;

    /* Initialize request */
    memset(&req, 0, sizeof(req));
    strncpy(req.prompt, prompt, MAX_PROMPT_LENGTH - 1);
    strncpy(req.role, "user", MAX_ROLE_NAME - 1);
    req.conversation_id = conversation_id;
    req.request_id = 0;
    req.max_tokens = 1000;
    req.temperature_x100 = 70;  /* 0.7 */
    req.timeout_ms = 30000;     /* 30 seconds */
    req.scheduler_algorithm = scheduler_algorithm;
    req.priority = priority;
    req.provider_override = provider_override;

    /* Setup timeout handler */
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = timeout_handler;
    sigaction(SIGALRM, &sa, NULL);

    /* Clear response buffer */
    if (response_buffer && buffer_size > 0) {
        memset(response_buffer, 0, buffer_size);
    }

    /* Send request to device */
    ret = write(fd, &req, sizeof(req));
    if (ret < 0) {
        fprintf(stderr, "Failed to write to device: %s (errno=%d)\n",
                strerror(errno), errno);
        return -errno;
    }

    if (ret != sizeof(req)) {
        fprintf(stderr, "Warning: Wrote only %d of %zu bytes\n",
                ret, sizeof(req));
    }

    /* Set alarm for 60 seconds */
    alarm(60);

    /* Read response - potentially in multiple chunks */
    while (!timeout_occurred) {
        /* Check if request completed */
        status = 0;
        ret = ioctl(fd, IOCTL_GET_REQUEST_STATUS, &status);

        /* Try to read data */
        bytes_read = read(fd, response_buffer + total_read,
                         buffer_size - total_read - 1);

        if (bytes_read < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                /* Not ready yet, try again */
                sleep(1);
                continue;
            } else {
                fprintf(stderr, "Failed to read response: %s (errno=%d)\n",
                       strerror(errno), errno);
                alarm(0); /* Cancel alarm */
                return -errno;
            }
        } else if (bytes_read == 0) {
            /* End of file, done reading */
            if (total_read > 0) {
                break;
            } else {
                /* Empty response, try again */
                sleep(1);
                continue;
            }
        } else {
            /* Got some data */
            total_read += bytes_read;
            response_buffer[total_read] = '\0';

            /* Check if we have a complete response */
            if (status == 1) {
                break;
            }

            /* Check if buffer is full */
            if (total_read >= buffer_size - 1) {
                break;
            }
        }
    }

    /* Cancel alarm */
    alarm(0);

    if (timeout_occurred) {
        fprintf(stderr, "Request timed out\n");
        return -ETIMEDOUT;
    }

    return total_read > 0 ? total_read : -EIO;
}

/* Thread function for multi-turn conversation */
void *multi_turn_thread(void *arg) {
    struct thread_data *data = (struct thread_data *)arg;
    int fd, turn, result;
    char turn_buffer[32]; /* For logging turns */

    printf("[Thread %d] Starting with conversation ID %d (%d turns)\n",
           data->thread_id, data->conversation_id, data->num_prompts);

    /* Check if the device file exists */
    struct stat st;
    if (stat(DEVICE_PATH, &st) != 0) {
        printf("[Thread %d] Error: Device file %s not found. Error: %s\n",
               data->thread_id, DEVICE_PATH, strerror(errno));
        return NULL;
    }

    /* Open the device - keep open for entire conversation */
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        printf("[Thread %d] Failed to open device: %s (errno=%d)\n",
               data->thread_id, strerror(errno), errno);
        return NULL;
    }

    /* Set up preferred provider using ioctl if specified */
    if (data->preferred_provider >= 0) {
        int ret = ioctl(fd, IOCTL_SET_PREFERRED_PROVIDER, &data->preferred_provider);
        if (ret < 0) {
            printf("[Thread %d] Failed to set preferred provider: %s (errno=%d)\n",
                   data->thread_id, strerror(errno), errno);
        } else {
            printf("[Thread %d] Set preferred provider to %d\n",
                   data->thread_id, data->preferred_provider);
        }
    }

    /* Set priority using ioctl if specified */
    if (data->priority >= 0) {
        int ret = ioctl(fd, IOCTL_SET_REQUEST_PRIORITY, &data->priority);
        if (ret < 0) {
            printf("[Thread %d] Failed to set priority: %s (errno=%d)\n",
                   data->thread_id, strerror(errno), errno);
        } else {
            printf("[Thread %d] Set priority to %d\n",
                   data->thread_id, data->priority);
        }
    }

    /* Process each prompt in sequence */
    for (turn = 0; turn < data->num_prompts; turn++) {
        /* Format a unique identifier for this turn */
        snprintf(turn_buffer, sizeof(turn_buffer), "T%d.%d", data->thread_id, turn+1);

        /* Allocate response buffer */
        data->responses[turn] = malloc(MAX_RESPONSE_LENGTH);
        if (!data->responses[turn]) {
            printf("[%s] Failed to allocate response buffer\n", turn_buffer);
            break;
        }

        /* Send the request and get response */
        printf("\n[%s] SENDING: %s\n", turn_buffer, data->prompts[turn]);

        result = send_request_and_get_response(
            fd,
            data->prompts[turn],
            data->conversation_id,
            data->scheduler_algorithm,
            data->priority,
            data->preferred_provider,
            data->responses[turn],
            MAX_RESPONSE_LENGTH
        );

        if (result < 0) {
            printf("[%s] Error: Failed to get response (error=%d)\n",
                   turn_buffer, result);
            /* Continue to next prompt despite error */
        } else {
            printf("[%s] RECEIVED (%d bytes):\n%s\n",
                   turn_buffer, result, data->responses[turn]);
        }

        /* Small delay between turns to avoid overwhelming the system */
        if (turn < data->num_prompts - 1) {
            usleep(500000); /* 500ms between turns */
        }
    }

    /* Close the device */
    close(fd);

    printf("[Thread %d] Completed all %d turns\n", data->thread_id, data->num_prompts);
    return NULL;
}

/* Function to run the multi-turn threaded test */
void run_multi_turn_test(int scheduler_algorithm) {
    pthread_t threads[3];
    struct thread_data thread_data_array[3];
    int i, ret;

    printf("\nStarting Multi-Turn Threaded Test with scheduler algorithm %d\n",
           scheduler_algorithm);
    printf("------------------------------------------------------------\n");

    /* Thread 1 prompts - science/physics conversation */
    const char *thread1_prompts[] = {
        "What is quantum entanglement?",
        "How does it relate to quantum computing?",
        "What are the main challenges in building quantum computers?"
    };

    /* Thread 2 prompts - literature conversation */
    const char *thread2_prompts[] = {
        "Who wrote 'Pride and Prejudice'?",
        "What are some themes in that novel?",
        "How does it compare to her other works?"
    };

    /* Thread 3 prompts - technology conversation */
    const char *thread3_prompts[] = {
        "What is a Linux kernel module?",
        "How do you write a basic kernel module?",
        "What are some security concerns with kernel modules?"
    };

    /* Setup thread data */
    for (i = 0; i < 3; i++) {
        memset(&thread_data_array[i], 0, sizeof(struct thread_data));
        thread_data_array[i].thread_id = i + 1;
        thread_data_array[i].conversation_id = 5000 + i;
        thread_data_array[i].scheduler_algorithm = scheduler_algorithm;
        thread_data_array[i].priority = PRIORITY_NORMAL;
        thread_data_array[i].preferred_provider = -1; /* Use scheduler */
        thread_data_array[i].num_prompts = 3;

        /* Initialize mutex */
        pthread_mutex_init(&thread_data_array[i].mutex, NULL);
    }

    /* Copy prompts for each thread */
    for (i = 0; i < thread_data_array[0].num_prompts; i++) {
        thread_data_array[0].prompts[i] = thread1_prompts[i];
        thread_data_array[1].prompts[i] = thread2_prompts[i];
        thread_data_array[2].prompts[i] = thread3_prompts[i];
    }

    /* Create threads */
    for (i = 0; i < 3; i++) {
        ret = pthread_create(&threads[i], NULL, multi_turn_thread, &thread_data_array[i]);
        if (ret) {
            fprintf(stderr, "Error creating thread %d: %s\n", i + 1, strerror(ret));
            exit(EXIT_FAILURE);
        }

        /* Small delay between thread creation to avoid overloading */
        usleep(250000); /* 250ms */
    }

    /* Wait for threads to complete */
    for (i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Free allocated memory */
    for (i = 0; i < 3; i++) {
        for (int j = 0; j < thread_data_array[i].num_prompts; j++) {
            if (thread_data_array[i].responses[j]) {
                free(thread_data_array[i].responses[j]);
            }
        }
        pthread_mutex_destroy(&thread_data_array[i].mutex);
    }

    printf("\nMulti-turn threaded test completed\n");
}

/* Print usage instructions */
void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  --multi-turn-rr     Run multi-turn test with Round Robin scheduler\n");
    printf("  --multi-turn-fifo   Run multi-turn test with FIFO scheduler\n");
    printf("  --multi-turn-prio   Run multi-turn test with Priority scheduler\n");
    printf("  --multi-turn-weighted Run multi-turn test with Weighted scheduler\n");
    printf("  --multi-turn-perf   Run multi-turn test with Performance scheduler\n");
    printf("  --help              Display this help message\n");
}

int main(int argc, char *argv[]) {
    int algorithm = SCHEDULER_ROUND_ROBIN;

    printf("LLM Orchestrator Multi-Turn Threaded Test v1.0\n");
    printf("---------------------------------------------\n");

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Parse command line arguments */
    if (strcmp(argv[1], "--multi-turn-rr") == 0) {
        run_multi_turn_test(SCHEDULER_ROUND_ROBIN);
    } else if (strcmp(argv[1], "--multi-turn-fifo") == 0) {
        run_multi_turn_test(SCHEDULER_FIFO);
    } else if (strcmp(argv[1], "--multi-turn-prio") == 0) {
        run_multi_turn_test(SCHEDULER_PRIORITY);
    } else if (strcmp(argv[1], "--multi-turn-weighted") == 0) {
        run_multi_turn_test(SCHEDULER_WEIGHTED);
    } else if (strcmp(argv[1], "--multi-turn-perf") == 0) {
        run_multi_turn_test(SCHEDULER_PERFORMANCE);
    } else if (strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
    } else {
        printf("Unknown option: %s\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
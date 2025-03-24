/**
 * test_stress.c - Stress test for LLM Orchestrator
 *
 * This test creates multiple threads that stress-test the LLM orchestrator
 * with edge cases, error conditions, large prompts, and rapid requests.
 * It runs for a configurable duration and collects detailed statistics.
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
#include <signal.h>
#include <limits.h>

/* Constants matching the kernel module */
#define DEVICE_PATH "/dev/llm_orchestrator"
#define MAX_PROMPT_LENGTH 4096
#define MAX_RESPONSE_LENGTH 8192
#define MAX_MODEL_NAME 64
#define MAX_ROLE_NAME 32
#define NUM_THREADS 20
#define TEST_DURATION_SEC 60

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

/* Test modes */
#define TEST_MODE_NORMAL 0
#define TEST_MODE_EDGE_CASE 1
#define TEST_MODE_ERROR_CASE 2
#define TEST_MODE_LARGE_PROMPT 3
#define TEST_MODE_SHORT_TIMEOUT 4
#define TEST_MODE_INVALID_MODEL 5

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

/* Thread statistics */
struct thread_stats {
    int requests_sent;
    int successful_responses;
    int failed_responses;
    int errors_by_type[6]; /* Count of each test mode error */
    long total_response_time_ms;
};

/* Global statistics */
struct global_stats {
    int total_requests;
    int successful_responses;
    int failed_responses;
    int errors_by_type[6];
    long total_response_time_ms;
    pthread_mutex_t mutex;
};

/* Thread data structure */
struct thread_data {
    int thread_id;
    int seed;
    struct global_stats *stats;
    volatile sig_atomic_t *keep_running;
};

/* Global variables */
volatile sig_atomic_t keep_running = 1;
struct global_stats g_stats = {0};

/* Array of test prompts */
const char *test_prompts[] = {
    "Explain the concept of memory management in operating systems",
    "Write a recursive algorithm to solve the Tower of Hanoi problem",
    "What are the key differences between IPv4 and IPv6?",
    "Explain the CAP theorem in distributed systems",
    "How does garbage collection work in modern programming languages?",
    "What are the principles of zero-trust security architecture?",
    "Describe the differences between ACID and BASE in database systems",
    "How do load balancers distribute network traffic efficiently?",
    "Explain the concept of idempotency in RESTful APIs",
    "What are the main challenges in implementing microservices?"
};
#define NUM_TEST_PROMPTS (sizeof(test_prompts) / sizeof(test_prompts[0]))

/* Function to get time in milliseconds */
long get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000) + (ts.tv_nsec / 1000000);
}

/* Signal handler for clean termination */
void signal_handler(int sig) {
    printf("\nReceived signal %d, stopping test...\n", sig);
    keep_running = 0;
}

/* Generate a prompt based on test mode */
void generate_prompt(char *prompt_buffer, size_t buffer_size, int test_mode, int thread_id) {
    switch (test_mode) {
        case TEST_MODE_NORMAL:
            /* Normal case - use a predefined prompt */
            snprintf(prompt_buffer, buffer_size, "%s",
                    test_prompts[rand() % NUM_TEST_PROMPTS]);
            break;

        case TEST_MODE_EDGE_CASE:
            /* Edge case - very specific or unusual query */
            snprintf(prompt_buffer, buffer_size,
                    "If a quantum computer with %d qubits processes the Shor's algorithm for "
                    "factoring a %d-bit RSA key, how many operations would be required and what "
                    "would be the expected runtime compared to classical computers?",
                    50 + (rand() % 150), 1024 + (rand() % 3072));
            break;

        case TEST_MODE_ERROR_CASE:
            /* Error case - malformed prompt with special characters */
            snprintf(prompt_buffer, buffer_size,
                    "Test prompt with special chars: \x01\x02\x03\x04\x05"
                    "Thread %d attempting to trigger error handling",
                    thread_id);
            break;

        case TEST_MODE_LARGE_PROMPT:
            /* Large prompt - approach max size limit */
            {
                char *base_text = "This is a very large prompt that repeats to test buffer handling capabilities. ";
                int base_len = strlen(base_text);
                int i, repetitions = (MAX_PROMPT_LENGTH - 100) / base_len;

                /* Start with a basic intro */
                snprintf(prompt_buffer, buffer_size,
                        "Thread %d testing large prompt handling. ", thread_id);

                /* Add repetitions of the base text to approach the limit */
                for (i = 0; i < repetitions && strlen(prompt_buffer) + base_len < buffer_size - 1; i++) {
                    strncat(prompt_buffer, base_text, buffer_size - strlen(prompt_buffer) - 1);
                }

                /* Add a final question */
                strncat(prompt_buffer, "Can you summarize the main points about large language models?",
                        buffer_size - strlen(prompt_buffer) - 1);
            }
            break;

        case TEST_MODE_SHORT_TIMEOUT:
            /* Normal prompt but will use very short timeout */
            snprintf(prompt_buffer, buffer_size,
                    "This request has a very short timeout. Please explain the concept of deadlocks "
                    "in operating systems and how to prevent them.");
            break;

        case TEST_MODE_INVALID_MODEL:
            /* Normal prompt but will use invalid model name */
            snprintf(prompt_buffer, buffer_size,
                    "This request specifies an invalid model. Please explain how virtual memory "
                    "works in modern operating systems.");
            break;

        default:
            /* Fallback case */
            snprintf(prompt_buffer, buffer_size,
                    "Generic test prompt from thread %d, test mode %d",
                    thread_id, test_mode);
    }
}

/* Configure request parameters based on test mode */
void configure_request(struct llm_request *req, int test_mode, int thread_id, int conversation_id) {
    /* Set common parameters */
    req->conversation_id = conversation_id;
    strncpy(req->role, "user", MAX_ROLE_NAME - 1);

    /* Configure specific parameters based on test mode */
    switch (test_mode) {
        case TEST_MODE_NORMAL:
            req->max_tokens = 1000;
            req->temperature_x100 = 70; /* 0.7 */
            req->timeout_ms = 30000;    /* 30 seconds */
            req->scheduler_algorithm = rand() % 7;
            req->priority = 50;
            req->model_name[0] = '\0';  /* Use default model */
            break;

        case TEST_MODE_EDGE_CASE:
            req->max_tokens = 2000;     /* Request more tokens */
            req->temperature_x100 = 90; /* Higher temperature 0.9 */
            req->timeout_ms = 45000;    /* 45 seconds */
            req->scheduler_algorithm = SCHEDULER_PERFORMANCE;
            req->priority = 75;         /* Higher priority */
            req->model_name[0] = '\0';  /* Use default model */
            break;

        case TEST_MODE_ERROR_CASE:
            req->max_tokens = -10;      /* Invalid token count */
            req->temperature_x100 = 500; /* Invalid temperature */
            req->timeout_ms = 30000;
            req->scheduler_algorithm = -1; /* Use default algorithm */
            req->priority = 0;
            req->model_name[0] = '\0';
            break;

        case TEST_MODE_LARGE_PROMPT:
            req->max_tokens = 1500;
            req->temperature_x100 = 70;
            req->timeout_ms = 60000;    /* Longer timeout for large prompt */
            req->scheduler_algorithm = SCHEDULER_ROUND_ROBIN;
            req->priority = 60;
            req->model_name[0] = '\0';
            break;

        case TEST_MODE_SHORT_TIMEOUT:
            req->max_tokens = 1000;
            req->temperature_x100 = 70;
            req->timeout_ms = 100;      /* Very short timeout (100ms) */
            req->scheduler_algorithm = SCHEDULER_FALLBACK;
            req->priority = 90;         /* High priority */
            req->model_name[0] = '\0';
            break;

        case TEST_MODE_INVALID_MODEL:
            req->max_tokens = 1000;
            req->temperature_x100 = 70;
            req->timeout_ms = 30000;
            req->scheduler_algorithm = SCHEDULER_PRIORITY;
            req->priority = 50;
            /* Set an invalid model name */
            strncpy(req->model_name, "non-existent-model-12345", MAX_MODEL_NAME - 1);
            break;
    }
}

/* Thread function */
void *stress_test_thread(void *arg) {
    struct thread_data *data = (struct thread_data *)arg;
    struct llm_request req;
    char prompt_buffer[MAX_PROMPT_LENGTH];
    char response[MAX_RESPONSE_LENGTH];
    int fd, ret, request_count = 0;
    ssize_t bytes_read;
    struct thread_stats local_stats = {0};

    /* Use thread-specific seed for better randomization */
    srand(data->seed);

    printf("[Thread %d] Starting stress test\n", data->thread_id);

    /* Open the device */
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        printf("[Thread %d] Failed to open device: %s\n",
               data->thread_id, strerror(errno));
        return NULL;
    }

    while (*(data->keep_running)) {
        int conversation_id = 3000 + (data->thread_id * 1000) + request_count;
        int test_mode = rand() % 6; /* Select a random test mode */
        long start_time, end_time;

        /* Initialize request */
        memset(&req, 0, sizeof(req));
        memset(prompt_buffer, 0, sizeof(prompt_buffer));

        /* Generate prompt based on test mode */
        generate_prompt(prompt_buffer, sizeof(prompt_buffer), test_mode, data->thread_id);
        strncpy(req.prompt, prompt_buffer, MAX_PROMPT_LENGTH - 1);

        /* Configure request parameters */
        configure_request(&req, test_mode, data->thread_id, conversation_id);

        printf("[Thread %d] Request %d: Mode %d, Conversation %d\n",
               data->thread_id, request_count + 1, test_mode, conversation_id);

        /* Measure response time */
        start_time = get_time_ms();

        /* Send request to device */
        ret = write(fd, &req, sizeof(req));
        if (ret < 0) {
            printf("[Thread %d] Failed to write to device: %s\n",
                   data->thread_id, strerror(errno));

            local_stats.requests_sent++;
            local_stats.failed_responses++;
            local_stats.errors_by_type[test_mode]++;

            /* Small delay before retrying */
            usleep(200000); /* 200ms */
            continue;
        }

        /* Read response */
        memset(response, 0, sizeof(response));
        bytes_read = read(fd, response, sizeof(response) - 1);

        end_time = get_time_ms();

        local_stats.requests_sent++;

        if (bytes_read < 0) {
            printf("[Thread %d] Failed to read response: %s\n",
                   data->thread_id, strerror(errno));
            local_stats.failed_responses++;
            local_stats.errors_by_type[test_mode]++;
        } else if (bytes_read == 0) {
            printf("[Thread %d] No response received\n", data->thread_id);
            local_stats.failed_responses++;
            local_stats.errors_by_type[test_mode]++;
        } else {
            response[bytes_read] = '\0';
            printf("[Thread %d] Response received (%zd bytes, %ld ms)\n",
                   data->thread_id, bytes_read, end_time - start_time);

            local_stats.successful_responses++;
            local_stats.total_response_time_ms += (end_time - start_time);
        }

        request_count++;

        /* Adaptive delay between requests based on success/failure */
        if (bytes_read <= 0) {
            /* Longer delay after failures */
            usleep(500000 + (rand() % 500000)); /* 500-1000ms */
        } else {
            /* Shorter delay after success */
            usleep(100000 + (rand() % 200000)); /* 100-300ms */
        }
    }

    close(fd);

    /* Update global statistics */
    pthread_mutex_lock(&data->stats->mutex);
    data->stats->total_requests += local_stats.requests_sent;
    data->stats->successful_responses += local_stats.successful_responses;
    data->stats->failed_responses += local_stats.failed_responses;
    data->stats->total_response_time_ms += local_stats.total_response_time_ms;

    for (int i = 0; i < 6; i++) {
        data->stats->errors_by_type[i] += local_stats.errors_by_type[i];
    }
    pthread_mutex_unlock(&data->stats->mutex);

    printf("[Thread %d] Thread completed after %d requests\n",
           data->thread_id, request_count);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data_array[NUM_THREADS];
    int i, ret;
    time_t start_time, current_time;

    printf("LLM Orchestrator Stress Test\n");
    printf("----------------------------\n");
    printf("Test duration: %d seconds\n", TEST_DURATION_SEC);
    printf("Number of threads: %d\n", NUM_THREADS);

    /* Initialize mutex for global statistics */
    pthread_mutex_init(&g_stats.mutex, NULL);

    /* Set up signal handler for clean termination */
    signal(SIGINT, signal_handler);

    /* Record start time */
    start_time = time(NULL);

    /* Configure thread data */
    for (i = 0; i < NUM_THREADS; i++) {
        thread_data_array[i].thread_id = i + 1;
        thread_data_array[i].seed = time(NULL) + i;
        thread_data_array[i].stats = &g_stats;
        thread_data_array[i].keep_running = &keep_running;
    }

    /* Create threads */
    for (i = 0; i < NUM_THREADS; i++) {
        ret = pthread_create(&threads[i], NULL, stress_test_thread,
                            &thread_data_array[i]);
        if (ret) {
            printf("Error creating thread %d: %s\n", i + 1, strerror(ret));
            exit(EXIT_FAILURE);
        }

        /* Stagger thread creation to prevent initial spike */
        usleep(50000); /* 50ms */
    }

    /* Monitor progress and stop after duration */
    while (keep_running) {
        sleep(1);

        current_time = time(NULL);
        if (current_time - start_time >= TEST_DURATION_SEC) {
            printf("\nTest duration reached, stopping threads...\n");
            keep_running = 0;
        }
    }

    /* Wait for threads to complete */
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Display final statistics */
    printf("\nStress Test Results:\n");
    printf("  Total Requests: %d\n", g_stats.total_requests);
    printf("  Successful Responses: %d (%.1f%%)\n",
           g_stats.successful_responses,
           (g_stats.total_requests > 0) ?
           (float)g_stats.successful_responses / g_stats.total_requests * 100 : 0);
    printf("  Failed Responses: %d (%.1f%%)\n",
           g_stats.failed_responses,
           (g_stats.total_requests > 0) ?
           (float)g_stats.failed_responses / g_stats.total_requests * 100 : 0);
    printf("  Average Response Time: %.1f ms\n",
           (g_stats.successful_responses > 0) ?
           (float)g_stats.total_response_time_ms / g_stats.successful_responses : 0);

    printf("\nErrors by Test Mode:\n");
    const char *mode_names[] = {
        "Normal", "Edge Case", "Error Case",
        "Large Prompt", "Short Timeout", "Invalid Model"
    };

    for (i = 0; i < 6; i++) {
        printf("  %s: %d\n", mode_names[i], g_stats.errors_by_type[i]);
    }

    pthread_mutex_destroy(&g_stats.mutex);

    return 0;
}

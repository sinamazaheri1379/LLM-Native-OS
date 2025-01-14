//
// Created by sina-mazaheri on 12/28/24.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define DEVICE_PATH "/dev/llm"
#define MAX_PROMPT_LEN 4096
#define MAX_RESPONSE_LEN 8192

struct llm_request {
    char prompt[MAX_PROMPT_LEN];
    char role[32];
    char model[64];
};

int main(int argc, char *argv[]) {
    int fd;
    struct llm_request req;
    char response[MAX_RESPONSE_LEN];

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <prompt> [role] [model]\n", argv[0]);
        return 1;
    }

    /* Open device */
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return 1;
    }

    /* Prepare request */
    strncpy(req.prompt, argv[1], MAX_PROMPT_LEN - 1);
    strncpy(req.role, argc > 2 ? argv[2] : "user", 31);
    strncpy(req.model, argc > 3 ? argv[3] : "gpt-3.5-turbo", 63);

    /* Send request */
    if (write(fd, &req, sizeof(req)) < 0) {
        perror("Failed to send request");
        close(fd);
        return 1;
    }

    /* Read response */
    ssize_t bytes_read = read(fd, response, MAX_RESPONSE_LEN - 1);
    if (bytes_read < 0) {
        perror("Failed to read response");
        close(fd);
        return 1;
    }
    response[bytes_read] = '\0';

    /* Print response */
    printf("Response:\n%s\n", response);

    close(fd);
    return 0;
}
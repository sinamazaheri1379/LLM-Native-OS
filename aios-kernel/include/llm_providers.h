//
// Created by sina-mazaheri on 12/17/24.
//
#ifndef LLM_PROVIDERS_H
#define LLM_PROVIDERS_H

#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/net.h>
#include <linux/in.h>
#include <net/sock.h>

/* Maximum buffer sizes */
#define MAX_API_KEY_LENGTH   256
#define MAX_PROMPT_LENGTH    4096
#define MAX_RESPONSE_LENGTH  8192
#define MAX_MODEL_NAME       64
#define MAX_ENDPOINT_LENGTH  256
#define MAX_ERROR_LENGTH     256

/* Provider types */
enum llm_provider {
    LLM_OPENAI = 0,
    LLM_ANTHROPIC,
    LLM_MISTRAL,
    LLM_HUGGINGFACE,
    LLM_GEMINI,
    LLM_MAX_PROVIDERS
};

/**
 * struct llm_config - Configuration for an LLM provider
 * @provider:        The selected LLM provider
 * @api_key:        Authentication key
 * @model:          Model identifier
 * @max_tokens:     Maximum response tokens
 * @temperature_X100: Temperature * 100 (70 = 0.7)
 * @endpoint:       API endpoint
 * @use_ssl:        SSL/TLS flag
 * @timeout_ms:     Network timeout in ms
 */
struct llm_config {
    enum llm_provider provider;
    char api_key[MAX_API_KEY_LENGTH];
    char model[MAX_MODEL_NAME];
    int max_tokens;
    int temperature_X100;
    char endpoint[MAX_ENDPOINT_LENGTH];
    bool use_ssl;
    int timeout_ms;
}

/**
 * struct llm_message - Message container
 * @content:     Message content
 * @length:      Content length
 * @is_response: Response flag
 * @status_code: Status/error code
 * @error:       Error message
 */
struct llm_message {
    char *content;
    size_t length;
    bool is_response;
    int status_code;
    char error[MAX_ERROR_LENGTH];
}

/**
 * struct llm_provider_ops - Provider operations
 * @init:    Initialize provider
 * @cleanup: Clean resources
 * @send:    Send request
 * @receive: Receive response
 *
 * All functions return 0 on success or negative error code
 */
struct llm_provider_ops {
    int (*init)(struct llm_config *config);
    void (*cleanup)(void);
    int (*send)(const char *prompt, size_t length);
    int (*receive)(struct llm_message *msg);
};

/* Public interface */
int llm_init(struct llm_config *config);
void llm_cleanup(void);
int llm_send(const char *prompt, size_t length);
int llm_receive(struct llm_message *msg);

#endif /* LLM_PROVIDERS_H */


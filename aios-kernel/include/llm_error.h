//
// Created by sina-mazaheri on 12/17/24.
//

#ifndef LLM_ERROR_H
#define LLM_ERROR_H

struct llm_error {
    int code;
    char message[256];
    bool recoverable;
    int retry_after;
};

void llm_handle_error(int error_code);
const char *llm_error_string(int error_code);

#endif /* LLM_ERROR_H */

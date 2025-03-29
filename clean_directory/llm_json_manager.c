/*
 * LLM Orchestrator - JSON utilities
 * Centralized JSON manipulation functions with improved robustness
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>
#include <linux/ctype.h>
#include "orchestrator_main.h"

/* Buffer size constants */
#define JSON_MIN_BUFFER_SIZE 1024
#define JSON_MAX_BUFFER_SIZE (1024 * 1024) /* 1MB max buffer */
#define JSON_RESIZE_MARGIN 128 /* Extra margin when resizing */
#define JSON_UNICODE_MAX_LEN 4 /* Max bytes for UTF-8 representation */
#define JSON_MAX_NUM_BUFFER 32 /* Buffer for numeric conversions */

/* Initialization state tracking */
static bool json_manager_initialized = false;
static DEFINE_SPINLOCK(json_lock);

/* Statistics for operations */
static atomic_t buffers_created = ATOMIC_INIT(0);
static atomic_t buffers_resized = ATOMIC_INIT(0);
static atomic_t json_parse_attempts = ATOMIC_INIT(0);
static atomic_t json_parse_successes = ATOMIC_INIT(0);

/* Known field names for more robust parsing */
static const char * const prompt_token_fields[] = {
        "prompt_tokens",
        "input_tokens",
        "in_tokens"
};
#define NUM_PROMPT_TOKEN_FIELDS ARRAY_SIZE(prompt_token_fields)

static const char * const completion_token_fields[] = {
        "completion_tokens",
        "output_tokens",
        "out_tokens"
};
#define NUM_COMPLETION_TOKEN_FIELDS ARRAY_SIZE(completion_token_fields)

static const char * const usage_section_fields[] = {
        "\"usage\"",
        "\"token_usage\"",
        "\"usage_metrics\""
};
#define NUM_USAGE_SECTION_FIELDS ARRAY_SIZE(usage_section_fields)

static const char * const content_field_patterns[] = {
        "\"content\":",
        "\"text\":",
        "\"message\":",
        "\"response\":",
        "\"answer\":",
        "\"result\":"
};
#define NUM_CONTENT_PATTERNS ARRAY_SIZE(content_field_patterns)

/*
 * Forward declarations of static helper functions
 */
static int ensure_buffer_space(struct llm_json_buffer *buf, size_t needed);
static int decode_unicode_escape(const char *hex, char *output, size_t output_size);
static const char *json_strcasestr(const char *haystack, const char *needle);
static int extract_number_after_field(const char *json, const char *field_name, int *result);




static int decode_chunked_response(const char *chunked, char *output, size_t output_size)
{
    const char *p = chunked;
    char *out = output;
    size_t remaining_out = output_size - 1; /* Leave room for null terminator */
    size_t out_len = 0;

    /* Skip any leading whitespace or control chars */
    while (*p && (*p <= ' ' || !isprint(*p))) p++;

    while (*p && remaining_out > 0) {
        /* Extract hex chunk size */
        char hex_str[16] = {0};
        int hex_idx = 0;
        size_t chunk_size = 0;

        /* Read hex digits until non-hex character */
        while (isxdigit(*p) && hex_idx < 15) {
            hex_str[hex_idx++] = *p++;
        }
        hex_str[hex_idx] = '\0';

        /* Convert hex string to number */
        if (hex_idx > 0) {
            if (kstrtoul(hex_str, 16, &chunk_size) != 0) {
                pr_warn("decode_chunked_response: Invalid hex chunk size: %s\n", hex_str);
                break;
            }
        } else {
            /* No valid hex digits found, assume not chunked */
            return -EINVAL;
        }

        /* Skip CRLF after chunk size */
        while (*p && (*p == '\r' || *p == '\n' || *p == ' ')) p++;

        /* If chunk size is 0, we've reached the end */
        if (chunk_size == 0) {
            break;
        }

        /* Copy chunk data (limiting to available output space) */
        if (chunk_size > remaining_out) {
            chunk_size = remaining_out;
        }

        /* Ensure we don't read past the end of the input */
        memcpy(out, p, chunk_size);
        out += chunk_size;
        out_len += chunk_size;
        remaining_out -= chunk_size;
        p += chunk_size;

        /* Skip CRLF after chunk data */
        while (*p && (*p == '\r' || *p == '\n')) p++;
    }

    /* Null terminate the output */
    *out = '\0';

    /* Debug output */
    pr_debug("decode_chunked_response: Successfully decoded %zu bytes\n", out_len);

    return out_len > 0 ? out_len : -EINVAL;
}
/*
 * Initialize the JSON manager subsystem
 * Returns 0 on success
 */
int json_manager_init(void)
{
    if (json_manager_initialized) {
        pr_warn("json_manager_init: Already initialized\n");
        return 0;
    }

    /* Reset statistics */
    atomic_set(&buffers_created, 0);
    atomic_set(&buffers_resized, 0);
    atomic_set(&json_parse_attempts, 0);
    atomic_set(&json_parse_successes, 0);

    json_manager_initialized = true;
    pr_info("JSON manager subsystem initialized\n");
    return 0;
}

/*
 * Clean up the JSON manager subsystem
 */
void json_manager_cleanup(void)
{
    if (!json_manager_initialized) {
        pr_warn("json_manager_cleanup: Not initialized\n");
        return;
    }

    json_manager_initialized = false;
    pr_info("JSON manager subsystem cleaned up\n");
}

/*
 * Check if JSON manager is initialized
 */
bool json_manager_initialized_check(void)
{
    return json_manager_initialized;
}

/*
 * Initialize a JSON buffer with validation
 * Returns 0 on success, negative error code on failure
 */
int json_buffer_init(struct llm_json_buffer *buf, size_t size)
{
    unsigned long flags;

    if (!buf)
        return -EINVAL;

    if (!json_manager_initialized) {
        pr_warn("json_buffer_init: JSON manager not initialized\n");
        return -EAGAIN;
    }

    /* Enforce minimum buffer size */
    if (size < JSON_MIN_BUFFER_SIZE)
        size = JSON_MIN_BUFFER_SIZE;

    /* Enforce maximum buffer size */
    if (size > JSON_MAX_BUFFER_SIZE) {
        pr_warn("json_buffer_init: Requested size %zu exceeds max %d, using max\n",
                size, JSON_MAX_BUFFER_SIZE);
        size = JSON_MAX_BUFFER_SIZE;
    }

    /* Use GFP_ATOMIC for allocations that might happen in interrupt context */
    buf->data = kmalloc(size, GFP_ATOMIC);
    if (!buf->data)
        return -ENOMEM;

    buf->size = size;
    buf->used = 0;
    buf->data[0] = '\0';

    /* Lock for statistical update */
    spin_lock_irqsave(&json_lock, flags);
    atomic_inc(&buffers_created);
    spin_unlock_irqrestore(&json_lock, flags);

    return 0;
}

/*
 * Resize a JSON buffer to accommodate more data
 * Returns 0 on success, negative error code on failure
 */
int json_buffer_resize(struct llm_json_buffer *buf, size_t new_size)
{
    char *new_data;
    unsigned long flags;

    if (!buf || !buf->data)
        return -EINVAL;

    if (!json_manager_initialized)
        return -EAGAIN;

    /* Don't resize if it's already big enough */
    if (new_size <= buf->size)
        return 0;

    /* Enforce maximum buffer size */
    if (new_size > JSON_MAX_BUFFER_SIZE) {
        pr_warn("json_buffer_resize: Requested size %zu exceeds max %d, using max\n",
                new_size, JSON_MAX_BUFFER_SIZE);
        new_size = JSON_MAX_BUFFER_SIZE;

        /* If current size is already at max, nothing to do */
        if (buf->size >= JSON_MAX_BUFFER_SIZE)
            return 0;
    }

    /* Allocate new buffer */
    new_data = kmalloc(new_size, GFP_ATOMIC);
    if (!new_data)
        return -ENOMEM;

    /* Copy existing data */
    memcpy(new_data, buf->data, buf->used + 1); /* +1 for null terminator */

    /* Replace old buffer */
    kfree(buf->data);
    buf->data = new_data;
    buf->size = new_size;

    /* Update statistics */
    spin_lock_irqsave(&json_lock, flags);
    atomic_inc(&buffers_resized);
    spin_unlock_irqrestore(&json_lock, flags);

    return 0;
}

/*
 * Free a JSON buffer with validation
 */
void json_buffer_free(struct llm_json_buffer *buf)
{
    if (!buf)
        return;

    if (buf->data) {
        kfree(buf->data);
        buf->data = NULL;
    }

    buf->size = 0;
    buf->used = 0;
}

/*
 * Helper function to ensure buffer has enough space
 * Returns 0 if buffer has enough space or was successfully resized
 * Returns negative error code on failure
 */
static int ensure_buffer_space(struct llm_json_buffer *buf, size_t needed)
{
    size_t remaining, new_size;

    if (!buf || !buf->data)
        return -EINVAL;

    /* Check if we already have enough space */
    remaining = buf->size - buf->used;
    if (remaining > needed)
        return 0;

    /* Calculate new size (double the buffer size or add needed space, whichever is larger) */
    new_size = buf->size * 2;
    if (new_size - buf->used < needed)
        new_size = buf->used + needed + JSON_RESIZE_MARGIN;

    /* Attempt to resize */
    return json_buffer_resize(buf, new_size);
}

/*
 * Helper function to append a JSON string with automatic buffer resizing
 * Returns 0 on success, negative error code on failure
 */
int append_json_string(struct llm_json_buffer *buf, const char *str)
{
    size_t len;
    int ret;

    if (!buf || !buf->data || !str)
        return -EINVAL;

    if (!json_manager_initialized)
        return -EAGAIN;

    len = strlen(str);

    /* Ensure we have enough space (including null terminator) */
    ret = ensure_buffer_space(buf, len + 1);
    if (ret)
        return ret;

    memcpy(buf->data + buf->used, str, len);
    buf->used += len;
    buf->data[buf->used] = '\0';

    return 0;
}

/*
 * Helper function to append a JSON number with automatic buffer resizing
 * Returns 0 on success, negative error code on failure
 */
int append_json_number(struct llm_json_buffer *buf, int number)
{
    char value[JSON_MAX_NUM_BUFFER];
    int ret;

    if (!buf || !buf->data)
        return -EINVAL;

    if (!json_manager_initialized)
        return -EAGAIN;

    ret = snprintf(value, sizeof(value), "%d", number);
    if (ret < 0 || ret >= sizeof(value))
        return -EINVAL;

    return append_json_string(buf, value);
}

/*
 * Helper function to append a JSON float value (stored as integer * 100)
 * Returns 0 on success, negative error code on failure
 */
int append_json_float(struct llm_json_buffer *buf, int value_x100)
{
    char value[JSON_MAX_NUM_BUFFER];
    int int_part, frac_part, ret;

    if (!buf || !buf->data)
        return -EINVAL;

    if (!json_manager_initialized)
        return -EAGAIN;

    int_part = value_x100 / 100;
    frac_part = abs(value_x100 % 100); /* Ensure positive fraction */
    ret = snprintf(value, sizeof(value), "%d.%02d", int_part, frac_part);
    if (ret < 0 || ret >= sizeof(value))
        return -EINVAL;

    return append_json_string(buf, value);
}

/*
 * Helper function to append a JSON boolean with validation
 * Returns 0 on success, negative error code on failure
 */
int append_json_boolean(struct llm_json_buffer *buf, bool value)
{
    if (!buf || !buf->data)
        return -EINVAL;

    if (!json_manager_initialized)
        return -EAGAIN;

    return append_json_string(buf, value ? "true" : "false");
}

/*
 * Decode a Unicode hex escape sequence into UTF-8
 * Writes to the output buffer and returns number of bytes written
 * Returns negative error code on failure
 */
static int decode_unicode_escape(const char *hex, char *output, size_t output_size)
{
    unsigned int unicode;
    int ret;

    if (!hex || !output || output_size < JSON_UNICODE_MAX_LEN)
        return -EINVAL;

    /* Parse hex digits to get Unicode code point */
    ret = kstrtouint(hex, 16, &unicode);
    if (ret)
        return ret;

    /* Convert to UTF-8 */
    if (unicode < 0x80) {
        /* Single byte (ASCII) */
        output[0] = (char)unicode;
        return 1;
    } else if (unicode < 0x800) {
        /* Two bytes */
        if (output_size < 2)
            return -ENOSPC;
        output[0] = 0xC0 | (unicode >> 6);
        output[1] = 0x80 | (unicode & 0x3F);
        return 2;
    } else if (unicode < 0x10000) {
        /* Three bytes */
        if (output_size < 3)
            return -ENOSPC;
        output[0] = 0xE0 | (unicode >> 12);
        output[1] = 0x80 | ((unicode >> 6) & 0x3F);
        output[2] = 0x80 | (unicode & 0x3F);
        return 3;
    } else if (unicode < 0x110000) {
        /* Four bytes */
        if (output_size < 4)
            return -ENOSPC;
        output[0] = 0xF0 | (unicode >> 18);
        output[1] = 0x80 | ((unicode >> 12) & 0x3F);
        output[2] = 0x80 | ((unicode >> 6) & 0x3F);
        output[3] = 0x80 | (unicode & 0x3F);
        return 4;
    }

    /* Invalid Unicode code point */
    return -EINVAL;
}

/* Fix 2: Enhanced bounds checking in append_json_value() in llm_json_manager.c */
int append_json_value(struct llm_json_buffer *buf, const char *value)
{
    size_t i, len;
    int ret;

    if (!buf || !buf->data || !value)
        return -EINVAL;

    if (!json_manager_initialized)
        return -EAGAIN;

    len = strlen(value);

    /* Ensure we have enough space for worst case (every char escaped as \uXXXX) */
    ret = ensure_buffer_space(buf, len * 6 + 1);
    if (ret)
        return ret;

    for (i = 0; i < len; i++) {
        char c = value[i];

        /* Check if we need to escape this character */
        if (c == '"' || c == '\\' || c == '\b' || c == '\f' ||
            c == '\n' || c == '\r' || c == '\t') {

            /* Make sure we have enough space (2 chars) */
            if (buf->used + 2 >= buf->size) {
                ret = ensure_buffer_space(buf, 2);
                if (ret)
                    return ret;
            }

            buf->data[buf->used++] = '\\';

            switch (c) {
                case '"':  buf->data[buf->used++] = '"';  break;
                case '\\': buf->data[buf->used++] = '\\'; break;
                case '\b': buf->data[buf->used++] = 'b';  break;
                case '\f': buf->data[buf->used++] = 'f';  break;
                case '\n': buf->data[buf->used++] = 'n';  break;
                case '\r': buf->data[buf->used++] = 'r';  break;
                case '\t': buf->data[buf->used++] = 't';  break;
            }
        } else if ((unsigned char)c < 32) {
            /* Control characters need special handling (6 chars for \uXXXX) */
            if (buf->used + 6 >= buf->size) {
                ret = ensure_buffer_space(buf, 6);
                if (ret)
                    return ret;
            }

            /* Format as Unicode escape sequence \u00XX */
            buf->data[buf->used++] = '\\';
            buf->data[buf->used++] = 'u';
            buf->data[buf->used++] = '0';
            buf->data[buf->used++] = '0';

            /* Convert to hex with bounds checking */
            if (buf->used + 2 <= buf->size) {
                ret = snprintf(buf->data + buf->used, buf->size - buf->used, "%02x", (unsigned char)c);
                if (ret != 2)
                    return -EIO; /* Unexpected failure in snprintf */
                buf->used += 2;
            } else {
                return -ENOSPC; /* Not enough space despite previous checks */
            }
        } else {
            /* Normal character */
            if (buf->used + 1 >= buf->size) {
                ret = ensure_buffer_space(buf, 1);
                if (ret)
                    return ret;
            }

            buf->data[buf->used++] = c;
        }
    }

    /* Ensure null termination */
    if (buf->used >= buf->size) {
        ret = ensure_buffer_space(buf, 1);
        if (ret)
            return ret;
    }
    buf->data[buf->used] = '\0';

    return 0;
}

/*
 * Case-insensitive search for a substring
 * Returns a pointer to the beginning of the match, or NULL if not found
 *
 * This implementation avoids nested loops for better performance
 */
static const char *json_strcasestr(const char *haystack, const char *needle)
{
    const char *h, *n, *match;
    char h_ch, n_ch;

    if (!haystack || !needle)
        return NULL;

    if (!*needle)
        return haystack;

    while (*haystack) {
        /* Find the first character match */
        h_ch = tolower(*haystack);
        n_ch = tolower(*needle);

        if (h_ch == n_ch) {
            /* Potential match found, compare the rest */
            h = haystack + 1;
            n = needle + 1;
            match = haystack;

            while (*h && *n) {
                h_ch = tolower(*h);
                n_ch = tolower(*n);

                if (h_ch != n_ch)
                    break;

                h++;
                n++;
            }

            /* If we reached the end of needle, we found a match */
            if (!*n)
                return match;
        }

        haystack++;
    }

    return NULL;
}

/*
 * Validate a JSON string (basic check)
 * Returns true if the string appears to be valid JSON
 */
bool validate_json(const char *json)
{
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    const char *p;

    if (!json)
        return false;

    /* Skip whitespace */
    for (p = json; *p && isspace(*p); p++)
        ;

    /* JSON must start with { or [ */
    if (*p != '{' && *p != '[')
        return false;

    /* Simple balancing check */
    for (p = json; *p; p++) {
        if (escaped) {
            escaped = false;
            continue;
        }

        if (*p == '\\') {
            escaped = true;
            continue;
        }

        if (*p == '"' && !escaped) {
            in_string = !in_string;
            continue;
        }

        if (!in_string) {
            if (*p == '{' || *p == '[')
                depth++;
            else if (*p == '}' || *p == ']')
                depth--;

            if (depth < 0)
                return false; /* Unbalanced */
        }
    }

    /* Check that we're not in the middle of a string and all brackets are balanced */
    return (depth == 0) && !in_string;
}

void display_content(char* contents){
        /* Create a copy of the body pointer for debugging */
        char *debug_body = contents; /* Skip past header separator */

        /* Get body length */
        size_t body_len = strlen(debug_body);
        size_t remaining = body_len;
        size_t offset = 0;

        /* Use static buffer to avoid stack overflow */
        static char tmp_buf[DEBUG_CHUNK_SIZE + 1];

        pr_info("=================== RESPONSE BODY DUMP ===================\n");

        while (remaining > 0) {
            /* Print in chunks to avoid overwhelming kernel log buffer */
            size_t chunk_size = (remaining > DEBUG_CHUNK_SIZE) ? DEBUG_CHUNK_SIZE : remaining;

            /* Copy chunk to temporary buffer and null-terminate */
            memcpy(tmp_buf, debug_body + offset, chunk_size);
            tmp_buf[chunk_size] = '\0';

            /* Print this chunk */
            pr_info("BODY CHUNK %zu/%zu:\n%s\n",
                    offset/DEBUG_CHUNK_SIZE + 1,
                    (body_len + DEBUG_CHUNK_SIZE - 1)/DEBUG_CHUNK_SIZE,
                    tmp_buf);

            /* Update counters */
            offset += chunk_size;
            remaining -= chunk_size;
        }
        pr_info("============== END OF RESPONSE BODY DUMP ================\n");

}


/*
 * Helper function to extract numeric value after a field
 * Returns 0 on success, negative error code on failure
 */
static int extract_number_after_field(const char *json, const char *field_name, int *result)
{
    const char *field;
    const char *colon;
    const char *num_start;
    char number_buf[JSON_MAX_NUM_BUFFER];
    int j = 0;

    if (!json || !field_name || !result)
        return -EINVAL;

    field = json_strcasestr(json, field_name);
    if (!field)
        return -ENOENT;

    colon = strchr(field, ':');
    if (!colon)
        return -EINVAL;

    /* Skip whitespace */
    num_start = colon + 1;
    while (*num_start && (*num_start == ' ' || *num_start == '\t'))
        num_start++;

    /* Extract the number */
    while (*num_start && j < sizeof(number_buf) - 1 &&
           (isdigit(*num_start) || *num_start == '-')) {
        number_buf[j++] = *num_start++;
    }
    number_buf[j] = '\0';

    /* Convert to integer */
    if (j > 0)
        return kstrtoint(number_buf, 10, result);

    return -EINVAL;
}

/*
 * Extract assistant message content from OpenAI response format
 * Specifically handles the OpenAI-style nested JSON with choices array
 */
/*
 * Extract assistant message content from OpenAI response format
 * Handles chunked transfer encoding
 */
int extract_openai_content(const char *json, char *output, size_t output_size)
{
    char *decoded_json = NULL;
    int decoded_len = 0;
    const char *json_to_use;
    const char *content_start = NULL;
    const char *content_end = NULL;
    int ret = -EINVAL;
    size_t out_idx = 0;
    bool escaped = false;

    if (!json || !output || output_size == 0)
        return -EINVAL;

    /* Clear output buffer */
    output[0] = '\0';

    /* Allocate buffer for decoded JSON */
    decoded_json = kmalloc(MAX_RESPONSE_LENGTH, GFP_KERNEL);
    if (!decoded_json)
        return -ENOMEM;

    /* Decode chunked encoding if present */
    decoded_len = decode_chunked_response(json, decoded_json, MAX_RESPONSE_LENGTH);

    /* Determine which JSON to parse */
    if (decoded_len > 0) {
        json_to_use = decoded_json;
        pr_debug("Using decoded JSON (%d bytes)\n", decoded_len);
    } else {
        json_to_use = json;
        pr_debug("Using original JSON\n");
    }
	display_content((char*) json_to_use);
    /* Look for choices array in a more direct way */
    const char *choices = strstr(json_to_use, "\"choices\"");
    if (!choices) {
        pr_debug("No choices field found\n");
        ret = -EINVAL;
        goto cleanup;
    }
	display_content((char*) choices);
    /* Find the content field - search for "content":" directly */
    content_start = strstr(choices, "\"content\"");
    if (!content_start) {
        pr_debug("No content field found\n");
        ret = -EINVAL;
        goto cleanup;
    }
	display_content((char*) content_start);
    /* Skip past "content":" */
    content_start += 12; /* Length of "content":" */

    /* Find the closing quote, handling escaped quotes */
    content_end = content_start;
    while (*content_end) {
        if (escaped) {
            escaped = false;
        } else if (*content_end == '\\') {
            escaped = true;
        } else if (*content_end == '"') {
            break;
        }
        content_end++;
    }

    if (*content_end != '"') {
        pr_debug("No closing quote found for content\n");
        ret = -EINVAL;
        goto cleanup;
    }
	display_content((char*) content_end);
    /* Copy and unescape the content */
    while (content_start < content_end && out_idx < output_size - 1) {
        if (*content_start == '\\' && content_start + 1 < content_end) {
            content_start++;
            switch (*content_start) {
                case 'n': output[out_idx++] = '\n'; break;
                case 'r': output[out_idx++] = '\r'; break;
                case 't': output[out_idx++] = '\t'; break;
                case 'b': output[out_idx++] = '\b'; break;
                case 'f': output[out_idx++] = '\f'; break;
                case '"': output[out_idx++] = '"'; break;
                case '\\': output[out_idx++] = '\\'; break;
                default: output[out_idx++] = *content_start;
            }
        } else {
            output[out_idx++] = *content_start;
        }
        content_start++;
    }
    output[out_idx] = '\0';
    ret = out_idx;
cleanup:
    if (decoded_json)
        kfree(decoded_json);

    return ret;
}

/*
 * Extract content from Anthropic (Claude) response format
 * Handles chunked transfer encoding
 */
int extract_anthropic_content(const char *json, char *output, size_t output_size)
{
    char *decoded_json = NULL;
    const char *content_text_marker = "\"text\":\"";  /* New Claude format */
    const char *completion_marker = "\"completion\":\"";  /* Older format */
    const char *content_pos = NULL;
    const char *content_end;
    int i, result = -EINVAL;
    size_t out_idx = 0;
    bool escaped = false;

    if (!json || !output || output_size == 0)
        return -EINVAL;

    output[0] = '\0';

    /* First, allocate a buffer for the decoded JSON */
    decoded_json = kmalloc(MAX_RESPONSE_LENGTH, GFP_KERNEL);
    if (!decoded_json) {
        pr_err("extract_anthropic_content: Failed to allocate decode buffer\n");
        return -ENOMEM;
    }

    /* Decode chunked encoding if present */
    if (decode_chunked_response(json, decoded_json, MAX_RESPONSE_LENGTH) <= 0) {
        /* Decoding failed or unnecessary, use original */
        pr_debug("extract_anthropic_content: Chunk decoding failed, using original\n");
        strncpy(decoded_json, json, MAX_RESPONSE_LENGTH - 1);
        decoded_json[MAX_RESPONSE_LENGTH - 1] = '\0';
    }

    /* Try to find the new format with "content" array with "text" field */
    content_pos = strstr(decoded_json, content_text_marker);

    /* If not found, try the older format with "completion" field */
    if (!content_pos) {
        content_pos = strstr(decoded_json, completion_marker);
        if (content_pos) {
            content_pos += strlen(completion_marker);
        }
    } else {
        content_pos += strlen(content_text_marker);
    }

    if (!content_pos) {
        pr_debug("extract_anthropic_content: Failed to find content field\n");
        result = -EINVAL;
        goto cleanup;
    }

    /* Find the end quote of the content */
    content_end = content_pos;
    while (*content_end) {
        if (escaped) {
            escaped = false;
        } else if (*content_end == '\\') {
            escaped = true;
        } else if (*content_end == '"') {
            break;
        }
        content_end++;
    }

    if (*content_end != '"') {
        pr_debug("extract_anthropic_content: Failed to find end of content\n");
        result = -EINVAL;
        goto cleanup;
    }

    /* Copy and unescape content */
    for (i = 0; content_pos + i < content_end && out_idx < output_size - 1; i++) {
        if (content_pos[i] == '\\' && i + 1 < (content_end - content_pos)) {
            i++;
            switch (content_pos[i]) {
                case 'n': output[out_idx++] = '\n'; break;
                case 'r': output[out_idx++] = '\r'; break;
                case 't': output[out_idx++] = '\t'; break;
                case 'b': output[out_idx++] = '\b'; break;
                case 'f': output[out_idx++] = '\f'; break;
                case '"': output[out_idx++] = '"'; break;
                case '\\': output[out_idx++] = '\\'; break;
                case 'u': /* Unicode escape - simplified handling */
                    output[out_idx++] = '?'; /* Placeholder for Unicode */
                    i += 4; /* Skip past the 4 hex digits */
                    break;
                default: output[out_idx++] = content_pos[i]; break;
            }
        } else {
            output[out_idx++] = content_pos[i];
        }
    }

    output[out_idx] = '\0';
    result = out_idx;

cleanup:
    if (decoded_json)
        kfree(decoded_json);

    return result;
}

/*
 * Extract content from Google Gemini response format
 * Handles chunked transfer encoding
 */
int extract_gemini_content(const char *json, char *output, size_t output_size)
{
    char *decoded_json = NULL;
    const char *candidates_marker = "\"candidates\":[";
    const char *parts_marker = "\"parts\":[";
    const char *text_marker = "\"text\":\"";
    const char *candidates_pos, *parts_pos, *text_pos, *content_end;
    int i, result = -EINVAL;
    size_t out_idx = 0;
    bool escaped = false;

    if (!json || !output || output_size == 0)
        return -EINVAL;

    output[0] = '\0';

    /* First, allocate a buffer for the decoded JSON */
    decoded_json = kmalloc(MAX_RESPONSE_LENGTH, GFP_KERNEL);
    if (!decoded_json) {
        pr_err("extract_gemini_content: Failed to allocate decode buffer\n");
        return -ENOMEM;
    }

    /* Decode chunked encoding if present */
    if (decode_chunked_response(json, decoded_json, MAX_RESPONSE_LENGTH) <= 0) {
        /* Decoding failed or unnecessary, use original */
        pr_debug("extract_gemini_content: Chunk decoding failed, using original\n");
        strncpy(decoded_json, json, MAX_RESPONSE_LENGTH - 1);
        decoded_json[MAX_RESPONSE_LENGTH - 1] = '\0';
    }

    /* Find the candidates array */
    candidates_pos = strstr(decoded_json, candidates_marker);
    if (!candidates_pos) {
        pr_debug("extract_gemini_content: Failed to find candidates array\n");
        result = -EINVAL;
        goto cleanup;
    }

    /* Find the parts array in the first candidate */
    parts_pos = strstr(candidates_pos, parts_marker);
    if (!parts_pos) {
        pr_debug("extract_gemini_content: Failed to find parts array\n");
        result = -EINVAL;
        goto cleanup;
    }

    /* Find the text field */
    text_pos = strstr(parts_pos, text_marker);
    if (!text_pos) {
        pr_debug("extract_gemini_content: Failed to find text field\n");
        result = -EINVAL;
        goto cleanup;
    }

    /* Skip past the text marker and opening quote */
    text_pos += strlen(text_marker);

    /* Find the end quote of the content */
    content_end = text_pos;
    while (*content_end) {
        if (escaped) {
            escaped = false;
        } else if (*content_end == '\\') {
            escaped = true;
        } else if (*content_end == '"') {
            break;
        }
        content_end++;
    }

    if (*content_end != '"') {
        pr_debug("extract_gemini_content: Failed to find end of content\n");
        result = -EINVAL;
        goto cleanup;
    }

    /* Copy and unescape content */
    for (i = 0; text_pos + i < content_end && out_idx < output_size - 1; i++) {
        if (text_pos[i] == '\\' && i + 1 < (content_end - text_pos)) {
            i++;
            switch (text_pos[i]) {
                case 'n': output[out_idx++] = '\n'; break;
                case 'r': output[out_idx++] = '\r'; break;
                case 't': output[out_idx++] = '\t'; break;
                case 'b': output[out_idx++] = '\b'; break;
                case 'f': output[out_idx++] = '\f'; break;
                case '"': output[out_idx++] = '"'; break;
                case '\\': output[out_idx++] = '\\'; break;
                case 'u': /* Unicode escape - simplified handling */
                    output[out_idx++] = '?'; /* Placeholder for Unicode */
                    i += 4; /* Skip past the 4 hex digits */
                    break;
                default: output[out_idx++] = text_pos[i]; break;
            }
        } else {
            output[out_idx++] = text_pos[i];
        }
    }

    output[out_idx] = '\0';
    result = out_idx;

cleanup:
    if (decoded_json)
        kfree(decoded_json);

    return result;
}
/* Fix 2: Improved JSON parsing robustness in extract_response_content() in llm_json_manager.c */
int extract_response_content(const char *json, char *output, size_t output_size)
{
    const char *content_start = NULL;
    const char *content_end = NULL;
    int json_depth = 0;
    bool in_string = false;
    bool escaped = false;
    unsigned long flags;
    int i;
    size_t content_len, out_idx;
    bool found_content_field = false;

    /* Update statistics */
    atomic_inc(&json_parse_attempts);

    if (!json_manager_initialized) {
        pr_warn("extract_response_content: JSON manager not initialized\n");
        return -EAGAIN;
    }

    if (!json || !output || output_size == 0)
        return -EINVAL;

    /* Ensure output buffer is clear */
    output[0] = '\0';

    /* Initial JSON structural validation */
    if (!validate_json(json)) {
        pr_warn("extract_response_content: Invalid JSON format\n");
        return -EINVAL;
    }

    /* First handle empty or null responses */
    if (strstr(json, "\"content\":null") != NULL) {
        pr_debug("extract_response_content: Null content field found\n");
        return 0; /* Empty but valid response */
    }

    /* Try different field patterns with case-insensitive search */
    for (i = 0; i < NUM_CONTENT_PATTERNS; i++) {
        const char *found = json_strcasestr(json, content_field_patterns[i]);
        if (found) {
            /* Skip field name and look for opening quote */
            const char *quote = strchr(found + strlen(content_field_patterns[i]), '"');
            if (quote) {
                content_start = quote + 1; /* Skip opening quote */
                found_content_field = true;
                break;
            }
        }
    }

    /* Try alternate JSON structures if not found */
    if (!found_content_field) {
        /* Check for direct message structure: {"message":"content"} */
        const char *direct_msg = strstr(json, "\"message\":");
        if (direct_msg) {
            const char *quote = strchr(direct_msg + 10, '"');
            if (quote) {
                content_start = quote + 1;
                found_content_field = true;
            }
        }
    }

    if (!found_content_field) {
        /* Last attempt: Try to extract from assistant message in array */
        const char *assistant_msg = strstr(json, "\"role\":\"assistant\"");
        if (assistant_msg) {
            const char *content_field = strstr(assistant_msg, "\"content\":");
            if (content_field) {
                const char *quote = strchr(content_field + 10, '"');
                if (quote) {
                    content_start = quote + 1;
                    found_content_field = true;
                }
            }
        }
    }

    if (!found_content_field) {
        /* No recognizable format found */
        pr_debug("extract_response_content: No content field found in JSON: %.100s...\n", json);
        return -EINVAL;
    }

    /* Find end of content by properly handling nested structures */
    content_end = content_start;
    while (*content_end) {
        if (escaped) {
            escaped = false;
            content_end++;
            continue;
        }

        if (*content_end == '\\') {
            escaped = true;
            content_end++;
            continue;
        }

        if (*content_end == '"' && !escaped) {
            /* If we're at the top level and this is the closing quote */
            if (json_depth == 0) {
                break;
            }
            in_string = !in_string;
        } else if (!in_string) {
            if (*content_end == '{') json_depth++;
            else if (*content_end == '}') json_depth--;
        }

        content_end++;
    }

    if (*content_end != '"') {
        pr_debug("extract_response_content: No closing quote found for content\n");
        return -EINVAL; /* No closing quote found */
    }

    /* Calculate content length and ensure it fits in output buffer */
    content_len = content_end - content_start;
    if (content_len >= output_size)
        content_len = output_size - 1;

    /* Copy and unescape the content */
    out_idx = 0;
    for (i = 0; i < content_len && out_idx < output_size - 1; i++) {
        if (content_start[i] == '\\' && i + 1 < content_len) {
            i++;
            switch (content_start[i]) {
                case 'n': output[out_idx++] = '\n'; break;
                case 'r': output[out_idx++] = '\r'; break;
                case 't': output[out_idx++] = '\t'; break;
                case 'b': output[out_idx++] = '\b'; break;
                case 'f': output[out_idx++] = '\f'; break;
                case '\\': output[out_idx++] = '\\'; break;
                case '"': output[out_idx++] = '"'; break;
                case 'u': /* Unicode escape */
                    if (i + 4 < content_len) {
                        char unicode_buf[JSON_UNICODE_MAX_LEN]; /* Maximum 4 bytes UTF-8 + null */
                        int unicode_len;

                        /* Extract the 4 hex digits */
                        char hex[5] = {0};
                        strncpy(hex, content_start + i + 1, 4);
                        hex[4] = '\0';
                        i += 4; /* Skip the 4 hex digits */

                        /* Convert to UTF-8 */
                        unicode_len = decode_unicode_escape(hex, unicode_buf, sizeof(unicode_buf));
                        if (unicode_len > 0 && out_idx + unicode_len < output_size - 1) {
                            memcpy(output + out_idx, unicode_buf, unicode_len);
                            out_idx += unicode_len;
                        } else {
                            /* Fallback for conversion errors */
                            output[out_idx++] = '?';
                        }
                    } else {
                        /* Not enough characters left */
                        output[out_idx++] = '?';
                        i = content_len; /* Exit loop */
                    }
                    break;
                default:
                    /* Unrecognized escape, just output the character */
                    if (out_idx < output_size - 1) {
                        output[out_idx++] = content_start[i];
                    }
                    break;
            }
        } else {
            /* Normal character */
            if (out_idx < output_size - 1) {
                output[out_idx++] = content_start[i];
            }
        }
    }

    output[out_idx] = '\0';

    /* Update statistics */
    spin_lock_irqsave(&json_lock, flags);
    atomic_inc(&json_parse_successes);
    spin_unlock_irqrestore(&json_lock, flags);

    return out_idx;
}

/* Fix 3: Implement extract_response_content_improved to match header declaration */
int extract_response_content_improved(const char *json, char *output, size_t output_size)
{
    int result;

    /* First try standard extraction */
    result = extract_response_content(json, output, output_size);
    if (result > 0) {
        return result;
    }

    /* If the standard extraction failed, try more aggressive parsing */
    if (json && output && output_size > 0) {
        /* Last resort: just look for any content between quotes */
        const char *start = strstr(json, "\":");
        if (start) {
            const char *content_start = NULL;
            const char *content_end = NULL;
            size_t copy_len;

            /* Find next quote after the colon */
            start = strchr(start + 2, '"');
            if (start) {
                content_start = start + 1;
                content_end = strchr(content_start, '"');

                if (content_end && content_end > content_start) {
                    /* Found something between quotes, copy it */
                    copy_len = content_end - content_start;
                    if (copy_len >= output_size)
                        copy_len = output_size - 1;

                    memcpy(output, content_start, copy_len);
                    output[copy_len] = '\0';

                    pr_warn("extract_response_content_improved: Using fallback parser\n");
                    return copy_len;
                }
            }
        }

        /* If all else fails, return a helpful message */
        strscpy(output, "[Failed to parse response]", output_size);
        return strlen(output);
    }

    return result;
}
/*
 * Parse token count from JSON response with improved robustness
 * Returns 0 on success, negative error on failure
 */
int parse_token_count(const char *json, int *prompt_tokens,
                      int *completion_tokens, int *total_tokens)
{
    const char *usage_start = NULL;
    int ret_prompt = -EINVAL;
    int ret_completion = -EINVAL;
    int ret_total = -EINVAL;
    int i;

    if (!json_manager_initialized)
        return -EAGAIN;

    if (!json || !prompt_tokens || !completion_tokens || !total_tokens)
        return -EINVAL;

    /* Initialize outputs */
    *prompt_tokens = 0;
    *completion_tokens = 0;
    *total_tokens = 0;

    /* Find usage section with case-insensitive search */
    for (i = 0; i < NUM_USAGE_SECTION_FIELDS; i++) {
        usage_start = json_strcasestr(json, usage_section_fields[i]);
        if (usage_start)
            break;
    }

    if (!usage_start)
        return -EINVAL;

    /* Parse prompt_tokens with multiple field name variants */
    for (i = 0; i < NUM_PROMPT_TOKEN_FIELDS && ret_prompt != 0; i++) {
        ret_prompt = extract_number_after_field(usage_start, prompt_token_fields[i], prompt_tokens);
    }

    /* Parse completion_tokens with multiple field name variants */
    for (i = 0; i < NUM_COMPLETION_TOKEN_FIELDS && ret_completion != 0; i++) {
        ret_completion = extract_number_after_field(usage_start, completion_token_fields[i], completion_tokens);
    }

    /* Parse total_tokens */
    ret_total = extract_number_after_field(usage_start, "total_tokens", total_tokens);

    /* Calculate total if not provided but we have prompt and completion */
    if (ret_total != 0 && ret_prompt == 0 && ret_completion == 0) {
        *total_tokens = *prompt_tokens + *completion_tokens;
        ret_total = 0;
    }

    /* Only return success if we found at least one valid token count */
    if (ret_prompt == 0 || ret_completion == 0 || ret_total == 0) {
        return 0;
    }

    return -EINVAL;
}

/* Get statistics for monitoring */
void json_get_stats(int *buffers_created_count, int *buffers_resized_count,
                    int *parse_attempts_count, int *parse_successes_count)
{
    if (buffers_created_count)
        *buffers_created_count = atomic_read(&buffers_created);

    if (buffers_resized_count)
        *buffers_resized_count = atomic_read(&buffers_resized);

    if (parse_attempts_count)
        *parse_attempts_count = atomic_read(&json_parse_attempts);

    if (parse_successes_count)
        *parse_successes_count = atomic_read(&json_parse_successes);
}

/* Show statistics via sysfs */
ssize_t json_stats_show(struct device *dev, struct device_attribute *attr, char *buf)
{
    int created, resized, attempts, successes;
    int success_rate;

    created = atomic_read(&buffers_created);
    resized = atomic_read(&buffers_resized);
    attempts = atomic_read(&json_parse_attempts);
    successes = atomic_read(&json_parse_successes);

    /* Calculate success rate with division-by-zero protection */
    success_rate = attempts > 0 ? (successes * 100) / attempts : 0;

    return scnprintf(buf, PAGE_SIZE,
                     "JSON Manager Statistics:\n"
                     "  Buffers Created: %d\n"
                     "  Buffers Resized: %d\n"
                     "  Parse Attempts: %d\n"
                     "  Parse Successes: %d\n"
                     "  Parse Success Rate: %d%%\n"
                     "  System Initialized: %s\n",
                     created,
                     resized,
                     attempts,
                     successes,
                     success_rate,
                     json_manager_initialized ? "Yes" : "No");
}

/* Module exports */
EXPORT_SYMBOL(json_manager_init);
EXPORT_SYMBOL(json_manager_cleanup);
EXPORT_SYMBOL(json_manager_initialized_check);
EXPORT_SYMBOL(json_buffer_init);
EXPORT_SYMBOL(json_buffer_resize);
EXPORT_SYMBOL(json_buffer_free);
EXPORT_SYMBOL(append_json_string);
EXPORT_SYMBOL(append_json_value);
EXPORT_SYMBOL(append_json_number);
EXPORT_SYMBOL(append_json_float);
EXPORT_SYMBOL(append_json_boolean);
EXPORT_SYMBOL(extract_response_content);
EXPORT_SYMBOL(extract_openai_content);
EXPORT_SYMBOL(extract_anthropic_content);
EXPORT_SYMBOL(extract_gemini_content);
EXPORT_SYMBOL(parse_token_count);
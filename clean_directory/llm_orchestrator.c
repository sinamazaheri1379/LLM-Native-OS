#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/net.h>
#include <linux/in.h>
#include <linux/socket.h>
#include <linux/kvec.h>
#include <net/sock.h>
#include <linux/jiffies.h>
#include <linux/timer.h>
#include <linux/random.h>
#include <linux/delay.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include <linux/sched.h>
#include <linux/version.h>
#include "llm_orchestrator.h"

#define MODULE_NAME "llm_orchestrator"
#define DRIVER_VERSION "2.0"

/* Module parameters for API keys */
static char *openai_api_key = NULL;
module_param(openai_api_key, charp, 0600);
MODULE_PARM_DESC(openai_api_key, "OpenAI API Key");

static char *anthropic_api_key = NULL;
module_param(anthropic_api_key, charp, 0600);
MODULE_PARM_DESC(anthropic_api_key, "Anthropic API Key");

static char *google_gemini_api_key = NULL;
module_param(google_gemini_api_key, charp, 0600);
MODULE_PARM_DESC(google_gemini_api_key, "Google Gemini API Key");

/* Auto-pruning parameter for old conversations */
static int prune_threshold_mins = 60; /* 1 hour */
module_param(prune_threshold_mins, int, 0600);
MODULE_PARM_DESC(prune_threshold_mins, "Auto-prune threshold for old conversations in minutes (0 to disable)");

/* Character device globals */
static int major_number;
static struct class *orchestrator_class;
static struct device *orchestrator_device;
static struct cdev orchestrator_cdev;
static DEFINE_MUTEX(orchestrator_mutex);

/* Global state */
static struct scheduler_state global_scheduler;
static struct llm_response global_response;

/* Maintenance timer */
static struct timer_list maintenance_timer;

/* --- TLS and Network Connection Functions --- */

/*
 * setup_tls - Stub TLS setup function.
 * In a production system, perform the TLS handshake using kernel crypto APIs.
 */
static int setup_tls(struct socket *sock)
{
    pr_debug("setup_tls: TLS handshake stubbed (assumed successful)\n");
    return 0;
}

/*
 * establish_connection - Creates and connects a kernel socket to the given host.
 * If use_tls is true, calls setup_tls().
 */
static int establish_connection(struct socket **sock, const char *host_ip,
                                int port, bool use_tls)
{
    struct socket *s;
    struct sockaddr_in server_addr;
    int ret;

    /* Create socket */
    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM, IPPROTO_TCP, &s);
    if (ret < 0) {
        pr_err("establish_connection: sock_create_kern failed: %d\n", ret);
        return ret;
    }

    /* Initialize server address */
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    ret = in4_pton(host_ip, -1, (u8 *)&server_addr.sin_addr.s_addr, -1, NULL);
    if (!ret) {
        pr_err("establish_connection: Invalid server IP: %s\n", host_ip);
        sock_release(s);
        return -EINVAL;
    }

    /* Connect to server */
    ret = kernel_connect(s, (struct sockaddr *)&server_addr, sizeof(server_addr), 0);
    if (ret < 0) {
        pr_err("establish_connection: kernel_connect failed: %d to %s:%d\n", 
               ret, host_ip, port);
        sock_release(s);
        return ret;
    }

    /* Setup TLS if required */
    if (use_tls) {
        ret = setup_tls(s);
        if (ret < 0) {
            pr_err("establish_connection: TLS setup failed: %d\n", ret);
            sock_release(s);
            return ret;
        }
    }

    *sock = s;
    return 0;
}

/* --- HTTP Request Sending and Receiving --- */

#define MAX_HEADER_SIZE 1024
#define MAX_RESPONSE_SIZE (MAX_RESPONSE_LENGTH + 1024) /* Response buffer plus headers */

/*
 * network_send_request - Sends an HTTP POST request and receives response.
 * @host_ip: Server hostname (e.g., "api.openai.com")
 * @port: Server port (usually 443 for HTTPS)
 * @http_path: The path of the API endpoint (e.g., "/v1/chat/completions")
 * @api_key: API key for authentication (used in header)
 * @auth_header: Auth header format (e.g., "Authorization: Bearer", "x-api-key: ")
 * @use_tls: Whether to use TLS (true for HTTPS)
 * @timeout_ms: Request timeout in milliseconds
 * @buf: JSON buffer with request body
 * @resp: Structure to store the received response
 *
 * Returns 0 on success, negative error code on failure.
 */
static int network_send_request(const char *host_ip, int port,
                                const char *http_path,
                                const char *api_key,
                                const char *auth_header,
                                bool use_tls,
                                unsigned long timeout_ms,
                                struct llm_json_buffer *buf,
                                struct llm_response *resp)
{
    struct socket *sock;
    struct msghdr msg = {0};
    struct kvec iov[2];
    char headers[MAX_HEADER_SIZE];
    int ret, header_len, total_len;
    char *recv_buf;
    int received = 0;
    long timeout = msecs_to_jiffies(timeout_ms);
    ktime_t start_time, end_time;
    s64 elapsed_ms;
    bool rate_limited = false;
    unsigned long reset_time_ms = 0;

    /* Record start time */
    start_time = ktime_get();

    /* Establish connection */
    ret = establish_connection(&sock, host_ip, port, use_tls);
    if (ret < 0)
        return ret;

    /* Prepare HTTP headers */
    header_len = snprintf(headers, sizeof(headers),
        "POST %s HTTP/1.1\r\n"
        "Host: %s\r\n"
        "%s%s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        http_path, host_ip,
        auth_header, api_key,
        buf->used);
        
    if (header_len < 0 || header_len >= sizeof(headers)) {
        ret = -EOVERFLOW;
        goto release_sock;
    }

    /* Prepare message vectors */
    total_len = header_len + buf->used;
    iov[0].iov_base = headers;
    iov[0].iov_len = header_len;
    iov[1].iov_base = buf->data;
    iov[1].iov_len = buf->used;

    /* Set socket timeout using struct timeval */
    {
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ret = sock_setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                              (char *)&tv, sizeof(tv));
        if (ret < 0) {
            pr_err("network_send_request: Failed to set socket timeout: %d\n", ret);
            goto release_sock;
        }
    }
    if (ret < 0) {
        pr_err("network_send_request: Failed to set socket timeout: %d\n", ret);
        goto release_sock;
    }

    /* Send request */
    ret = kernel_sendmsg(sock, &msg, iov, 2, total_len);
    if (ret < 0) {
        pr_err("network_send_request: kernel_sendmsg failed: %d\n", ret);
        goto release_sock;
    }

    /* Allocate receive buffer */
    recv_buf = kmalloc(MAX_RESPONSE_SIZE, GFP_KERNEL);
    if (!recv_buf) {
        ret = -ENOMEM;
        goto release_sock;
    }

    /* Receive response */
    while (received < MAX_RESPONSE_SIZE - 1) {
        struct kvec recv_iov;
        recv_iov.iov_base = recv_buf + received;
        recv_iov.iov_len = MAX_RESPONSE_SIZE - received - 1;

        ret = kernel_recvmsg(sock, &msg, &recv_iov, 1, recv_iov.iov_len, 0);
        
        /* Handle errors */
        if (ret < 0) {
            if (ret == -EAGAIN || ret == -EWOULDBLOCK) {
                /* Timeout */
                pr_warn("network_send_request: request timed out\n");
                ret = -ETIMEDOUT;
            } else {
                pr_err("network_send_request: kernel_recvmsg error: %d\n", ret);
            }
            kfree(recv_buf);  // Ensure memory is freed on error path
            goto release_sock;
        }
        
        /* End of data */
        if (ret == 0)
            break;
            
        received += ret;
        
        /* Check for complete response */
        if (strstr(recv_buf, "\r\n\r\n")) {
            /* Find content length if available */
            const char *cl_header = strcasestr(recv_buf, "Content-Length:");
            if (cl_header) {
                int content_length = 0;
                if (sscanf(cl_header, "Content-Length: %d", &content_length) == 1) {
                    /* Check if we've received all the data */
                    const char *body = strstr(recv_buf, "\r\n\r\n");
                    if (body && (received >= ((body + 4) - recv_buf) + content_length)) {
                        break;
                    }
                }
            }
            
            /* If we can't determine content length, look for JSON end marker */
            if (received > 0 && recv_buf[received-1] == '}')
                break;
        }
    }
    recv_buf[received] = '\0';

    /* Extract HTTP status code */
    if (strncmp(recv_buf, "HTTP/1.1 ", 9) == 0) {
        int status_code;
        if (sscanf(recv_buf + 9, "%d", &status_code) == 1) {
            resp->status = status_code;
            
            /* Check for rate limiting response (status 429) */
            if (status_code == 429) {
                rate_limited = true;
                
                /* Try to extract rate limit reset time */
                const char *retry_after = strcasestr(recv_buf, "Retry-After:");
                if (retry_after) {
                    int seconds = 0;
                    if (sscanf(retry_after, "Retry-After: %d", &seconds) == 1) {
                        reset_time_ms = seconds * 1000;
                    }
                } else {
                    /* Default to 60 seconds if not specified */
                    reset_time_ms = 60000;
                }
                
                ret = -LLM_ERR_RATE_LIMIT;
                goto cleanup_buffer;
            }
            
            /* Check for authorization error */
            if (status_code == 401 || status_code == 403) {
                ret = -LLM_ERR_AUTH;
                goto cleanup_buffer;
            }
            
            /* Check for other error status codes */
            if (status_code < 200 || status_code >= 300) {
                ret = -LLM_ERR_API_RESPONSE;
                goto cleanup_buffer;
            }
        }
    }

    /* Extract response body */
    {
        char *body = strstr(recv_buf, "\r\n\r\n");
        if (body) {
            body += 4; /* Skip over the separator */
            
            /* Copy response to output buffer */
            strncpy(resp->content, body, MAX_RESPONSE_LENGTH - 1);
            resp->content[MAX_RESPONSE_LENGTH - 1] = '\0';
            resp->content_length = strlen(resp->content);
            
            /* Parse token counts if available */
            {
                int prompt_tokens = 0, completion_tokens = 0, total_tokens = 0;
                if (parse_token_count(body, &prompt_tokens, &completion_tokens, &total_tokens) == 0) {
                    resp->tokens_used = total_tokens;
                }
            }
        } else {
            /* No body separator found */
            strncpy(resp->content, recv_buf, MAX_RESPONSE_LENGTH - 1);
            resp->content[MAX_RESPONSE_LENGTH - 1] = '\0';
            resp->content_length = strlen(resp->content);
        }
    }
    
    ret = 0;

cleanup_buffer:
    kfree(recv_buf);

release_sock:
    /* Close socket */
    sock_release(sock);
    
    /* Calculate latency */
    end_time = ktime_get();
    elapsed_ms = ktime_to_ms(ktime_sub(end_time, start_time));
    resp->latency_ms = elapsed_ms;
    
    /* Handle rate limiting */
    if (rate_limited && reset_time_ms > 0) {
        int provider = resp->provider_used;
        if (provider >= 0 && provider < PROVIDER_COUNT) {
            handle_rate_limit(provider, &global_scheduler, reset_time_ms);
        }
    }
    
    return ret;
}

/* --- Provider Functions --- */

/*
 * Format OpenAI request JSON with context
 */
static int format_openai_request(struct llm_request *req, struct llm_json_buffer *json_buf)
{
    struct llm_json_buffer context_buf;
    int ret;
    const char *model;
    
    /* Initialize JSON buffer */
    json_buf->used = 0;
    
    /* Set model name */
    if (req->model_name[0] != '\0' && is_model_supported(PROVIDER_OPENAI, req->model_name)) {
        model = req->model_name;
    } else {
        model = get_default_model(PROVIDER_OPENAI);
    }
    
    /* If conversation exists, get context */
    if (req->conversation_id > 0) {
        ret = json_buffer_init(&context_buf, MAX_PAYLOAD_SIZE);
        if (ret < 0)
            return ret;
            
        ret = context_get_conversation(req->conversation_id, &context_buf);
        if (ret == 0 && context_buf.used > 0) {
            /* Conversation found, use it for messages array */
            ret = append_json_string(json_buf, "{\"model\": \"");
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, model);
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, "\", \"messages\": ");
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, context_buf.data);
            if (ret)
                goto cleanup;
                
            /* Add current message */
            ret = append_json_string(json_buf, ", {\"role\": \"");
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, req->role);
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, "\", \"content\": \"");
            if (ret)
                goto cleanup;
                
            ret = append_json_value(json_buf, req->prompt);
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, "\"}]");
            
            /* Add parameters */
            if (!ret) {
                ret = append_json_string(json_buf, ", \"temperature\": ");
                if (!ret) {
                    ret = append_json_float(json_buf, req->temperature_x100 > 0 ? 
                                          req->temperature_x100 : 70);
                    
                    if (!ret) {
                        ret = append_json_string(json_buf, ", \"max_tokens\": ");
                        if (!ret) {
                            ret = append_json_number(json_buf, req->max_tokens > 0 ? 
                                                   req->max_tokens : 500);
                            if (!ret) {
                                ret = append_json_string(json_buf, "}");
                            }
                        }
                    }
                }
            }
            
            if (ret)
                goto cleanup;
                
            json_buffer_free(&context_buf);
            return 0;
        }
        
        if (context_buf.data)
            json_buffer_free(&context_buf);
    }
    
    /* No existing conversation, create a new request format */
    ret = append_json_string(json_buf, "{\"model\": \"");
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, model);
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, 
                          "\", \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"");
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, req->role);
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, "\", \"content\": \"");
    if (ret)
        return ret;
        
    ret = append_json_value(json_buf, req->prompt);
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, "\"}]");
    if (ret)
        return ret;
        
    /* Add parameters */
    ret = append_json_string(json_buf, ", \"temperature\": ");
    if (ret)
        return ret;
        
    ret = append_json_float(json_buf, req->temperature_x100 > 0 ? 
                          req->temperature_x100 : 70);
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, ", \"max_tokens\": ");
    if (ret)
        return ret;
        
    ret = append_json_number(json_buf, req->max_tokens > 0 ? 
                           req->max_tokens : 500);
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, "}");
    return ret;
    
cleanup:
    json_buffer_free(&context_buf);
    return ret;
}

/*
 * llm_send_openai: Uses the OpenAI API.
 * Endpoint: POST https://api.openai.com/v1/chat/completions
 * Header: Authorization: Bearer YOUR_API_KEY
 */
int llm_send_openai(const char *api_key,
                    struct llm_request *req,
                    struct llm_response *resp)
{
    struct llm_json_buffer json_buf;
    int ret;

    if (!api_key || !req || !resp)
        return -EINVAL;
        
    /* Check API key */
    if (strlen(api_key) < 20) {
        pr_err("llm_send_openai: Invalid API key\n");
        return -LLM_ERR_AUTH;
    }

    /* Initialize JSON buffer */
    ret = json_buffer_init(&json_buf, MAX_PAYLOAD_SIZE);
    if (ret < 0)
        return ret;
        
    /* Format request JSON */
    ret = format_openai_request(req, &json_buf);
    if (ret < 0) {
        json_buffer_free(&json_buf);
        return ret;
    }

    /* Set provider info in response */
    resp->provider_used = PROVIDER_OPENAI;
    strncpy(resp->model_used, 
            req->model_name[0] ? req->model_name : get_default_model(PROVIDER_OPENAI),
            MAX_MODEL_NAME);
    resp->timestamp = ktime_get();

    /* Send the request */
    ret = network_send_request("api.openai.com", 443, "/v1/chat/completions",
                             api_key, "Authorization: Bearer ", true, 
                             req->timeout_ms, &json_buf, resp);
    
    /* Store context if request was successful */
    if (ret == 0 && req->conversation_id > 0) {
        /* Add the user's message to context */
        context_add_entry(req->conversation_id, req->role, req->prompt);
        
        /* Extract assistant response */
        {
            char assistant_content[MAX_PROMPT_LENGTH];
            ret = extract_response_content(resp->content, assistant_content, sizeof(assistant_content));
            if (ret > 0) {
                /* Add assistant response to context */
                context_add_entry(req->conversation_id, "assistant", assistant_content);
            }
        }
    }
    
    json_buffer_free(&json_buf);
    return ret;
}

/*
 * Format Anthropic request JSON with context
 */
static int format_anthropic_request(struct llm_request *req, struct llm_json_buffer *json_buf)
{
    struct llm_json_buffer context_buf;
    int ret;
    const char *model;
    
    /* Initialize JSON buffer */
    json_buf->used = 0;
    
    /* Set model name */
    if (req->model_name[0] != '\0' && is_model_supported(PROVIDER_ANTHROPIC, req->model_name)) {
        model = req->model_name;
    } else {
        model = get_default_model(PROVIDER_ANTHROPIC);
    }
    
    /* If conversation exists, get context */
    if (req->conversation_id > 0) {
        ret = json_buffer_init(&context_buf, MAX_PAYLOAD_SIZE);
        if (ret < 0)
            return ret;
            
        ret = context_get_conversation(req->conversation_id, &context_buf);
        if (ret == 0 && context_buf.used > 0) {
            /* Conversation found, use it for messages array */
            ret = append_json_string(json_buf, "{\"model\": \"");
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, model);
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, "\", \"messages\": ");
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, context_buf.data);
            if (ret)
                goto cleanup;
                
            /* Add current message */
            ret = append_json_string(json_buf, ", {\"role\": \"");
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, req->role);
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, "\", \"content\": \"");
            if (ret)
                goto cleanup;
                
            ret = append_json_value(json_buf, req->prompt);
            if (ret)
                goto cleanup;
                
            ret = append_json_string(json_buf, "\"}]");
            
            /* Add parameters */
            if (!ret) {
                ret = append_json_string(json_buf, ", \"max_tokens\": ");
                if (!ret) {
                    ret = append_json_number(json_buf, req->max_tokens > 0 ? 
                                           req->max_tokens : 1000);
                    
                    if (!ret) {
                        ret = append_json_string(json_buf, ", \"temperature\": ");
                        if (!ret) {
                            ret = append_json_float(json_buf, req->temperature_x100 > 0 ? 
                                                  req->temperature_x100 : 70);
                            if (!ret) {
                                ret = append_json_string(json_buf, "}");
                            }
                        }
                    }
                }
            }
            
            if (ret)
                goto cleanup;
                
            json_buffer_free(&context_buf);
            return 0;
        }
        
        if (context_buf.data)
            json_buffer_free(&context_buf);
    }
    
    /* No existing conversation, create a new request format */
    ret = append_json_string(json_buf, "{\"model\": \"");
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, model);
    if (ret)
        return ret;
        
ret = append_json_string(json_buf, "\", \"messages\": [{\"role\": \"");
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, req->role);
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, "\", \"content\": \"");
    if (ret)
        return ret;
        
    ret = append_json_value(json_buf, req->prompt);
    if (ret)
        return ret;
        
    ret = append_json_string(json_buf, "\"}]");
    
    /* Add parameters */
    if (!ret) {
        ret = append_json_string(json_buf, ", \"max_tokens\": ");
        if (!ret) {
            ret = append_json_number(json_buf, req->max_tokens > 0 ? 
                                   req->max_tokens : 1000);
            
            if (!ret) {
                ret = append_json_string(json_buf, ", \"temperature\": ");
                if (!ret) {
                    ret = append_json_float(json_buf, req->temperature_x100 > 0 ? 
                                          req->temperature_x100 : 70);
                    if (!ret) {
                        ret = append_json_string(json_buf, "}");
                    }
                }
            }
        }
    }
    
    return ret;
    
cleanup:
    json_buffer_free(&context_buf);
    return ret;
}

/*
 * llm_send_anthropic: Uses the Anthropic API.
 * Endpoint: POST https://api.anthropic.com/v1/messages
 * Headers: x-api-key and anthropic-version.
 */
int llm_send_anthropic(const char *api_key,
                       struct llm_request *req,
                       struct llm_response *resp)
{
    struct llm_json_buffer json_buf;
    int ret;
    char auth_header[64];
    
    if (!api_key || !req || !resp)
        return -EINVAL;
        
    /* Check API key */
    if (strlen(api_key) < 20) {
        pr_err("llm_send_anthropic: Invalid API key\n");
        return -LLM_ERR_AUTH;
    }
    
    /* Initialize JSON buffer */
    ret = json_buffer_init(&json_buf, MAX_PAYLOAD_SIZE);
    if (ret < 0)
        return ret;
        
    /* Format request JSON */
    ret = format_anthropic_request(req, &json_buf);
    if (ret < 0) {
        json_buffer_free(&json_buf);
        return ret;
    }

    /* Set provider info in response */
    resp->provider_used = PROVIDER_ANTHROPIC;
    strncpy(resp->model_used, 
            req->model_name[0] ? req->model_name : get_default_model(PROVIDER_ANTHROPIC),
            MAX_MODEL_NAME);
    resp->timestamp = ktime_get();
    
    /* Format auth header */
    snprintf(auth_header, sizeof(auth_header), "x-api-key: %s\r\nanthropic-version: 2023-06-01", api_key);

    /* Send the request */
    ret = network_send_request("api.anthropic.com", 443, "/v1/messages",
                             "", auth_header, true, 
                             req->timeout_ms, &json_buf, resp);
    
    /* Store context if request was successful */
    if (ret == 0 && req->conversation_id > 0) {
        /* Add the user's message to context */
        context_add_entry(req->conversation_id, req->role, req->prompt);
        
        /* Extract assistant response */
        {
            char assistant_content[MAX_PROMPT_LENGTH];
            ret = extract_response_content(resp->content, assistant_content, sizeof(assistant_content));
            if (ret > 0) {
                /* Add assistant response to context */
                context_add_entry(req->conversation_id, "assistant", assistant_content);
            }
        }
    }
    
    json_buffer_free(&json_buf);
    return ret;
}

/*
 * Format Gemini request JSON with context
 */
static int format_gemini_request(struct llm_request *req, struct llm_json_buffer *json_buf)
{
    struct llm_json_buffer context_buf;
    int ret;
    const char *model;
    
    /* Initialize JSON buffer */
    json_buf->used = 0;
    
    /* Set model name */
    if (req->model_name[0] != '\0' && is_model_supported(PROVIDER_GOOGLE_GEMINI, req->model_name)) {
        model = req->model_name;
    } else {
        model = get_default_model(PROVIDER_GOOGLE_GEMINI);
    }
    
    /* Start building request */
    ret = append_json_string(json_buf, "{\"contents\": [");
    if (ret)
        return ret;
    
    /* If conversation exists, add context messages */
    if (req->conversation_id > 0) {
        int entry_count = context_get_entry_count(req->conversation_id);
        
        /* Only add context if we have entries */
        if (entry_count > 0) {
            /* For Gemini, we need to convert the context to a summary since
             * it doesn't support the same messages format as OpenAI/Anthropic */
            char context_summary[MAX_PROMPT_LENGTH];
            snprintf(context_summary, sizeof(context_summary),
                "This is a continuation of a conversation. " 
                "Previous context: (conversation ID %d with %d messages). "
                "Continue from here with the following message: %s",
                req->conversation_id, entry_count, req->prompt);
            
            /* Add special context message */
            ret = append_json_string(json_buf, "{\"role\": \"user\", \"parts\": [{\"text\": \"");
            if (ret)
                return ret;
                
            ret = append_json_value(json_buf, context_summary);
            if (ret)
                return ret;
                
            ret = append_json_string(json_buf, "\"}]}");
            if (ret)
                return ret;
        } else {
            /* No existing conversation, add current message */
            ret = append_json_string(json_buf, "{\"role\": \"user\", \"parts\": [{\"text\": \"");
            if (ret)
                return ret;
                
            ret = append_json_value(json_buf, req->prompt);
            if (ret)
                return ret;
                
            ret = append_json_string(json_buf, "\"}]}");
            if (ret)
                return ret;
        }
    } else {
        /* No conversation ID, simple request */
        ret = append_json_string(json_buf, "{\"role\": \"user\", \"parts\": [{\"text\": \"");
        if (ret)
            return ret;
            
        ret = append_json_value(json_buf, req->prompt);
        if (ret)
            return ret;
            
        ret = append_json_string(json_buf, "\"}]}");
        if (ret)
            return ret;
    }
    
    /* Close contents array and add generation config */
    ret = append_json_string(json_buf, "], \"generationConfig\": {");
    if (ret)
        return ret;
    
    /* Add max tokens parameter */
    ret = append_json_string(json_buf, "\"maxOutputTokens\": ");
    if (ret)
        return ret;
        
    ret = append_json_number(json_buf, req->max_tokens > 0 ? req->max_tokens : 1000);
    if (ret)
        return ret;
    
    /* Add temperature parameter */
    ret = append_json_string(json_buf, ", \"temperature\": ");
    if (ret)
        return ret;
        
    ret = append_json_float(json_buf, req->temperature_x100 > 0 ? req->temperature_x100 : 70);
    if (ret)
        return ret;
    
    /* Add topP parameter (fixed value) */
    ret = append_json_string(json_buf, ", \"topP\": 0.95");
    if (ret)
        return ret;
    
    /* Close generationConfig and request */
    ret = append_json_string(json_buf, "}}");
    
    return ret;
}

/*
 * llm_send_google_gemini: Uses the Google Gemini API.
 * Endpoint: POST https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key=YOUR_API_KEY
 */
int llm_send_google_gemini(const char *api_key,
                           struct llm_request *req,
                           struct llm_response *resp)
{
    struct llm_json_buffer json_buf;
    int ret;
    char http_path[512];
    const char *model_name;
    
    if (!api_key || !req || !resp)
        return -EINVAL;
        
    /* Check API key */
    if (strlen(api_key) < 10) {
        pr_err("llm_send_google_gemini: Invalid API key\n");
        return -LLM_ERR_AUTH;
    }
    
    /* Get model name */
    if (req->model_name[0] != '\0' && is_model_supported(PROVIDER_GOOGLE_GEMINI, req->model_name)) {
        model_name = req->model_name;
    } else {
        model_name = get_default_model(PROVIDER_GOOGLE_GEMINI);
    }
    
    /* Initialize JSON buffer */
    ret = json_buffer_init(&json_buf, MAX_PAYLOAD_SIZE);
    if (ret < 0)
        return ret;
        
    /* Format request JSON */
    ret = format_gemini_request(req, &json_buf);
    if (ret < 0) {
        json_buffer_free(&json_buf);
        return ret;
    }

    /* Set provider info in response */
    resp->provider_used = PROVIDER_GOOGLE_GEMINI;
    strncpy(resp->model_used, model_name, MAX_MODEL_NAME);
    resp->timestamp = ktime_get();
    
    /* Format HTTP path with API key and model */
    snprintf(http_path, sizeof(http_path),
             "/v1/models/%s:generateContent?key=%s", model_name, api_key);

    /* Send the request */
    ret = network_send_request("generativelanguage.googleapis.com", 443,
                             http_path, "", "", true,
                             req->timeout_ms, &json_buf, resp);
    
    /* Store context if request was successful */
    if (ret == 0 && req->conversation_id > 0) {
        /* Add the user's message to context */
        context_add_entry(req->conversation_id, req->role, req->prompt);
        
        /* Extract assistant response */
        {
            char assistant_content[MAX_PROMPT_LENGTH];
            ret = extract_response_content(resp->content, assistant_content, sizeof(assistant_content));
            if (ret > 0) {
                /* Add assistant response to context */
                context_add_entry(req->conversation_id, "assistant", assistant_content);
            }
        }
    }
    
    json_buffer_free(&json_buf);
    return ret;
}

/* --- Orchestration Function --- */

/*
 * orchestrate_request - Uses scheduler to select a provider and falls back if necessary.
 */
static int orchestrate_request(struct llm_request *req, struct llm_response *resp)
{
    int selected_provider, ret = -EINVAL;
    int i;

    /* Set default values if necessary */
    req->timeout_ms = (req->timeout_ms > 0) ? req->timeout_ms : 30000;
    if (req->temperature_x100 <= 0)
        req->temperature_x100 = 70;

    /* Use scheduler to select the initial provider */
    selected_provider = select_provider(req, &global_scheduler);
    if (selected_provider < 0)
        return selected_provider;

    /* Try each provider in sequence starting from the selected one */
    for (i = 0; i < PROVIDER_COUNT; i++) {
        int provider = (selected_provider + i) % PROVIDER_COUNT;
        switch (provider) {
            case PROVIDER_OPENAI:
                ret = llm_send_openai(openai_api_key, req, resp);
                break;
            case PROVIDER_ANTHROPIC:
                ret = llm_send_anthropic(anthropic_api_key, req, resp);
                break;
            case PROVIDER_GOOGLE_GEMINI:
                ret = llm_send_google_gemini(google_gemini_api_key, req, resp);
                break;
            default:
                ret = -EINVAL;
                break;
        }
        update_provider_metrics(provider, ret, resp->latency_ms, resp->tokens_used);
        if (ret == 0)
            break;  /* Success */
    }
    return ret;
}

/* --- Maintenance Timer Functions --- */

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 15, 0)
static void maintenance_timer_callback(struct timer_list *t)
#else
static void maintenance_timer_callback(unsigned long data)
#endif
{
    /* Prune old conversations */
    if (prune_threshold_mins > 0) {
        context_prune_old_conversations(prune_threshold_mins * 60 * 1000);
    }
    
    /* Reset the timer */
    mod_timer(&maintenance_timer, jiffies + HZ * 60 * 10);  /* 10 minutes */
}

/* --- Character Device File Operations --- */
static int orchestrator_open(struct inode *inode, struct file *file)
{
    /* Store scheduler state pointer in task struct */
    return 0;
}

static int orchestrator_release(struct inode *inode, struct file *file)
{
    return 0;
}

static ssize_t orchestrator_write(struct file *file,
                                  const char __user *buf,
                                  size_t count,
                                  loff_t *offset)
{
    struct llm_request req;
    int ret;

    if (count != sizeof(struct llm_request))
        return -EINVAL;
    if (copy_from_user(&req, buf, sizeof(req)))
        return -EFAULT;

    /* Validate request */
    if (req.timeout_ms > 300000) {  /* Max 5 minutes timeout */
        req.timeout_ms = 300000;
    } else if (req.timeout_ms <= 0) {
        req.timeout_ms = 30000;  /* Default 30 seconds */
    }
    
    /* Role should be valid */
    if (req.role[0] == '\0') {
        strcpy(req.role, "user");  /* Default role */
    }

    mutex_lock(&orchestrator_mutex);
    ret = orchestrate_request(&req, &global_response);
    mutex_unlock(&orchestrator_mutex);
    
    return ret < 0 ? ret : count;
}

static ssize_t orchestrator_read(struct file *file,
                                 char __user *buf,
                                 size_t count,
                                 loff_t *offset)
{
    ssize_t ret;
    mutex_lock(&orchestrator_mutex);
    if (global_response.content_length == 0) {
        mutex_unlock(&orchestrator_mutex);
        return 0;
    }
    if (count < global_response.content_length) {
        mutex_unlock(&orchestrator_mutex);
        return -EINVAL;
    }
    if (copy_to_user(buf, global_response.content, global_response.content_length)) {
        mutex_unlock(&orchestrator_mutex);
        return -EFAULT;
    }
    ret = global_response.content_length;
    global_response.content_length = 0;
    mutex_unlock(&orchestrator_mutex);
    return ret;
}

/* Sysfs interfaces for scheduler metrics and configuration */

/* Show current scheduler algorithm */
static ssize_t scheduler_algorithm_show(struct device *dev,
                                       struct device_attribute *attr,
                                       char *buf)
{
    const char *algorithm_names[] = {
        "Round Robin",
        "Weighted",
        "Priority",
        "Performance",
        "Cost Aware",
        "Fallback",
        "FIFO"
    };
    int algorithm = atomic_read(&global_scheduler.current_algorithm);
    
    if (algorithm < 0 || algorithm > SCHEDULER_FIFO)
        return sprintf(buf, "Unknown (%d)\n", algorithm);
        
    return sprintf(buf, "%d (%s)\n", algorithm, algorithm_names[algorithm]);
}

/* Set scheduler algorithm */
static ssize_t scheduler_algorithm_store(struct device *dev,
                                        struct device_attribute *attr,
                                        const char *buf, size_t count)
{
    int algorithm;
    
    if (kstrtoint(buf, 0, &algorithm) < 0)
        return -EINVAL;
        
    if (algorithm < 0 || algorithm > SCHEDULER_FIFO)
        return -EINVAL;
        
    atomic_set(&global_scheduler.current_algorithm, algorithm);
    pr_info("Scheduler algorithm set to %d\n", algorithm);
    
    return count;
}

/* Show provider metrics */
static ssize_t provider_metrics_show(struct device *dev,
                                    struct device_attribute *attr,
                                    char *buf)
{
    const char *provider_names[] = {
        "OpenAI",
        "Anthropic",
        "Google Gemini"
    };
    int i;
    ssize_t len = 0;
    unsigned long avg_latency;
    
    for (i = 0; i < PROVIDER_COUNT; i++) {
        struct provider_metrics *m = &global_scheduler.metrics[i];
        int successful = atomic_read(&m->successful_requests);
        int total = atomic_read(&m->total_requests);
        
        /* Calculate average latency */
        if (successful > 0) {
            avg_latency = atomic64_read(&m->total_latency_ms) / successful;
        } else {
            avg_latency = 0;
        }
        
        /* Add provider stats to output */
        len += sprintf(buf + len, 
                      "Provider: %s\n"
                      "  Status: %d\n"
                      "  Total Requests: %d\n"
                      "  Successful: %d\n"
                      "  Failed: %d\n"
                      "  Timeouts: %d\n"
                      "  Rate Limited: %d\n"
                      "  Avg Latency: %lu ms\n"
                      "  Min Latency: %lu ms\n"
                      "  Max Latency: %lu ms\n"
                      "  Success Rate: %d%%\n"
                      "  Total Tokens: %d\n\n",
                      provider_names[i],
                      atomic_read(&m->current_status),
                      total,
                      successful,
                      atomic_read(&m->failed_requests),
                      atomic_read(&m->timeouts),
                      atomic_read(&m->rate_limited),
                      avg_latency,
                      m->min_latency_ms == ULONG_MAX ? 0 : m->min_latency,
                      m->max_latency_ms,
                      total > 0 ? (successful * 100) / total : 0,
                      atomic_read(&m->total_tokens));
    }
    
    /* Add scheduler info */
    len += sprintf(buf + len,
                  "Scheduler Configuration:\n"
                  "  Algorithm: %d\n"
                  "  Weights: OpenAI=%d%%, Anthropic=%d%%, Gemini=%d%%\n"
                  "  Auto-adjust: %s\n",
                  atomic_read(&global_scheduler.current_algorithm),
                  global_scheduler.weights[PROVIDER_OPENAI],
                  global_scheduler.weights[PROVIDER_ANTHROPIC],
                  global_scheduler.weights[PROVIDER_GOOGLE_GEMINI],
                  global_scheduler.auto_adjust ? "enabled" : "disabled");
    
    return len;
}

/* Reset provider metrics */
static ssize_t reset_metrics_store(struct device *dev,
                                  struct device_attribute *attr,
                                  const char *buf, size_t count)
{
    int reset;
    
    if (kstrtoint(buf, 0, &reset) < 0)
        return -EINVAL;
        
    if (reset == 1) {
        scheduler_reset_metrics(&global_scheduler);
        pr_info("Scheduler metrics reset\n");
    }
        
    return count;
}

/* Show configured weights for weighted scheduler */
static ssize_t scheduler_weights_show(struct device *dev,
                                    struct device_attribute *attr,
                                    char *buf)
{
    return sprintf(buf, 
                 "OpenAI: %d%%\n"
                 "Anthropic: %d%%\n"
                 "Google Gemini: %d%%\n",
                 global_scheduler.weights[PROVIDER_OPENAI],
                 global_scheduler.weights[PROVIDER_ANTHROPIC],
                 global_scheduler.weights[PROVIDER_GOOGLE_GEMINI]);
}

/* Configure weights for weighted scheduler */
static ssize_t scheduler_weights_store(struct device *dev,
                                     struct device_attribute *attr,
                                     const char *buf, size_t count)
{
    int openai, anthropic, gemini;
    
    /* Parse comma-separated weights */
    if (sscanf(buf, "%d,%d,%d", &openai, &anthropic, &gemini) != 3)
        return -EINVAL;
        
    /* Validate weights */
    if (openai < 0 || anthropic < 0 || gemini < 0)
        return -EINVAL;
        
    if (openai + anthropic + gemini != 100)
        return -EINVAL;
        
    /* Update weights */
    global_scheduler.weights[PROVIDER_OPENAI] = openai;
    global_scheduler.weights[PROVIDER_ANTHROPIC] = anthropic;
    global_scheduler.weights[PROVIDER_GOOGLE_GEMINI] = gemini;
    
    pr_info("Scheduler weights updated: OpenAI=%d%%, Anthropic=%d%%, Gemini=%d%%\n",
            openai, anthropic, gemini);
    
    return count;
}

/* Show auto-adjust setting */
static ssize_t auto_adjust_show(struct device *dev,
                               struct device_attribute *attr,
                               char *buf)
{
    return sprintf(buf, "%d\n", global_scheduler.auto_adjust ? 1 : 0);
}

/* Set auto-adjust setting */
static ssize_t auto_adjust_store(struct device *dev,
                                struct device_attribute *attr,
                                const char *buf, size_t count)
{
    int enable;
    
    if (kstrtoint(buf, 0, &enable) < 0)
        return -EINVAL;
        
    global_scheduler.auto_adjust = (enable != 0);
    
    pr_info("Scheduler auto-adjust %s\n", global_scheduler.auto_adjust ? "enabled" : "disabled");
    
    /* If enabling, do an immediate adjustment */
    if (global_scheduler.auto_adjust) {
        adjust_scheduler_weights(&global_scheduler);
    }
    
    return count;
}

/* Show FIFO queue status */
static ssize_t fifo_status_show(struct device *dev,
                               struct device_attribute *attr,
                               char *buf)
{
    unsigned long flags;
    int count;
    
    spin_lock_irqsave(&global_scheduler.fifo.lock, flags);
    count = global_scheduler.fifo.count;
    spin_unlock_irqrestore(&global_scheduler.fifo.lock, flags);
    
    return sprintf(buf, "FIFO queue size: %d/%d\n", count, MAX_FIFO_QUEUE_SIZE);
}

/* Show context status */
static ssize_t context_status_show(struct device *dev,
                                  struct device_attribute *attr,
                                  char *buf)
{
    return sprintf(buf,
                 "Context management:\n"
                 "  Auto-prune threshold: %d minutes\n"
                 "  Max context entries: %d\n",
                 prune_threshold_mins,
                 MAX_CONTEXT_ENTRIES);
}

/* Define sysfs attributes */
static DEVICE_ATTR(scheduler_algorithm, 0644, scheduler_algorithm_show, scheduler_algorithm_store);
static DEVICE_ATTR(provider_metrics, 0444, provider_metrics_show, NULL);
static DEVICE_ATTR(reset_metrics, 0200, NULL, reset_metrics_store);
static DEVICE_ATTR(scheduler_weights, 0644, scheduler_weights_show, scheduler_weights_store);
static DEVICE_ATTR(auto_adjust, 0644, auto_adjust_show, auto_adjust_store);
static DEVICE_ATTR(fifo_status, 0444, fifo_status_show, NULL);
static DEVICE_ATTR(context_status, 0444, context_status_show, NULL);

static struct file_operations orchestrator_fops = {
    .owner = THIS_MODULE,
    .open = orchestrator_open,
    .release = orchestrator_release,
    .write = orchestrator_write,
    .read = orchestrator_read,
};

/* --- Module Initialization and Exit --- */
static int __init orchestrator_init(void)
{
    int ret;
    dev_t dev;
    if (prune_threshold_mins < 0) {
        pr_warn("orchestrator_init: Invalid prune_threshold_mins, setting to default (60)\n");
        prune_threshold_mins = 60;
    }
    pr_info("LLM Orchestrator: Initializing module version %s\n", DRIVER_VERSION);

    /* Initialize scheduler */
    scheduler_init(&global_scheduler);
    set_scheduler_state(&global_scheduler);  // Add this line

    /* Initialize global response */
    memset(&global_response, 0, sizeof(global_response));

    /* Register character device */
    ret = alloc_chrdev_region(&dev, 0, 1, MODULE_NAME);
    if (ret < 0) {
        pr_err("orchestrator_init: Failed to allocate chrdev region\n");
        return ret;
    }
    major_number = MAJOR(dev);

    cdev_init(&orchestrator_cdev, &orchestrator_fops);
    orchestrator_cdev.owner = THIS_MODULE;
    ret = cdev_add(&orchestrator_cdev, dev, 1);
    if (ret < 0) {
        unregister_chrdev_region(dev, 1);
        pr_err("orchestrator_init: Failed to add cdev\n");
        return ret;
    }

    orchestrator_class = class_create(THIS_MODULE, MODULE_NAME);
    if (IS_ERR(orchestrator_class)) {
        cdev_del(&orchestrator_cdev);
        unregister_chrdev_region(dev, 1);
        pr_err("orchestrator_init: Failed to create device class\n");
        return PTR_ERR(orchestrator_class);
    }

    orchestrator_device = device_create(orchestrator_class, NULL, dev, NULL, MODULE_NAME);
    if (IS_ERR(orchestrator_device)) {
        class_destroy(orchestrator_class);
        cdev_del(&orchestrator_cdev);
        unregister_chrdev_region(dev, 1);
        pr_err("orchestrator_init: Failed to create device\n");
        return PTR_ERR(orchestrator_device);
    }

    /* Create sysfs attributes */
    ret = device_create_file(orchestrator_device, &dev_attr_scheduler_algorithm);
    if (ret) {
        pr_warn("orchestrator_init: Failed to create scheduler_algorithm sysfs file\n");
    }
    
    ret = device_create_file(orchestrator_device, &dev_attr_provider_metrics);
    if (ret) {
        pr_warn("orchestrator_init: Failed to create provider_metrics sysfs file\n");
    }
    
    ret = device_create_file(orchestrator_device, &dev_attr_reset_metrics);
    if (ret) {
        pr_warn("orchestrator_init: Failed to create reset_metrics sysfs file\n");
    }
    
    ret = device_create_file(orchestrator_device, &dev_attr_scheduler_weights);
    if (ret) {
        pr_warn("orchestrator_init: Failed to create scheduler_weights sysfs file\n");
    }
    
    ret = device_create_file(orchestrator_device, &dev_attr_auto_adjust);
    if (ret) {
        pr_warn("orchestrator_init: Failed to create auto_adjust sysfs file\n");
    }
    
    ret = device_create_file(orchestrator_device, &dev_attr_fifo_status);
    if (ret) {
        pr_warn("orchestrator_init: Failed to create fifo_status sysfs file\n");
    }
    
    ret = device_create_file(orchestrator_device, &dev_attr_context_status);
    if (ret) {
        pr_warn("orchestrator_init: Failed to create context_status sysfs file\n");
    }

    /* Initialize mutex */
    mutex_init(&orchestrator_mutex);

    /* Set up maintenance timer */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 15, 0)
    timer_setup(&maintenance_timer, maintenance_timer_callback, 0);
#else
    setup_timer(&maintenance_timer, maintenance_timer_callback, 0);
#endif
    mod_timer(&maintenance_timer, jiffies + HZ * 60 * 10);  /* 10 minutes */

    /* Check for API keys */
    if (!openai_api_key || strlen(openai_api_key) < 20) {
        pr_warn("orchestrator_init: No valid OpenAI API key provided\n");
    }
    
    if (!anthropic_api_key || strlen(anthropic_api_key) < 20) {
        pr_warn("orchestrator_init: No valid Anthropic API key provided\n");
    }
    
    if (!google_gemini_api_key || strlen(google_gemini_api_key) < 10) {
        pr_warn("orchestrator_init: No valid Google Gemini API key provided\n");
    }
    
    if (!openai_api_key && !anthropic_api_key && !google_gemini_api_key) {
        pr_err("orchestrator_init: No API keys provided, module may not function correctly\n");
    }

    pr_info("LLM Orchestrator: Module loaded successfully with major number %d\n", major_number);
    return 0;
}

static void __exit orchestrator_exit(void)
{
    /* Delete timer */
    del_timer_sync(&maintenance_timer);

    /* Remove sysfs attributes */
    device_remove_file(orchestrator_device, &dev_attr_scheduler_algorithm);
    device_remove_file(orchestrator_device, &dev_attr_provider_metrics);
    device_remove_file(orchestrator_device, &dev_attr_reset_metrics);
    device_remove_file(orchestrator_device, &dev_attr_scheduler_weights);
    device_remove_file(orchestrator_device, &dev_attr_auto_adjust);
    device_remove_file(orchestrator_device, &dev_attr_fifo_status);
    device_remove_file(orchestrator_device, &dev_attr_context_status);

    /* Clean up context */
    context_cleanup_all();
    
    /* Clean up FIFO queue */
    fifo_cleanup(&global_scheduler.fifo);
    
    /* Clean up device */
    device_destroy(orchestrator_class, MKDEV(major_number, 0));
    class_destroy(orchestrator_class);
    cdev_del(&orchestrator_cdev);
    unregister_chrdev_region(MKDEV(major_number, 0), 1);
    
    /* Free any allocated keys */
    if (openai_api_key) {
        memzero_explicit(openai_api_key, strlen(openai_api_key));
    }
    
    if (anthropic_api_key) {
        memzero_explicit(anthropic_api_key, strlen(anthropic_api_key));
    }
    
    if (google_gemini_api_key) {
        memzero_explicit(google_gemini_api_key, strlen(google_gemini_api_key));
    }
    
    pr_info("LLM Orchestrator: Module unloaded\n");
}

module_init(orchestrator_init);
module_exit(orchestrator_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LLM Orchestrator");
MODULE_DESCRIPTION("Enhanced LLM Orchestrator with Context Management and Advanced Scheduling");
MODULE_VERSION(DRIVER_VERSION);
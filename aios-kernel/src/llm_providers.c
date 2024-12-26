//
// Created by sina-mazaheri on 12/17/24.
//
//
// Created by sina-mazaheri on 12/17/24.
//
#include "../include/llm_providers.h"
#include <linux/module.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/net.h>
#include <linux/socket.h>
#include <linux/tcp.h>
#include <linux/in.h>
#include <linux/mutex.h>
#include <linux/inet.h>
#include <net/sock.h>
#include <linux/dns_resolver.h>
#include <linux/tls.h>
#include <linux/tcp.h>
#include <linux/time.h>

#define LLM_CONNECT_TIMEOUT_MS 5000
#define LLM_MAX_RETRIES 3
#define LLM_RETRY_DELAY_MS 1000

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Sina Mazaheri");
MODULE_DESCRIPTION("LLM Provider Kernel Module");
MODULE_VERSION("1.0");

static char *api_key = "default-key";
module_param(api_key, charp, 0660);
MODULE_PARM_DESC(api_key, "API key for LLM provider");

static char *provider = "openai";
module_param(provider, charp, 0660);
MODULE_PARM_DESC(provider, "LLM provider (openai, anthropic, etc.)");

static char *model = "gpt-4";
module_param(model, charp, 0660);
MODULE_PARM_DESC(model, "Model name for the LLM provider");

static struct llm_config *current_config;
static struct socket *api_socket;
static DEFINE_MUTEX(llm_mutex);


/* Forward declarations for OpenAI functions */
static int openai_init(struct llm_config *config);
static void openai_cleanup(void);
static int openai_send(const char *prompt, size_t length);
static int openai_receive(struct llm_message *msg);

/* Forward declarations for Anthropic functions */
static int anthropic_init(struct llm_config *config);
static void anthropic_cleanup(void);
static int anthropic_send(const char *prompt, size_t length);
static int anthropic_receive(struct llm_message *msg);

/* Helper functions */
static char *extract_content(const char *response);
static int extract_status_code(const char *response);

static int resolve_hostname(const char *hostname, __be32 *ip_addr)
{
    struct sockaddr_in *sin;
    struct dns_lookup *lookup;
    const char *end;
    int ret;

    /* Start DNS lookup */
    lookup = dns_resolver_start(hostname, strlen(hostname), NULL, 0, NULL);
    if (IS_ERR(lookup))
        return PTR_ERR(lookup);

    /* Wait for resolution */
    ret = dns_resolver_wait(lookup);
    if (ret < 0)
        goto out;

    /* Get first IP address */
    sin = dns_lookup_result(lookup, &end);
    if (!sin) {
        ret = -ENOENT;
        goto out;
    }

    *ip_addr = sin->sin_addr.s_addr;
    ret = 0;

    out:
    dns_resolver_put(lookup);
    return ret;
}

static int setup_tls(struct socket *sock)
{
    struct tls_context *ctx;
    int ret;

    /* Create TLS context */
    ctx = tls_context_new(TLS_VERSION_1_3, TLS_ROLE_CLIENT);
    if (IS_ERR(ctx))
        return PTR_ERR(ctx);

    /* Set TLS options */
    ret = kernel_setsockopt(sock, SOL_TCP, TCP_ULP, "tls", sizeof("tls"));
    if (ret < 0)
        goto out_free_ctx;

    /* Set TLS context */
    ret = kernel_setsockopt(sock, SOL_TLS, TLS_TX, ctx, sizeof(*ctx));
    if (ret < 0)
        goto out_free_ctx;

    return 0;

    out_free_ctx:
    tls_context_free(ctx);
    return ret;
}

static struct llm_provider_ops providers[LLM_MAX_PROVIDERS] = {
        [LLM_OPENAI] = {
                .init = openai_init,
                .cleanup = openai_cleanup,
                .send = openai_send,
                .receive = openai_receive,
        },
        [LLM_ANTHROPIC] = {
                .init = anthropic_init,
                .cleanup = anthropic_cleanup,
                .send = anthropic_send,
                .receive = anthropic_receive,
        },
        /* Add other providers similarly */
};

/* In llm_module_init, use the module parameters to set the config fields */
static int __init llm_module_init(void)
{
    struct llm_config config = {
            .max_tokens = 1000,
            .temperature_X100 = 70,
            .use_ssl = true,
            .timeout_ms = 30000
    };

    /* Copy the module parameters to the config */
    strncpy(config.api_key, api_key, MAX_API_KEY_LENGTH - 1);
    config.api_key[MAX_API_KEY_LENGTH - 1] = '\0';

    /* Map the provider string to enum. For example: */
    if (strcmp(provider, "openai") == 0)
        config.provider = LLM_OPENAI;
    else if (strcmp(provider, "anthropic") == 0)
        config.provider = LLM_ANTHROPIC;
    else if (strcmp(provider, "mistral") == 0)
        config.provider = LLM_MISTRAL;
    else if (strcmp(provider, "huggingface") == 0)
        config.provider = LLM_HUGGINGFACE;
    else if (strcmp(provider, "gemini") == 0)
        config.provider = LLM_GEMINI;
    else
        config.provider = LLM_OPENAI; /* default fallback */

    strncpy(config.model, model, MAX_MODEL_NAME - 1);
    config.model[MAX_MODEL_NAME - 1] = '\0';

    return llm_init(&config);
}

static void __exit llm_module_exit(void)
{
    llm_cleanup();
}

module_init(llm_module_init);
module_exit(llm_module_exit);

/* OpenAI Implementation */

static int openai_init(struct llm_config *config)
{
    struct sockaddr_in server = {0};
    __be32 ip_addr;
    struct timeval timeout;
    int ret, retries = 0;
    bool connected = false;

    /* Resolve OpenAI API hostname */
    ret = resolve_hostname("api.openai.com", &ip_addr);
    if (ret < 0) {
        pr_err("LLM: Failed to resolve api.openai.com: %d\n", ret);
        return ret;
    }

    /* Create socket */
    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM, IPPROTO_TCP, &api_socket);
    if (ret < 0) {
        pr_err("LLM: Failed to create socket: %d\n", ret);
        return ret;
    }

    /* Set socket timeout */
    timeout.tv_sec = LLM_CONNECT_TIMEOUT_MS / 1000;
    timeout.tv_usec = (LLM_CONNECT_TIMEOUT_MS % 1000) * 1000;
    ret = kernel_setsockopt(api_socket, SOL_SOCKET, SO_RCVTIMEO,
                            (char *)&timeout, sizeof(timeout));
    if (ret < 0)
        pr_warn("LLM: Failed to set receive timeout\n");

    /* Set up server address */
    server.sin_family = AF_INET;
    server.sin_port = htons(443);
    server.sin_addr.s_addr = ip_addr;

    /* Connection retry loop */
    while (retries < LLM_MAX_RETRIES && !connected) {
        ret = kernel_connect(api_socket, (struct sockaddr *)&server,
                             sizeof(server), O_NONBLOCK);
        if (ret == 0) {
            connected = true;
            break;
        }

        if (ret != -EINPROGRESS) {
            retries++;
            if (retries < LLM_MAX_RETRIES) {
                pr_warn("LLM: Connection failed, retry %d/%d\n",
                        retries, LLM_MAX_RETRIES);
                msleep(LLM_RETRY_DELAY_MS);
                continue;
            }
            goto err_connect;
        }

        /* Wait for connection completion */
        ret = wait_for_connection(api_socket, LLM_CONNECT_TIMEOUT_MS);
        if (ret == 0) {
            connected = true;
            break;
        }

        retries++;
        if (retries < LLM_MAX_RETRIES) {
            pr_warn("LLM: Connection timeout, retry %d/%d\n",
                    retries, LLM_MAX_RETRIES);
            msleep(LLM_RETRY_DELAY_MS);
        }
    }

    if (!connected) {
        pr_err("LLM: Failed to connect after %d retries\n", LLM_MAX_RETRIES);
        goto err_connect;
    }

    /* Setup TLS */
    ret = setup_tls(api_socket);
    if (ret < 0) {
        pr_err("LLM: Failed to setup TLS: %d\n", ret);
        goto err_tls;
    }

    pr_info("LLM: Successfully connected to OpenAI (TLS enabled)\n");
    return 0;

    err_tls:
    kernel_sock_shutdown(api_socket, SHUT_RDWR);
    err_connect:
    sock_release(api_socket);
    api_socket = NULL;
    return ret;
}

/* Helper function for nonblocking connect */
static int wait_for_connection(struct socket *sock, int timeout_ms)
{
    struct timeval tv;
    fd_set write_fds;
    int ret;

    FD_ZERO(&write_fds);
    FD_SET(sock->file->f_inode->i_rdev, &write_fds);

    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    ret = sock_select(sock->file->f_inode->i_rdev + 1, NULL, &write_fds,
                      NULL, &tv);
    if (ret < 0)
        return ret;
    if (ret == 0)
        return -ETIMEDOUT;
    return 0;
}

static void openai_cleanup(void)
{
    if (api_socket) {
        kernel_sock_shutdown(api_socket, SHUT_RDWR);
        sock_release(api_socket);
        api_socket = NULL;
    }
}

static int openai_send(const char *prompt, size_t length)
{
    struct kvec iov;
    struct msghdr msg = { .msg_flags = MSG_DONTWAIT };
    char *request;
    int ret;
    size_t body_len;
    size_t model_len = strnlen(current_config->model, MAX_MODEL_NAME);

    /*
     * Calculate the length of the JSON body:
     * JSON body format: {"model":"%s","messages":[{"role":"user","content":"%s"}]}
     *
     * Counting chars:
     * "{\"model\":\"" -> 10 chars
     * model -> model_len chars
     * "\",\"messages\":[{\"role\":\"user\",\"content\":\"" -> 40 chars
     * prompt -> length chars
     * "\"}]}" -> 4 chars
     * Total = 10 + model_len + 40 + length + 4 = length + model_len + 54
     */
    body_len = length + model_len + 54;

    /* Allocate request buffer */
    request = kmalloc(body_len + 1024, GFP_KERNEL); // 1024 for headers & safety
    if (!request)
        return -ENOMEM;

    /*
     * Construct the HTTP request:
     * Remember:
     * - Headers end with "\r\n\r\n"
     * - Then the JSON body follows
     */
    snprintf(request, body_len + 1024,
             "POST /v1/chat/completions HTTP/1.1\r\n"
             "Host: api.openai.com\r\n"
             "Authorization: Bearer %s\r\n"
             "Content-Type: application/json\r\n"
             "Content-Length: %zu\r\n"
             "\r\n"
             "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}",
             current_config->api_key,
             body_len,
             current_config->model,
             prompt);

    iov.iov_base = request;
    iov.iov_len = strlen(request);
    ret = kernel_sendmsg(api_socket, &msg, &iov, 1, iov.iov_len);

    kfree(request);
    return ret;
}

static int openai_receive(struct llm_message *msg)
{
    struct kvec iov;
    struct msghdr recv_msg = { .msg_flags = MSG_DONTWAIT };
    char *response;
    int ret;

    response = kmalloc(MAX_RESPONSE_LENGTH, GFP_KERNEL);
    if (!response)
        return -ENOMEM;

    iov.iov_base = response;
    iov.iov_len = MAX_RESPONSE_LENGTH - 1;
    ret = kernel_recvmsg(api_socket, &recv_msg, &iov, 1, iov.iov_len, recv_msg.msg_flags);

    if (ret > 0) {
        response[ret] = '\0';
        msg->status_code = extract_status_code(response);
        msg->content = extract_content(response);
        if (msg->content)
            msg->length = strlen(msg->content);
        else
            msg->length = 0;
    } else {
        msg->status_code = ret < 0 ? ret : 0;
        msg->content = NULL;
        msg->length = 0;
    }

    /* The msg->content memory is separately allocated by extract_content(), if any */
    kfree(response);
    return ret;
}

static char *extract_content(const char *response)
{
    char *content;
    const char *start, *end;
    size_t length;

    /* Find content in JSON response */
    start = strstr(response, "\"content\":\"");
    if (!start)
        return NULL;

    start += strlen("\"content\":\"");  // move past the content start
    end = strchr(start, '\"');
    if (!end)
        return NULL;

    length = end - start;
    content = kmalloc(length + 1, GFP_KERNEL);
    if (!content)
        return NULL;

    memcpy(content, start, length);
    content[length] = '\0';
    return content;
}

static int extract_status_code(const char *response)
{
    const char *p;
    char status_str[4] = {0}; // HTTP status code is 3 digits typically
    int status;

    p = strstr(response, "HTTP/1.1");
    if (!p)
        return -1;

    p = strchr(p, ' ');
    if (!p)
        return -1;
    p++;

    /*
     * Copy up to three digits of the status code.
     * Example: "200 OK" -> status_str = "200"
     */
    strncpy(status_str, p, 3);
    status_str[3] = '\0';

    if (kstrtoint(status_str, 10, &status) < 0)
        return -1;

    return status;
}

/* Anthropic Provider Stub */
static int anthropic_init(struct llm_config *config) {
    struct sockaddr_in server = {0};
    int ret;

    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM, IPPROTO_TCP, &api_socket);
    if (ret < 0)
        return ret;

    server.sin_family = AF_INET;
    server.sin_port = htons(443);

    /* Replace with a known IP or implement DNS resolution.
       For demonstration, we use example.com IP again (93.184.216.34).
       In real usage, you'd need the actual IP of api.anthropic.com or DNS resolution. */
    server.sin_addr.s_addr = in_aton("93.184.216.34");

    ret = kernel_connect(api_socket, (struct sockaddr *)&server, sizeof(server), 0);
    if (ret < 0) {
        sock_release(api_socket);
        api_socket = NULL;
        return ret;
    }

    return 0;
}

static void anthropic_cleanup(void) {
    if (api_socket) {
        kernel_sock_shutdown(api_socket, SHUT_RDWR);
        sock_release(api_socket);
        api_socket = NULL;
    }
}

/* Anthropics send:
 * Use Anthropic-Key header instead of Authorization, and POST /v1/complete.
 * JSON body fields: prompt, model, max_tokens_to_sample, temperature, etc.
 */
static int anthropic_send(const char *prompt, size_t length)
{
    struct kvec iov;
    struct msghdr msg = { .msg_flags = MSG_DONTWAIT };
    char *request;
    char *body;
    int ret;
    size_t prompt_len = length;
    int max_tokens = current_config->max_tokens > 0 ? current_config->max_tokens : 300;

    body = kmalloc(2048, GFP_KERNEL);
    if (!body)
        return -ENOMEM;

    // Since we no longer use floating point, temperature_x100 is already an int.
    int temp_int = current_config->temperature_X100;

    int body_used = snprintf(body, 2048,
                             "{\"prompt\":\"%.*s\",\"model\":\"%s\",\"max_tokens_to_sample\":%d,"
                             "\"temperature\":%d.%02d,\"stop_sequences\":[\"\\n\\nHuman:\"]}",
                             (int)prompt_len, prompt,
                             current_config->model,
                             max_tokens,
                             temp_int / 100, temp_int % 100
    );

    if (body_used < 0 || body_used >= 2048) {
        kfree(body);
        return -EINVAL;
    }

    request = kmalloc(body_used + 1024, GFP_KERNEL);
    if (!request) {
        kfree(body);
        return -ENOMEM;
    }

    int header_len = snprintf(request, body_used + 1024,
                              "POST /v1/complete HTTP/1.1\r\n"
                              "Host: api.anthropic.com\r\n"
                              "Anthropic-Key: %s\r\n"
                              "Content-Type: application/json\r\n"
                              "Content-Length: %d\r\n"
                              "\r\n",
                              current_config->api_key,
                              body_used
    );

    if (header_len < 0 || header_len >= (int)(body_used + 1024)) {
        kfree(body);
        kfree(request);
        return -EINVAL;
    }

    memcpy(request + header_len, body, body_used);
    request[header_len + body_used] = '\0';

    kfree(body);

    iov.iov_base = request;
    iov.iov_len = header_len + body_used;
    ret = kernel_sendmsg(api_socket, &msg, &iov, 1, iov.iov_len);

    kfree(request);
    return ret;
}

/* Anthropics receive:
 * The response should have "completion":"<text>" in JSON.
 * We'll parse similarly to extract_content(), but look for "completion":".
 */

static int anthropic_receive(struct llm_message *msg)
{
    struct kvec iov;
    struct msghdr recv_msg = { .msg_flags = MSG_DONTWAIT };
    char *response;
    int ret;

    response = kmalloc(MAX_RESPONSE_LENGTH, GFP_KERNEL);
    if (!response)
        return -ENOMEM;

    iov.iov_base = response;
    iov.iov_len = MAX_RESPONSE_LENGTH - 1;
    ret = kernel_recvmsg(api_socket, &recv_msg, &iov, 1, iov.iov_len, recv_msg.msg_flags);
    if (ret > 0) {
        response[ret] = '\0';
        msg->status_code = extract_status_code(response);
        msg->content = NULL;
        msg->length = 0;

        /* Extract "completion":"...". Similar logic to extract_content(). */
        {
            const char *start = strstr(response, "\"completion\":\"");
            if (start) {
                start += strlen("\"completion\":\"");
                const char *end = strchr(start, '\"');
                if (end) {
                    size_t length = end - start;
                    char *completion = kmalloc(length + 1, GFP_KERNEL);
                    if (completion) {
                        memcpy(completion, start, length);
                        completion[length] = '\0';
                        msg->content = completion;
                        msg->length = length;
                    }
                }
            }
        }
    } else {
        msg->status_code = ret < 0 ? ret : 0;
        msg->content = NULL;
        msg->length = 0;
    }

    kfree(response);
    return ret;
}

/* Main interface functions */

int llm_init(struct llm_config *config)
{
    int ret;

    if (!config || config->provider >= LLM_MAX_PROVIDERS)
        return -EINVAL;

    mutex_lock(&llm_mutex);

    current_config = kmalloc(sizeof(*current_config), GFP_KERNEL);
    if (!current_config) {
        mutex_unlock(&llm_mutex);
        return -ENOMEM;
    }

    memcpy(current_config, config, sizeof(*current_config));

    if (providers[config->provider].init) {
        ret = providers[config->provider].init(config);
        if (ret < 0) {
            kfree(current_config);
            current_config = NULL;
            mutex_unlock(&llm_mutex);
            return ret;
        }
    }

    mutex_unlock(&llm_mutex);
    return 0;
}

void llm_cleanup(void)
{
    mutex_lock(&llm_mutex);

    if (current_config && providers[current_config->provider].cleanup)
        providers[current_config->provider].cleanup();

    kfree(current_config);
    current_config = NULL;

    mutex_unlock(&llm_mutex);
}

int llm_send(const char *prompt, size_t length)
{
    int ret;

    if (!prompt || !length || length > MAX_PROMPT_LENGTH)
        return -EINVAL;

    mutex_lock(&llm_mutex);

    if (!current_config || !providers[current_config->provider].send) {
        mutex_unlock(&llm_mutex);
        return -EINVAL;
    }

    ret = providers[current_config->provider].send(prompt, length);

    mutex_unlock(&llm_mutex);
    return ret;
}

int llm_receive(struct llm_message *msg)
{
    int ret;

    if (!msg)
        return -EINVAL;

    mutex_lock(&llm_mutex);

    if (!current_config || !providers[current_config->provider].receive) {
        mutex_unlock(&llm_mutex);
        return -EINVAL;
    }

    ret = providers[current_config->provider].receive(msg);

    mutex_unlock(&llm_mutex);
    return ret;
}

/* Helper Functions */



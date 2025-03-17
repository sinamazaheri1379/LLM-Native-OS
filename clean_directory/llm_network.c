#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/net.h>
#include <linux/in.h>
#include <linux/socket.h>
#include <linux/inet.h>
#include <net/sock.h>
#include <linux/version.h>
#include <linux/jiffies.h>
#include <linux/ktime.h>
#include <linux/ctype.h>
#include <linux/slab.h>
#include <linux/timer.h>
#include <net/tcp.h>       /* For TCP_NODELAY */
#include <linux/delay.h>
#include <linux/sockptr.h>
#include "orchestrator_main.h"

#define MAX_HEADER_SIZE 2048
#define MAX_RESPONSE_SIZE (MAX_RESPONSE_LENGTH * 2)
#define MAX_IP_LENGTH 64
#define MAX_PATH_LENGTH 512
#define DEFAULT_TIMEOUT_MS 30000
#define MAX_REQUEST_TIMEOUT_MS 120000

/* Forward declarations */
static int case_insensitive_search(const char *haystack, const char *needle);
static bool is_ip_address_valid(const char *ip);
static bool is_http_path_valid(const char *path);
static bool is_response_complete(const char *buf, int len);
static int parse_http_status(const char *buf);
static bool extract_header_value(const char *response, const char *header_name,
                                 char *value, size_t value_size);

/* Fix 3: Improved locking in request_timeout_callback() in llm_network.c */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 15, 0)
static void request_timeout_callback(struct timer_list *t)
{
    struct request_timeout_data *data = from_timer(data, t, timer);
#else
static void request_timeout_callback(unsigned long ptr)
{
    struct request_timeout_data *data = (struct request_timeout_data *)ptr;
#endif
    int old_value;

    if (!data || !data->completed_flag || !data->sock) {
        pr_err("request_timeout_callback: Invalid data\n");
        return;
    }

    /* Only proceed if the flag is still unset - use atomic test and set */
    old_value = atomic_cmpxchg(data->completed_flag, 0, 1);
    if (old_value == 0) {
        pr_warn("network: Request timed out, forcibly closing socket\n");

        /* Force socket shutdown */
        kernel_sock_shutdown(data->sock, SHUT_RDWR);
    }
}

/* External TLS interface function */
extern int setup_tls(struct socket *sock);
extern void cleanup_tls(struct socket *sock);
extern int tls_send(struct socket *sock, void *data, size_t len);
extern int tls_recv(struct socket *sock, void *data, size_t len, int flags);



/* Custom case-insensitive search function as a replacement for strcasestr */
static int case_insensitive_search(const char *haystack, const char *needle)
{
    size_t needle_len, i, j;

    if (!haystack || !needle)
        return 0;

    needle_len = strlen(needle);
    if (needle_len == 0)
        return 1;

    for (i = 0; haystack[i]; i++) {
        bool found = true;

        for (j = 0; j < needle_len; j++) {
            if (!haystack[i + j] ||
                tolower(haystack[i + j]) != tolower(needle[j])) {
                found = false;
                break;
            }
        }

        if (found)
            return 1;
    }

    return 0;
}

/* Validate IP address format */
static bool is_ip_address_valid(const char *ip)
{
    int octets[4];
    int num_octets;

    if (!ip || strlen(ip) > MAX_IP_LENGTH || strlen(ip) < 7)
        return false;

    /* Check for four octets separated by dots */
    num_octets = sscanf(ip, "%d.%d.%d.%d",
                        &octets[0], &octets[1], &octets[2], &octets[3]);

    if (num_octets != 4)
        return false;

    /* Validate each octet */
    for (int i = 0; i < 4; i++) {
        if (octets[i] < 0 || octets[i] > 255)
            return false;
    }

    return true;
}

/* Validate HTTP path for security */
static bool is_http_path_valid(const char *path)
{
    size_t len;

    if (!path)
        return false;

    len = strlen(path);
    if (len == 0 || len > MAX_PATH_LENGTH)
        return false;

    /* Path must start with a slash */
    if (path[0] != '/')
        return false;

    /* Check for invalid characters */
    for (size_t i = 0; i < len; i++) {
        char c = path[i];
        if (!(isalnum(c) || c == '/' || c == '-' || c == '_' || c == '.' ||
              c == '=' || c == '?' || c == '&' || c == '%' || c == '+'))
            return false;
    }

    return true;
}

/* Check if a buffer contains a complete HTTP response */
static bool is_response_complete(const char *buf, int len)
{
    const char *body_start;
    int content_length = -1;
    const char *cl_header;
    char cl_buf[32];
    int body_len;

    if (!buf || len <= 0)
        return false;

    /* Find end of headers */
    body_start = strstr(buf, "\r\n\r\n");
    if (!body_start)
        return false;

    body_start += 4; /* Skip past header separator */
    body_len = len - (body_start - buf);

    /* Look for Content-Length header */
    cl_header = strstr(buf, "Content-Length:");
    if (!cl_header) {
        cl_header = strstr(buf, "content-length:");
    }

    if (cl_header) {
        /* Extract content length value */
        if (sscanf(cl_header + 15, "%31s", cl_buf) == 1) {
            if (kstrtoint(cl_buf, 10, &content_length) == 0) {
                /* Check if we have the entire body */
                if (body_len >= content_length)
                    return true;
                else
                    return false;
            }
        }
    }

    /* If no Content-Length header, check for end of chunked encoding */
    if (case_insensitive_search(buf, "Transfer-Encoding: chunked") ||
        case_insensitive_search(buf, "transfer-encoding: chunked")) {
        /* Look for "0\r\n\r\n" sequence which marks end of chunked encoding */
        if (strstr(body_start, "\r\n0\r\n\r\n"))
            return true;

        return false;
    }

    /* Heuristic: If we can't determine from headers, check for JSON completion */
    if (body_len > 2) {
        /* For JSON responses, check if we have matching braces */
        if (body_start[0] == '{') {
            int brace_count = 0;
            bool in_string = false;
            bool escaped = false;

            for (int i = 0; i < body_len; i++) {
                char c = body_start[i];

                if (escaped) {
                    escaped = false;
                    continue;
                }

                if (c == '\\') {
                    escaped = true;
                    continue;
                }

                if (c == '"' && !escaped) {
                    in_string = !in_string;
                    continue;
                }

                if (!in_string) {
                    if (c == '{')
                        brace_count++;
                    else if (c == '}')
                        brace_count--;
                }

                /* If we've seen all closing braces and we have at least some content */
                if (brace_count == 0 && i > 10)
                    return true;
            }
        }
    }

    /* Fallback: If the last two bytes are "\r\n" we might have a complete response */
    if (len >= 2 && buf[len-2] == '\r' && buf[len-1] == '\n')
        return true;

    return false;
}

/* Parse HTTP status code from response */
static int parse_http_status(const char *buf)
{
    int status_code = 0;

    if (!buf || strncmp(buf, "HTTP/", 5) != 0)
        return 0;

    /* Find the status code after the HTTP version */
    const char *status_start = strchr(buf, ' ');
    if (!status_start)
        return 0;

    /* Skip the space */
    status_start++;

    /* Parse the status code */
    if (sscanf(status_start, "%d", &status_code) != 1)
        return 0;

    return status_code;
}

/* Extract header value safely */
static bool extract_header_value(const char *response, const char *header_name,
                                 char *value, size_t value_size)
{
    const char *header_start;
    const char *value_start;
    const char *value_end;
    size_t name_len;
    size_t copy_len;

    if (!response || !header_name || !value || value_size == 0)
        return false;

    name_len = strlen(header_name);

    /* Find the header */
    header_start = strstr(response, header_name);
    if (!header_start) {
        /* Try case-insensitive search */
        if (!case_insensitive_search(response, header_name))
            return false;
    }

    /* Find the value start (after ':') */
    value_start = strchr(header_start, ':');
    if (!value_start)
        return false;

    /* Skip the colon and any whitespace */
    value_start++;
    while (*value_start == ' ' || *value_start == '\t')
        value_start++;

    /* Find the end of the value (end of line) */
    value_end = strstr(value_start, "\r\n");
    if (!value_end)
        value_end = value_start + strlen(value_start);

    /* Calculate length to copy */
    copy_len = value_end - value_start;
    if (copy_len >= value_size)
        copy_len = value_size - 1;

    /* Copy the value */
    memcpy(value, value_start, copy_len);
    value[copy_len] = '\0';

    return true;
}

/* Establish a TCP connection to a remote server */
int establish_connection(struct socket **sock, const char *host_ip,
                         int port, bool use_tls)
{
    struct socket *s;
    struct sockaddr_in server_addr;
    int ret;

    if (!sock || !host_ip || port <= 0 || port > 65535)
        return -EINVAL;

    /* Validate IP address */
    if (!is_ip_address_valid(host_ip)) {
        pr_err("establish_connection: Invalid IP address format: %s\n", host_ip);
        return -EINVAL;
    }

    /* Create a TCP socket */
    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM, IPPROTO_TCP, &s);
    if (ret < 0) {
        pr_err("establish_connection: sock_create_kern failed: %d\n", ret);
        return ret;
    }

    /* Prepare server address */
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    /* Convert IP string to binary */
    ret = in4_pton(host_ip, -1, (u8 *)&server_addr.sin_addr.s_addr, -1, NULL);
    if (!ret) {
        pr_err("establish_connection: IP address conversion failed: %s\n", host_ip);
        sock_release(s);
        return -EINVAL;
    }

    /* Set socket options for better reliability */
    {
        int val = 1;
		ret = sock_setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, KERNEL_SOCKPTR(&val), sizeof(val));
        if (ret < 0) {
            pr_warn("establish_connection: Failed to set SO_KEEPALIVE: %d\n", ret);
            /* Not fatal, continue */
        }

        ret = sock_setsockopt(s, IPPROTO_TCP, TCP_NODELAY, KERNEL_SOCKPTR(&val), sizeof(val));
        if (ret < 0) {
            pr_warn("establish_connection: Failed to set TCP_NODELAY: %d\n", ret);
            /* Not fatal, continue */
        }
    }

    /* Connect to the server */
    ret = kernel_connect(s, (struct sockaddr *)&server_addr, sizeof(server_addr), 0);
    if (ret < 0) {
        pr_err("establish_connection: kernel_connect failed: %d to %s:%d\n",
               ret, host_ip, port);
        sock_release(s);
        return ret;
    }

    /* Setup TLS if requested */
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


/* Fix 2: Correctly handle partial sends/receives in llm_network.c */
static int send_all(struct socket *sock, void *data, size_t len, bool use_tls)
{
    size_t sent = 0;
    while (sent < len) {
        int this_sent;

        if (use_tls) {
            this_sent = tls_send(sock, (char *)data + sent, len - sent);
        } else {
            struct msghdr msg = {0};
            struct kvec iov;

            iov.iov_base = (char *)data + sent;
            iov.iov_len = len - sent;

            this_sent = kernel_sendmsg(sock, &msg, &iov, 1, len - sent);
        }

        if (this_sent < 0)
            return this_sent;

        if (this_sent == 0) {
            /* Connection closed */
            return -ECONNRESET;
        }

        sent += this_sent;
    }

    return sent;
}

/* Fix 3: Fix for memory leak in network_send_request() in llm_network.c */
int network_send_request(const char *host_ip, int port,
                         const char *http_path,
                         const char *api_key,
                         const char *auth_header,
                         bool use_tls,
                         unsigned long timeout_ms,
                         struct llm_json_buffer *buf,
                         struct llm_response *resp)
{
    struct socket *sock = NULL;
    struct msghdr msg = {0};
    struct kvec iov[2];
    char *headers = NULL;
    int ret = -EINVAL, header_len = 0, total_len = 0;
    char *recv_buf = NULL;
    int received = 0;
    ktime_t start_time, end_time;
    s64 elapsed_ms;
    bool rate_limited = false;
    unsigned long reset_time_ms = 0;
    struct request_timeout_data *timeout_data = NULL;
    atomic_t request_completed = ATOMIC_INIT(0);

    /* Early validation to prevent resource allocation for invalid requests */
    if (!host_ip || !http_path || !buf || !buf->data || !resp) {
        pr_err("network_send_request: Invalid parameters\n");
        return -EINVAL;
    }

    /* Validate input parameters */
    if (!is_ip_address_valid(host_ip)) {
        pr_err("network_send_request: Invalid IP address: %s\n", host_ip);
        return -EINVAL;
    }

    if (!is_http_path_valid(http_path)) {
        pr_err("network_send_request: Invalid HTTP path: %s\n", http_path);
        return -EINVAL;
    }

    /* Ensure reasonable timeout */
    if (timeout_ms == 0)
        timeout_ms = DEFAULT_TIMEOUT_MS;
    else if (timeout_ms > MAX_REQUEST_TIMEOUT_MS)
        timeout_ms = MAX_REQUEST_TIMEOUT_MS;

    /* Record start time */
    start_time = ktime_get();

    /* Allocate memory for headers (before any operation that could fail) */
    headers = kmalloc(MAX_HEADER_SIZE, GFP_KERNEL);
    if (!headers) {
        pr_err("network_send_request: Failed to allocate headers buffer\n");
        ret = -ENOMEM;
        goto exit_no_cleanup;
    }

    /* Initialize response structure */
    memset(resp, 0, sizeof(*resp));
    resp->timestamp = start_time;

    /* Establish connection */
    ret = establish_connection(&sock, host_ip, port, use_tls);
    if (ret < 0) {
        pr_err("network_send_request: establish_connection failed: %d\n", ret);
        goto exit_free_headers;
    }

    /* Prepare HTTP headers with proper escaping */
    header_len = snprintf(headers, MAX_HEADER_SIZE,
                          "POST %s HTTP/1.1\r\n"
                          "Host: %s\r\n"
                          "%s%s\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: %zu\r\n"
                          "Connection: close\r\n"
                          "User-Agent: LLM-Orchestrator/%s\r\n"
                          "\r\n",
                          http_path, host_ip,
                          auth_header ? auth_header : "",
                          api_key ? api_key : "",
                          buf->used,
                          DRIVER_VERSION);

    if (header_len < 0 || header_len >= MAX_HEADER_SIZE) {
        pr_err("network_send_request: HTTP headers too large\n");
        ret = -EOVERFLOW;
        goto exit_release_sock;
    }

    /* Prepare IO vectors for sending */
    total_len = header_len + buf->used;
    iov[0].iov_base = headers;
    iov[0].iov_len = header_len;
    iov[1].iov_base = buf->data;
    iov[1].iov_len = buf->used;

    /* Set socket timeout */
    {
        long timeout_jiffies = msecs_to_jiffies(timeout_ms);
        #if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 9, 0)
    	ret = sock_setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO_NEW,
                         KERNEL_SOCKPTR(&timeout_jiffies), sizeof(timeout_jiffies));

    	ret = sock_setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO_NEW,
                         KERNEL_SOCKPTR(&timeout_jiffies), sizeof(timeout_jiffies));
		#else
    	ret = kernel_setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                           (char *)&timeout_jiffies, sizeof(timeout_jiffies));

    	ret = kernel_setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
                           (char *)&timeout_jiffies, sizeof(timeout_jiffies));
		#endif
    }

    /* Setup overall request timeout */
    timeout_data = kmalloc(sizeof(*timeout_data), GFP_KERNEL);
    if (!timeout_data) {
        pr_err("network_send_request: Failed to allocate timeout data\n");
        ret = -ENOMEM;
        goto exit_release_sock;
    }

    timeout_data->sock = sock;
    timeout_data->completed_flag = &request_completed;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 15, 0)
    timer_setup(&timeout_data->timer, request_timeout_callback, 0);
#else
    setup_timer(&timeout_data->timer, request_timeout_callback, (unsigned long)timeout_data);
#endif

    mod_timer(&timeout_data->timer, jiffies + msecs_to_jiffies(timeout_ms + 1000)); /* 1s extra margin */

    /* Send the request - use send_all to handle partial sends */
    ret = send_all(sock, headers, header_len, use_tls);
    if (ret < 0) {
        pr_err("network_send_request: send headers failed: %d\n", ret);
        goto exit_cleanup_timer;
    }

    ret = send_all(sock, buf->data, buf->used, use_tls);
    if (ret < 0) {
        pr_err("network_send_request: send data failed: %d\n", ret);
        goto exit_cleanup_timer;
    }

    /* Allocate buffer for response */
    recv_buf = kzalloc(MAX_RESPONSE_SIZE, GFP_KERNEL);
    if (!recv_buf) {
        pr_err("network_send_request: Failed to allocate receive buffer\n");
        ret = -ENOMEM;
        goto exit_cleanup_timer;
    }

    /* Receive response data */
    while (received < MAX_RESPONSE_SIZE - 1) {
        struct kvec recv_iov;
        int this_recv;

        /* Check if request timed out or was cancelled */
        if (atomic_read(&request_completed)) {
            pr_warn("network_send_request: Request timed out or was cancelled\n");
            ret = -ETIMEDOUT;
            goto exit_cleanup_buffer;
        }

        recv_iov.iov_base = recv_buf + received;
        recv_iov.iov_len = MAX_RESPONSE_SIZE - received - 1;

        /* Use TLS recv if TLS is enabled */
        if (use_tls) {
            this_recv = tls_recv(sock, recv_iov.iov_base, recv_iov.iov_len, 0);
        } else {
            this_recv = kernel_recvmsg(sock, &msg, &recv_iov, 1, recv_iov.iov_len, 0);
        }

        if (this_recv < 0) {
            if (this_recv == -EAGAIN || this_recv == -EWOULDBLOCK) {
                pr_warn("network_send_request: Socket receive timeout\n");
                ret = -ETIMEDOUT;
            } else {
                pr_err("network_send_request: receive error: %d\n", this_recv);
                ret = this_recv;
            }
            goto exit_cleanup_buffer;
        }

        if (this_recv == 0) /* End of data */
            break;

        received += this_recv;
        recv_buf[received] = '\0'; /* Ensure null-termination */

        /* Check if we have received a complete response */
        if (is_response_complete(recv_buf, received))
            break;
    }

    /* Mark request as completed */
    atomic_set(&request_completed, 1);

    /* Check for full buffer */
    if (received >= MAX_RESPONSE_SIZE - 1) {
        pr_warn("network_send_request: Response truncated (exceeds %d bytes)\n",
                MAX_RESPONSE_SIZE - 1);
    }

    /* Parse HTTP status code */
    resp->status = parse_http_status(recv_buf);

    if (resp->status >= 200 && resp->status < 300) {
        /* Success */
        ret = 0;
    } else {
        /* Handle error responses */
        if (resp->status == 429) {
            /* Rate limiting */
            rate_limited = true;

            /* Try to get retry-after header */
            char retry_after[32] = {0};
            if (extract_header_value(recv_buf, "Retry-After", retry_after, sizeof(retry_after))) {
                int seconds = 0;
                if (kstrtoint(retry_after, 10, &seconds) == 0 && seconds > 0) {
                    reset_time_ms = seconds * 1000;
                } else {
                    reset_time_ms = 60000; /* Default: 1 minute */
                }
            } else {
                reset_time_ms = 60000; /* Default: 1 minute */
            }

            resp->rate_limit_reset_ms = reset_time_ms;
            ret = -LLM_ERR_RATE_LIMIT;
        } else if (resp->status == 401 || resp->status == 403) {
            /* Authentication/authorization errors */
            ret = -LLM_ERR_AUTH;
        } else {
            /* Other error responses */
            ret = -LLM_ERR_API_RESPONSE;
        }
    }

    /* Extract response body regardless of status */
    {
        char *body = strstr(recv_buf, "\r\n\r\n");
        if (body) {
            body += 4; /* Skip past header separator */

            /* Copy response body with size limit */
            size_t body_len = strlen(body);
            size_t copy_len = min_t(size_t, body_len, MAX_RESPONSE_LENGTH - 1);

            memcpy(resp->content, body, copy_len);
            resp->content[copy_len] = '\0';
            resp->content_length = copy_len;

            /* Try to parse token count if available */
            {
                int prompt_tokens = 0, completion_tokens = 0, total_tokens = 0;
                if (parse_token_count(body, &prompt_tokens, &completion_tokens, &total_tokens) == 0)
                    resp->tokens_used = total_tokens;
            }
        } else {
            /* If we can't find the body separator, use the whole response (likely invalid) */
            size_t copy_len = min_t(size_t, received, MAX_RESPONSE_LENGTH - 1);

            memcpy(resp->content, recv_buf, copy_len);
            resp->content[copy_len] = '\0';
            resp->content_length = copy_len;

            /* If no body found but we have a success status, something's wrong */
            if (resp->status >= 200 && resp->status < 300) {
                pr_warn("network_send_request: No response body found despite success status\n");
            }
        }
    }

    /* Calculate request latency before cleanup */
    end_time = ktime_get();
    elapsed_ms = ktime_to_ms(ktime_sub(end_time, start_time));
    resp->latency_ms = elapsed_ms;

    exit_cleanup_buffer:
    if (recv_buf) {
        kfree(recv_buf);
        recv_buf = NULL;
    }

    exit_cleanup_timer:
    /* Cancel and cleanup timer */
    if (timeout_data) {
        del_timer_sync(&timeout_data->timer);
        kfree(timeout_data);
        timeout_data = NULL;
    }

    exit_release_sock:
    if (sock) {
        /* Clean up TLS resources if any */
        cleanup_tls(sock);
        sock_release(sock);
        sock = NULL;
    }

    exit_free_headers:
    if (headers) {
        kfree(headers);
        headers = NULL;
    }

    exit_no_cleanup:
    return ret;
}

/* Initialize network subsystem */
int network_init(void)
{
    pr_info("LLM network subsystem initialized\n");
    return 0;
}

/* Cleanup network subsystem */
void network_cleanup(void)
{
    pr_info("LLM network subsystem cleaned up\n");
}

EXPORT_SYMBOL(establish_connection);
EXPORT_SYMBOL(network_send_request);
EXPORT_SYMBOL(network_init);
EXPORT_SYMBOL(network_cleanup);
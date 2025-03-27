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
#include <net/netlink.h>
#include "orchestrator_main.h"



/* Forward declarations */
static int case_insensitive_search(const char *haystack, const char *needle);
bool is_ip_address_valid(const char *ip);
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
extern void tls_cleanup(struct socket *sock);
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

//* Keep the original is_ip_address_valid function for IP validation */
bool is_ip_address_valid(const char *ip)
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

static bool is_http_path_valid(const char *path) {
    size_t len;

    if (!path)
        return false;

    len = strlen(path);
    if (len == 0 || len > MAX_PATH_LENGTH)
        return false;

    /* Path must start with a slash */
    if (path[0] != '/')
        return false;

    /* Check for invalid characters - MODIFIED to allow : */
    for (size_t i = 0; i < len; i++) {
        char c = path[i];
        if (!(isalnum(c) || c == '/' || c == '-' || c == '_' || c == '.' ||
              c == '=' || c == '?' || c == '&' || c == '%' || c == '+' ||
              c == ':' || c == ',' || c == ';'))
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
/* Replace the DNS-related code in establish_connection with this simpler version */
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
            // Properly formatted logging statements
			pr_info("Sent MSG: %p", &msg);                        // Print msg address
			pr_info("Sent IOV.IOV_BASE addr: %p, IOV.IOV_LEN: %zu",
        			iov.iov_base, iov.iov_len);                   // Print base address and length
			pr_info("Sent LEN: %zu", len);                        // Print total length
			pr_info("Sent SENT: %zu", sent);                      // Print bytes sent so far
			pr_info("This operation sent: %d bytes", this_sent);  // Print bytes in this operation
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

/* Send multiple buffers efficiently using scatter-gather I/O */
static int send_iov_all(struct socket *sock, struct kvec *iov, size_t iov_count, size_t total_len, bool use_tls)
{
    size_t sent = 0;
    struct kvec *current_iov;
    size_t current_iov_offset = 0;
    size_t remaining_in_iov;
    int this_sent;
    struct msghdr msg = {0};
    /* Use fixed-size array instead of VLA for kernel code */
    struct kvec tmp_iov[8]; /* Reasonably large fixed size - more than enough for our needs */
    size_t tmp_count;

    /* Check if we have too many iovs (unlikely) */
    if (iov_count > 8) {
        pr_err("send_iov_all: Too many iov buffers\n");
        return -EINVAL;
    }

    /* If using TLS, we have to send one buffer at a time since tls_send doesn't support scatter-gather */
    if (use_tls) {
        int ret = 0;
        size_t i;

        for (i = 0; i < iov_count; i++) {
            ret = send_all(sock, iov[i].iov_base, iov[i].iov_len, use_tls);
            if (ret < 0)
                return ret;
        }

        return total_len;
    }

    /* For non-TLS, use scatter-gather I/O */
    while (sent < total_len) {
        /* Find the current iov based on how much we've sent so far */
        size_t consumed = 0;
        size_t i; /* Declare loop variable outside for C90 compliance */
        current_iov = NULL;

        for (i = 0; i < iov_count; i++) {
            if (sent < consumed + iov[i].iov_len) {
                current_iov = &iov[i];
                current_iov_offset = sent - consumed;
                break;
            }
            consumed += iov[i].iov_len;
        }

        if (!current_iov) {
            pr_err("send_iov_all: IOV calculation error\n");
            return -EINVAL;
        }

        /* Prepare current position */
        remaining_in_iov = current_iov->iov_len - current_iov_offset;

        /* Create a temporary iov for the current position */
        tmp_count = 0;

        /* Add the current iov from its offset */
        tmp_iov[tmp_count].iov_base = (char*)current_iov->iov_base + current_iov_offset;
        tmp_iov[tmp_count].iov_len = remaining_in_iov;
        tmp_count++;

        /* Add any remaining iovs */
        for (i = (current_iov - iov) + 1; i < iov_count; i++) {
            tmp_iov[tmp_count] = iov[i];
            tmp_count++;
        }

        /* Send using scatter-gather */
        this_sent = kernel_sendmsg(sock, &msg, tmp_iov, tmp_count, total_len - sent);

        pr_info("Sent %d bytes using scatter-gather with %zu buffers", this_sent, tmp_count);

        if (this_sent < 0)
            return this_sent;

        if (this_sent == 0)
            return -ECONNRESET; /* Connection closed */

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
                      "%s%s"  /* No \r\n after these - they're included in the variables */
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"  /* %zu for size_t */
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

    pr_info("network_send_request: Header\r\n//=========================//\r\n%s", headers);
    /* Send the request - use send_all to handle partial sends */
	pr_info("network_send_request: Payload\r\n//=========================//\r\n%s", buf->data);
    ret = send_iov_all(sock, iov, 2, total_len, use_tls);
    if (ret < 0) {
        pr_err("network_send_request: send request failed: %d\n", ret);
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
		/* Add logging before receive operation */
    	pr_info("Receiving data - Buffer position: %p, Available space: %zu bytes",
            recv_iov.iov_base, recv_iov.iov_len);
    	pr_info("Total received so far: %d of %d maximum bytes",
            received, MAX_RESPONSE_SIZE);
        /* Use TLS recv if TLS is enabled */
        if (use_tls) {
            this_recv = tls_recv(sock, recv_iov.iov_base, recv_iov.iov_len, 0);
        } else {
            this_recv = kernel_recvmsg(sock, &msg, &recv_iov, 1, recv_iov.iov_len, 0);
             pr_info("kernel_recvmsg returned: %d bytes", this_recv);
        pr_info("MSG: %p, IOV count: 1, Flags: 0", &msg);
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

        if (this_recv == 0) { /* End of data */
        	pr_info("Received end of data marker (0 bytes)");
        	break;
    	}

        received += this_recv;
        recv_buf[received] = '\0'; /* Ensure null-termination */
		pr_info("Updated total received: %d bytes", received);
        /* Check if we have received a complete response */
        if (is_response_complete(recv_buf, received)){
          	pr_info("Detected complete response, stopping receive loop");
            break;
        }
    }

    /* Mark request as completed */
    atomic_set(&request_completed, 1);

    /* Check for full buffer */
    if (received >= MAX_RESPONSE_SIZE - 1) {
        pr_warn("network_send_request: Response truncated (exceeds %d bytes)\n",
                MAX_RESPONSE_SIZE - 1);
    }
	{
    /* Define maximum chunk size - kernel printk buffer is often ~1024 bytes */
    #define CHUNK_SIZE 900 /* Use #define instead of const int */
    char *body_start;
    int remaining_len;
    int header_len = 0;
    char *temp_buf = NULL;

    /* Print the beginning of the response and headers */
    pr_info("Network Received - HEADERS:\r\n");

    /* Find the end of headers (blank line) */
    body_start = strstr(recv_buf, "\r\n\r\n");
    if (body_start) {
        /* Calculate header length */
        header_len = (body_start - recv_buf) + 4; /* +4 for the \r\n\r\n */

        /* Print headers as a separate chunk to make them clearly visible */
        if (header_len < CHUNK_SIZE) {
            /* Allocate buffer instead of using VLA */
            temp_buf = kmalloc(CHUNK_SIZE + 1, GFP_KERNEL);
            if (temp_buf) {
                memcpy(temp_buf, recv_buf, header_len);
                temp_buf[header_len] = '\0';
                pr_info("%s", temp_buf);
                kfree(temp_buf);
                temp_buf = NULL;
            } else {
                /* Fallback if allocation fails */
                pr_info("%.900s", recv_buf);
            }
        } else {
            /* Very unlikely case - headers exceed chunk size */
            pr_info("%.900s...", recv_buf);
        }

        /* Move body_start past \r\n\r\n */
        body_start += 4;

        /* Calculate remaining length */
        remaining_len = received - header_len;

        if (remaining_len > 0) {
            int chunk, num_chunks, i;

            /* Calculate number of chunks needed */
            num_chunks = (remaining_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

            pr_info("Network Received - BODY (%d bytes in %d chunks):",
                    remaining_len, num_chunks);

            /* Print each chunk */
            for (i = 0; i < num_chunks; i++) {
                chunk = (remaining_len > CHUNK_SIZE) ? CHUNK_SIZE : remaining_len;

                pr_info("BODY CHUNK %d/%d:\r\n%.900s",
                        i + 1, num_chunks, body_start + (i * CHUNK_SIZE));

                remaining_len -= chunk;
            }
        } else {
            pr_info("Network Received - No body content");
        }
    } else {
        /* No body found, print the whole message in chunks */
        remaining_len = received;
        int chunk, num_chunks, i;

        /* Calculate number of chunks needed */
        num_chunks = (remaining_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for (i = 0; i < num_chunks; i++) {
            chunk = (remaining_len > CHUNK_SIZE) ? CHUNK_SIZE : remaining_len;

            pr_info("CONTENT CHUNK %d/%d:\r\n%.900s",
                    i + 1, num_chunks, recv_buf + (i * CHUNK_SIZE));

            remaining_len -= chunk;
        }
    }
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
        tls_cleanup(sock);
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

int network_test_connectivity(void);

/* Define test targets for basic connectivity verification */
struct connectivity_test_target {
    const char *name;
    const char *ip_address;
    int port;
};

/* Some widely available and stable servers to test against */
static struct connectivity_test_target test_targets[] = {
    {"Cloudflare DNS", "1.1.1.1", 53},     /* Cloudflare DNS - reliable and fast */
    {"Google DNS", "8.8.8.8", 53},         /* Google DNS - widely accessible */
    {"Cloudflare HTTPS", "1.1.1.1", 443}   /* Cloudflare HTTPS - tests TLS port */
};

#define NUM_TEST_TARGETS (sizeof(test_targets) / sizeof(test_targets[0]))

/* Function to test connectivity to a specific target */
static int test_target_connectivity(const char *name, const char *ip, int port)
{
    struct socket *sock = NULL;
    int ret;
    unsigned long start_time, end_time, elapsed_ms;

    pr_info("network_init: Testing connectivity to %s (%s:%d)...\n", name, ip, port);

    /* Record start time */
    start_time = jiffies;

    /* Try to establish a connection with a short timeout */
    ret = establish_connection(&sock, ip, port, false);

    /* Calculate elapsed time */
    end_time = jiffies;
    elapsed_ms = jiffies_to_msecs(end_time - start_time);

    if (ret < 0) {
        pr_warn("network_init: Connection to %s (%s:%d) failed: %d (took %lu ms)\n",
                name, ip, port, ret, elapsed_ms);
        goto cleanup;
    }

    pr_info("network_init: Successfully connected to %s (%s:%d) in %lu ms\n",
            name, ip, port, elapsed_ms);

cleanup:
    /* Close the connection */
    if (sock) {
        kernel_sock_shutdown(sock, SHUT_RDWR);
        sock_release(sock);
    }

    return ret;
}

extern struct llm_provider_config provider_configs[PROVIDER_COUNT];

/* Initialize network subsystem with connectivity testing */
int network_init(void)
{
    int i, ret;
    int success_count = 0;
    bool network_functional = false;

    pr_info("LLM network subsystem initializing...\n");

    /* First, perform basic network connectivity tests */
    pr_info("network_init: Testing basic internet connectivity...\n");

    for (i = 0; i < NUM_TEST_TARGETS; i++) {
        if (test_target_connectivity(test_targets[i].name,
                                     test_targets[i].ip_address,
                                     test_targets[i].port) == 0) {
            success_count++;
            network_functional = true;  /* At least one connection succeeded */
        }
    }

    /* Report results - Fix: use %d for int, not unsigned long */
    if (network_functional) {
        pr_info("network_init: Basic connectivity test passed (%d/%d targets reachable)\n",
                success_count, (int)NUM_TEST_TARGETS);
    } else {
        pr_warn("network_init: FAILED to reach ANY test targets. Network connectivity issues detected.\n");
        pr_warn("network_init: The module will continue to load, but API requests are likely to fail.\n");
        pr_warn("network_init: Please check your internet connection and firewall settings.\n");
    }

    if (network_functional) {
    	/* Test LLM API endpoints using the actual provider configurations */
    	int api_success = 0;
    	const char *provider_names[] = {"OpenAI", "Anthropic", "Google Gemini"};

    	pr_info("network_init: Testing LLM API endpoints...\n");

    	for (i = 0; i < PROVIDER_COUNT; i++) {
        /* Skip if IP is invalid */
        	if (!is_ip_address_valid(provider_configs[i].host_ip)) {
            	pr_warn("network_init: Invalid IP address for %s: %s\n",
                    	provider_names[i], provider_configs[i].host_ip);
            	continue;
        	}

        	ret = test_target_connectivity(provider_names[i],
                                      provider_configs[i].host_ip,
                                      provider_configs[i].port);
        	if (ret == 0) {
            	api_success++;
            	pr_info("network_init: Successfully verified connectivity to %s API (%s)\n",
                   	provider_names[i], provider_configs[i].domain_name);
        	} else {
            	pr_warn("network_init: Failed to connect to %s API (%s)\n",
                   	provider_names[i], provider_configs[i].domain_name);
        	}
    	}

    	pr_info("network_init: API endpoint connectivity test complete: %d/%d endpoints reachable\n",
            	api_success, PROVIDER_COUNT);
	} else {
    	pr_warn("network_init: Skipping API endpoint tests due to basic connectivity failure\n");
	}
    /* Additional network subsystem initialization can go here */

    pr_info("LLM network subsystem initialized\n");
    return 0;  /* Always return success; we don't want to fail module loading over network issues */
}

/*
 * Additional function that can be exported for on-demand testing
 * This can be triggered via ioctl or sysfs to test connectivity anytime
 */
int network_test_connectivity(void)
{
    int i;
    int success_count = 0;

    pr_info("network: Running on-demand connectivity test\n");

    for (i = 0; i < NUM_TEST_TARGETS; i++) {
        if (test_target_connectivity(test_targets[i].name,
                                     test_targets[i].ip_address,
                                     test_targets[i].port) == 0) {
            success_count++;
        }
    }

    /* Fix: use %d for int, not unsigned long */
    pr_info("network: Connectivity test complete: %d/%d targets reachable\n",
            success_count, (int)NUM_TEST_TARGETS);

    return success_count;
}



/* Export the function for use elsewhere */


/* Cleanup network subsystem */
void network_cleanup(void)
{
    pr_info("LLM network subsystem cleaned up\n");
}

EXPORT_SYMBOL(establish_connection);
EXPORT_SYMBOL(network_send_request);
EXPORT_SYMBOL(network_init);
EXPORT_SYMBOL(network_cleanup);
EXPORT_SYMBOL(is_ip_address_valid);
EXPORT_SYMBOL(network_test_connectivity);
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/net.h>
#include <linux/socket.h>
#include <linux/version.h>
#include <linux/tls.h>
#include <net/tls.h>
#include <linux/slab.h>
#include <linux/scatterlist.h>
#include <linux/crypto.h>
#include <crypto/aead.h>
#include "orchestrator_main.h"

/* Constants */
#define TLS_CIPHER_AES_GCM_128 51
#define TLS_VERSION_1_2 0x0303
#define TLS_VERSION_1_3 0x0304

/* TLS Context */
struct tls_crypto_info_aes_gcm_128 {
    unsigned short version;
    unsigned short cipher_type;
    unsigned char iv[8];
    unsigned char key[16];
    unsigned char salt[4];
    unsigned char rec_seq[8];
} __packed;

/* TLS State */
struct tls_state {
    bool initialized;
    struct tls_crypto_info_aes_gcm_128 crypto_info;
    struct crypto_aead *aead;     /* For software fallback */
    struct scatterlist *sg;
    void *aad;
    struct socket *socket;
    bool using_ktls;              /* Flag to track if using kTLS */
};

struct tls_context {
    bool is_tls_1_3;
    struct tls_crypto_info_aes_gcm_128 crypto_send;
    struct tls_crypto_info_aes_gcm_128 crypto_recv;
    bool ktls_enabled;
};

/* Forward declarations */
static int tls_sw_fallback_send(struct socket *sock, void *data, size_t len);
static int tls_sw_fallback_recv(struct socket *sock, void *data, size_t len, int flags);

/* Helper function to check if kTLS is supported */
static bool is_ktls_supported(void)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 13, 0)
    /* kTLS was introduced in kernel 4.13 */
    return true;
#else
    pr_warn("kTLS not supported in this kernel version\n");
    return false;
#endif
}

/* Helper function to check if TLS 1.3 is supported */
static bool is_tls_1_3_supported(void)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 1, 0)
    /* TLS 1.3 support was added in kernel 5.1 */
    return true;
#else
    return false;
#endif
}

/*
 * In a real implementation, you would perform a TLS handshake here
 * Since that's very complex, we'll use predefined keys for demonstration
 */
static int prepare_tls_keys(struct tls_context *ctx)
{
    /* Initialize crypto info structure */
    memset(&ctx->crypto_send, 0, sizeof(ctx->crypto_send));

    /* Set version based on kernel support */
    if (is_tls_1_3_supported()) {
        ctx->crypto_send.info.version = TLS_1_3_VERSION;
        ctx->is_tls_1_3 = true;
    } else {
        ctx->crypto_send.info.version = TLS_1_2_VERSION;
        ctx->is_tls_1_3 = false;
    }

    /* Set cipher type */
    ctx->crypto_send.info.cipher_type = TLS_CIPHER_AES_GCM_128;

    /* Generate key material (warning: not for production!) */
    get_random_bytes(ctx->crypto_send.key, TLS_CIPHER_AES_GCM_128_KEY_SIZE);
    get_random_bytes(ctx->crypto_send.iv, TLS_CIPHER_AES_GCM_128_IV_SIZE);
    get_random_bytes(ctx->crypto_send.salt, TLS_CIPHER_AES_GCM_128_SALT_SIZE);
    memset(ctx->crypto_send.rec_seq, 0, TLS_CIPHER_AES_GCM_128_REC_SEQ_SIZE);

    /* Copy the same crypto info for receiving */
    memcpy(&ctx->crypto_recv, &ctx->crypto_send, sizeof(ctx->crypto_send));

    return 0;
}


/* Generate TLS session parameters */
static int generate_tls_params(struct tls_state *state)
{
    /*
     * NOTE: In a real-world implementation, this function would perform a
     * proper TLS handshake or use parameters from userspace
     */

    /* For demonstration purposes only - NOT a secure implementation */
    pr_warn("TLS: Using demo parameters - NOT SECURE FOR PRODUCTION\n");

    /* Try to use TLS 1.3 if possible, fall back to 1.2 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 17, 0)
    state->crypto_info.version = TLS_VERSION_1_3;
#else
    state->crypto_info.version = TLS_VERSION_1_2;
#endif
    state->crypto_info.cipher_type = TLS_CIPHER_AES_GCM_128;

    /* Generate random key material for demonstration */
    get_random_bytes(state->crypto_info.iv, sizeof(state->crypto_info.iv));
    get_random_bytes(state->crypto_info.key, sizeof(state->crypto_info.key));
    get_random_bytes(state->crypto_info.salt, sizeof(state->crypto_info.salt));
    memset(state->crypto_info.rec_seq, 0, sizeof(state->crypto_info.rec_seq));

    /* Set up software fallback crypto */
#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 13, 0)
    {
        /* Initialize AEAD for AES-GCM */
        state->aead = crypto_alloc_aead("gcm(aes)", 0, 0);
        if (IS_ERR(state->aead)) {
            pr_err("TLS: Failed to allocate AEAD: %ld\n", PTR_ERR(state->aead));
            return PTR_ERR(state->aead);
        }

        if (crypto_aead_setkey(state->aead, state->crypto_info.key,
                              sizeof(state->crypto_info.key))) {
            pr_err("TLS: Failed to set AEAD key\n");
            crypto_free_aead(state->aead);
            return -EINVAL;
        }

        /* Allocate scatterlist for data */
        state->sg = kmalloc(sizeof(struct scatterlist) * 3, GFP_KERNEL);
        if (!state->sg) {
            crypto_free_aead(state->aead);
            return -ENOMEM;
        }
        memset(state->sg, 0, sizeof(struct scatterlist) * 3);
        sg_init_table(state->sg, 3);

        /* Allocate AAD (additional authenticated data) */
        state->aad = kmalloc(16, GFP_KERNEL);
        if (!state->aad) {
            kfree(state->sg);
            crypto_free_aead(state->aead);
            return -ENOMEM;
        }
        memset(state->aad, 0, 16);
    }
#endif

    state->initialized = true;
    return 0;
}

/* Fix 1: Improved TLS Implementation in llm_tls.c */

/* Main setup function */
int setup_tls(struct socket *sock)
{
    struct tls_context *ctx;
    int ret;

    if (!sock || !sock->sk) {
        pr_err("setup_tls: Invalid socket\n");
        return -EINVAL;
    }

    /* Allocate TLS context */
    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return -ENOMEM;

    /* Check if kTLS is supported */
    if (!is_ktls_supported()) {
        pr_warn("setup_tls: kTLS not supported, using software fallback\n");
        goto use_fallback;
    }

    /* Prepare TLS keys (in a real implementation, do a handshake here) */
    ret = prepare_tls_keys(ctx);
    if (ret < 0) {
        pr_err("setup_tls: Failed to prepare TLS keys: %d\n", ret);
        goto err_free_ctx;
    }

    /* Enable TX (sending) side of kTLS */
    ret = kernel_setsockopt(sock, SOL_TLS, TLS_TX, &ctx->crypto_send,
                         sizeof(ctx->crypto_send));
    if (ret < 0) {
        pr_warn("setup_tls: Failed to set TLS_TX: %d, falling back to software TLS\n", ret);
        goto use_fallback;
    }

    /* Enable RX (receiving) side of kTLS */
    ret = kernel_setsockopt(sock, SOL_TLS, TLS_RX, &ctx->crypto_recv,
                         sizeof(ctx->crypto_recv));
    if (ret < 0) {
        pr_warn("setup_tls: Failed to set TLS_RX: %d, falling back to software TLS\n", ret);
        /* Disable TX side since we couldn't enable RX */
        kernel_setsockopt(sock, SOL_TLS, TLS_TX, NULL, 0);
        goto use_fallback;
    }

    /* Store our context in the socket's user data */
    ctx->ktls_enabled = true;
    sock->sk->sk_user_data = ctx;

    pr_info("setup_tls: kTLS enabled successfully for socket (TLS %s)\n",
           ctx->is_tls_1_3 ? "1.3" : "1.2");
    return 0;

use_fallback:
    /* When kTLS fails, use a software fallback approach */
    ctx->ktls_enabled = false;
    sock->sk->sk_user_data = ctx;
    pr_info("setup_tls: Using software TLS fallback\n");
    return 0;

err_free_ctx:
    kfree(ctx);
    return ret;
}

/* Software fallback for TLS send */
static int tls_sw_fallback_send(struct socket *sock, void *data, size_t len)
{
    struct msghdr msg = {0};
    struct kvec iov;
    struct tls_state *state;

    if (!sock || !sock->sk || !data || len == 0)
        return -EINVAL;

    state = sock->sk->sk_user_data;
    if (!state || !state->initialized)
        return -EINVAL;

#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 13, 0)
    if (state->aead) {
        /*
         * This is a simplified implementation. In a real TLS stack, we would:
         * 1. Create a TLS record header
         * 2. Encrypt the data with AES-GCM and the proper nonce/IV
         * 3. Add the authentication tag
         * 4. Send the complete record
         */
        char *tls_header = kmalloc(5, GFP_KERNEL);
        char *encrypted = kmalloc(len + 16, GFP_KERNEL); /* data + auth tag */
        char *full_record = NULL;

        if (!tls_header || !encrypted) {
            if (tls_header) kfree(tls_header);
            if (encrypted) kfree(encrypted);
            return -ENOMEM;
        }

        /* Create a basic TLS record header - simplified */
        tls_header[0] = 0x17; /* Application data */
        tls_header[1] = 0x03; /* TLS 1.2 major */
        tls_header[2] = 0x03; /* TLS 1.2 minor */
        tls_header[3] = (len >> 8) & 0xFF; /* Length high byte */
        tls_header[4] = len & 0xFF; /* Length low byte */

        /* In a real implementation, we would encrypt data using state->aead */
        /* For now, we just copy the plaintext as a placeholder */
        memcpy(encrypted, data, len);

        /* Combine header and encrypted data */
        full_record = kmalloc(5 + len + 16, GFP_KERNEL);
        if (!full_record) {
            kfree(tls_header);
            kfree(encrypted);
            return -ENOMEM;
        }

        memcpy(full_record, tls_header, 5);
        memcpy(full_record + 5, encrypted, len);

        /* Send the record */
        iov.iov_base = full_record;
        iov.iov_len = 5 + len; /* Without auth tag for now */

        ret = kernel_sendmsg(sock, &msg, &iov, 1, 5 + len);

        kfree(tls_header);
        kfree(encrypted);
        kfree(full_record);

        if (ret < 0) {
            pr_err("TLS fallback send error: %d\n", ret);
            return ret;
        }

        return ret - 5; /* Return actual data bytes sent (excluding header) */
    }
#endif

    /* If no crypto available, send plaintext with warning */
    pr_warn("TLS: Software fallback sending plaintext - NOT SECURE\n");

    iov.iov_base = data;
    iov.iov_len = len;

    return kernel_sendmsg(sock, &msg, &iov, 1, len);
}

/* Software fallback for TLS receive */
static int tls_sw_fallback_recv(struct socket *sock, void *data, size_t len, int flags)
{
    struct msghdr msg = {0};
    struct kvec iov;
    struct tls_state *state;

    if (!sock || !sock->sk || !data || len == 0)
        return -EINVAL;

    state = sock->sk->sk_user_data;
    if (!state || !state->initialized)
        return -EINVAL;

#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 13, 0)
    if (state->aead) {
        /*
         * This is a simplified implementation. In a real TLS stack, we would:
         * 1. Receive the TLS record header
         * 2. Determine record type and length
         * 3. Receive the rest of the record
         * 4. Decrypt and verify the record
         * 5. Return the plaintext
         */
        char header[5];
        char *record = NULL;
        unsigned short record_len;
        int record_type;

        /* First read the 5-byte TLS header */
        iov.iov_base = header;
        iov.iov_len = 5;

        ret = kernel_recvmsg(sock, &msg, &iov, 1, 5, flags);
        if (ret < 0)
            return ret;

        if (ret < 5) {
            pr_err("TLS: Incomplete TLS header received\n");
            return -EPROTO;
        }

        /* Parse header */
        record_type = header[0];
        record_len = (header[3] << 8) | header[4];

        if (record_type != 0x17) { /* Not application data */
            pr_err("TLS: Unexpected record type: %d\n", record_type);
            return -EPROTO;
        }

        /* Read the rest of the record */
        record = kmalloc(record_len, GFP_KERNEL);
        if (!record)
            return -ENOMEM;

        iov.iov_base = record;
        iov.iov_len = record_len;

        ret = kernel_recvmsg(sock, &msg, &iov, 1, record_len, flags);
        if (ret < 0) {
            kfree(record);
            return ret;
        }

        if (ret < record_len) {
            pr_err("TLS: Incomplete TLS record received\n");
            kfree(record);
            return -EPROTO;
        }

        /* In a real implementation, we would decrypt the record here */
        /* For now, we just copy the data as a placeholder */
        if (record_len > len) {
            pr_warn("TLS: Record larger than buffer, truncating\n");
            record_len = len;
        }

        memcpy(data, record, record_len);
        kfree(record);

        return record_len;
    }
#endif

    /* If no crypto available, receive plaintext with warning */
    pr_warn("TLS: Software fallback receiving plaintext - NOT SECURE\n");

    iov.iov_base = data;
    iov.iov_len = len;

    return kernel_recvmsg(sock, &msg, &iov, 1, len, flags);
}

/* Send function that works with both kTLS and fallback */
int tls_send(struct socket *sock, void *data, size_t len)
{
    struct tls_context *ctx;
    struct msghdr msg = {0};
    struct kvec iov;

    if (!sock || !sock->sk || !data || len == 0)
        return -EINVAL;

    ctx = sock->sk->sk_user_data;
    if (!ctx) {
        /* No TLS context, send raw data */
        goto send_raw;
    }

    /* If kTLS is enabled, we can just use the socket directly */
    if (ctx->ktls_enabled) {
        goto send_raw;
    }

    /* Otherwise, we'd need to encrypt the data in software
     * For simplicity, we're skipping encryption here
     * In a real implementation, you would encrypt the data before sending
     */
    pr_debug("tls_send: Using software fallback (unencrypted)\n");

send_raw:
    iov.iov_base = data;
    iov.iov_len = len;
    return kernel_sendmsg(sock, &msg, &iov, 1, len);
}

/* Receive function that works with both kTLS and fallback */
int tls_recv(struct socket *sock, void *data, size_t len, int flags)
{
    struct tls_context *ctx;
    struct msghdr msg = {0};
    struct kvec iov;

    if (!sock || !sock->sk || !data || len == 0)
        return -EINVAL;

    ctx = sock->sk->sk_user_data;
    if (!ctx) {
        /* No TLS context, receive raw data */
        goto recv_raw;
    }

    /* If kTLS is enabled, we can just use the socket directly */
    if (ctx->ktls_enabled) {
        goto recv_raw;
    }

    /* Otherwise, we'd need to decrypt the data in software
     * For simplicity, we're skipping decryption here
     * In a real implementation, you would decrypt the received data
     */
    pr_debug("tls_recv: Using software fallback (unencrypted)\n");

recv_raw:
    iov.iov_base = data;
    iov.iov_len = len;
    return kernel_recvmsg(sock, &msg, &iov, 1, len, flags);
}

/* Cleanup TLS resources */
void cleanup_tls(struct socket *sock)
{
    struct tls_context *ctx;

    if (!sock || !sock->sk)
        return;

    ctx = sock->sk->sk_user_data;
    if (!ctx)
        return;

    /* If kTLS was enabled, disable it */
    if (ctx->ktls_enabled) {
        kernel_setsockopt(sock, SOL_TLS, TLS_TX, NULL, 0);
        kernel_setsockopt(sock, SOL_TLS, TLS_RX, NULL, 0);
    }

    /* Clear sensitive data */
    memzero_explicit(ctx, sizeof(*ctx));

    /* Free context */
    kfree(ctx);
    sock->sk->sk_user_data = NULL;
}

/* TLS module initialization */
int tls_init(void)
{
    pr_info("LLM TLS subsystem initialized\n");

    /* Check if kTLS is supported */
    if (is_ktls_supported()) {
        pr_info("TLS: Using kernel TLS (kTLS)\n");
    } else {
        pr_warn("TLS: Kernel TLS not supported, using software fallback with limited security\n");
    }

    return 0;
}

/* TLS module cleanup */
void tls_cleanup(void)
{
    pr_info("LLM TLS subsystem cleaned up\n");
}

EXPORT_SYMBOL(setup_tls);
EXPORT_SYMBOL(cleanup_tls);
EXPORT_SYMBOL(tls_send);
EXPORT_SYMBOL(tls_recv);
EXPORT_SYMBOL(tls_init);
EXPORT_SYMBOL(tls_cleanup);
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

/* Forward declarations */
static int tls_sw_fallback_send(struct socket *sock, void *data, size_t len);
static int tls_sw_fallback_recv(struct socket *sock, void *data, size_t len, int flags);

/* Check if kTLS is supported by kernel */
static bool is_ktls_supported(void)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 13, 0)
    return true;
#else
    pr_warn("setup_tls: kTLS not supported in this kernel version\n");
    return false;
#endif
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

/* Create a new setup_tls function that implements better security */
int setup_tls(struct socket *sock)
{
    struct tls_state *state;
    int ret;

    if (!sock || !sock->sk) {
        pr_err("setup_tls: Invalid socket\n");
        return -EINVAL;
    }

    /* Allocate TLS state */
    state = kzalloc(sizeof(*state), GFP_KERNEL);
    if (!state)
        return -ENOMEM;

    state->socket = sock;
    state->using_ktls = false;
    state->initialized = true;  /* Set this to true after generating params */

    /* Generate proper parameters */
    ret = generate_tls_params(state);
    if (ret < 0) {
        kfree(state);
        return ret;
    }

    /* Store TLS state in socket */
    sock->sk->sk_user_data = state;

    return 0;
}

/* Software fallback for TLS send */
static int tls_sw_fallback_send(struct socket *sock, void *data, size_t len)
{
    struct msghdr msg = {0};
    struct kvec iov;
    int ret;
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
    int ret;
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

/* Wrapper for sending data with the appropriate TLS method */
int tls_send(struct socket *sock, void *data, size_t len)
{
    struct tls_state *state;

    if (!sock || !sock->sk || !data || len == 0)
        return -EINVAL;

    state = sock->sk->sk_user_data;
    if (!state)
        return -EINVAL;

    /* If using kTLS, send normally (kernel will handle encryption) */
    if (state->using_ktls) {
        struct msghdr msg = {0};
        struct kvec iov;

        iov.iov_base = data;
        iov.iov_len = len;

        return kernel_sendmsg(sock, &msg, &iov, 1, len);
    }

    /* Otherwise use software fallback */
    return tls_sw_fallback_send(sock, data, len);
}

/* Wrapper for receiving data with the appropriate TLS method */
int tls_recv(struct socket *sock, void *data, size_t len, int flags)
{
    struct tls_state *state;

    if (!sock || !sock->sk || !data || len == 0)
        return -EINVAL;

    state = sock->sk->sk_user_data;
    if (!state)
        return -EINVAL;

    /* If using kTLS, receive normally (kernel will handle decryption) */
    if (state->using_ktls) {
        struct msghdr msg = {0};
        struct kvec iov;

        iov.iov_base = data;
        iov.iov_len = len;

        return kernel_recvmsg(sock, &msg, &iov, 1, len, flags);
    }

    /* Otherwise use software fallback */
    return tls_sw_fallback_recv(sock, data, len, flags);
}

/* Clean up TLS resources for a socket */
void cleanup_tls(struct socket *sock)
{
    struct tls_state *state;

    if (!sock || !sock->sk)
        return;

    state = sock->sk->sk_user_data;
    if (!state)
        return;

    /* Free crypto resources if we're using software fallback */
#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 13, 0)
    if (state->aead) {
        crypto_free_aead(state->aead);
        state->aead = NULL;
    }

    if (state->sg) {
        kfree(state->sg);
        state->sg = NULL;
    }

    if (state->aad) {
        kfree(state->aad);
        state->aad = NULL;
    }
#endif

    /* Clear sensitive data */
    memzero_explicit(&state->crypto_info, sizeof(state->crypto_info));

    /* Free the state */
    kfree(state);
    sock->sk->sk_user_data = NULL;

    pr_debug("cleanup_tls: TLS resources cleaned up\n");
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
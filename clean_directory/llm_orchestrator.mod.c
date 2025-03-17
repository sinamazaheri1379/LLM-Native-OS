#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_MITIGATION_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

KSYMTAB_FUNC(context_add_entry, "", "");
KSYMTAB_FUNC(context_get_conversation, "", "");
KSYMTAB_FUNC(context_clear_conversation, "", "");
KSYMTAB_FUNC(context_cleanup_all, "", "");
KSYMTAB_FUNC(context_get_entry_count, "", "");
KSYMTAB_FUNC(context_prune_old_conversations, "", "");
KSYMTAB_FUNC(context_get_stats, "", "");
KSYMTAB_FUNC(context_management_init, "", "");
KSYMTAB_FUNC(context_management_cleanup, "", "");
KSYMTAB_FUNC(context_management_initialized, "", "");
KSYMTAB_FUNC(find_conversation_internal, "", "");
KSYMTAB_FUNC(find_conversation, "", "");
KSYMTAB_FUNC(scheduler_init, "", "");
KSYMTAB_FUNC(select_provider, "", "");
KSYMTAB_FUNC(update_provider_metrics, "", "");
KSYMTAB_FUNC(handle_rate_limit, "", "");
KSYMTAB_FUNC(adjust_scheduler_weights, "", "");
KSYMTAB_FUNC(scheduler_reset_metrics, "", "");
KSYMTAB_FUNC(set_scheduler_state, "", "");
KSYMTAB_FUNC(get_scheduler_state, "", "");
KSYMTAB_FUNC(fifo_init, "", "");
KSYMTAB_FUNC(fifo_add_provider, "", "");
KSYMTAB_FUNC(fifo_cleanup, "", "");
KSYMTAB_FUNC(get_default_model, "", "");
KSYMTAB_FUNC(is_model_supported, "", "");
KSYMTAB_FUNC(json_manager_init, "", "");
KSYMTAB_FUNC(json_manager_cleanup, "", "");
KSYMTAB_FUNC(json_manager_initialized_check, "", "");
KSYMTAB_FUNC(json_buffer_init, "", "");
KSYMTAB_FUNC(json_buffer_resize, "", "");
KSYMTAB_FUNC(json_buffer_free, "", "");
KSYMTAB_FUNC(append_json_string, "", "");
KSYMTAB_FUNC(append_json_value, "", "");
KSYMTAB_FUNC(append_json_number, "", "");
KSYMTAB_FUNC(append_json_float, "", "");
KSYMTAB_FUNC(append_json_boolean, "", "");
KSYMTAB_FUNC(extract_response_content, "", "");
KSYMTAB_FUNC(parse_token_count, "", "");
KSYMTAB_FUNC(establish_connection, "", "");
KSYMTAB_FUNC(network_send_request, "", "");
KSYMTAB_FUNC(network_init, "", "");
KSYMTAB_FUNC(network_cleanup, "", "");
KSYMTAB_FUNC(setup_tls, "", "");
KSYMTAB_FUNC(cleanup_tls, "", "");
KSYMTAB_FUNC(tls_send, "", "");
KSYMTAB_FUNC(tls_recv, "", "");
KSYMTAB_FUNC(tls_init, "", "");
KSYMTAB_FUNC(tls_cleanup, "", "");
KSYMTAB_FUNC(context_register_memory, "", "");
KSYMTAB_FUNC(context_unregister_memory, "", "");
KSYMTAB_FUNC(context_get_memory_stats, "", "");
KSYMTAB_FUNC(context_set_memory_limits, "", "");
KSYMTAB_FUNC(context_cleanup_memory_tracking, "", "");
KSYMTAB_FUNC(context_allocate_entry, "", "");
KSYMTAB_FUNC(context_free_entry, "", "");
KSYMTAB_FUNC(memory_management_init, "", "");
KSYMTAB_FUNC(memory_management_cleanup, "", "");
KSYMTAB_FUNC(memory_stats_show, "", "");
KSYMTAB_FUNC(memory_limits_show, "", "");
KSYMTAB_FUNC(memory_limits_store, "", "");

SYMBOL_CRC(context_add_entry, 0x779550f0, "");
SYMBOL_CRC(context_get_conversation, 0x10cd18b0, "");
SYMBOL_CRC(context_clear_conversation, 0x51937cdf, "");
SYMBOL_CRC(context_cleanup_all, 0xbef6564d, "");
SYMBOL_CRC(context_get_entry_count, 0x13ae5618, "");
SYMBOL_CRC(context_prune_old_conversations, 0x46c02e1b, "");
SYMBOL_CRC(context_get_stats, 0x3578b6ea, "");
SYMBOL_CRC(context_management_init, 0xd9fb2f41, "");
SYMBOL_CRC(context_management_cleanup, 0x13a7f10a, "");
SYMBOL_CRC(context_management_initialized, 0x310bc9f2, "");
SYMBOL_CRC(find_conversation_internal, 0x6aab7111, "");
SYMBOL_CRC(find_conversation, 0x429803f5, "");
SYMBOL_CRC(scheduler_init, 0xbc4034a6, "");
SYMBOL_CRC(select_provider, 0xc551bfe1, "");
SYMBOL_CRC(update_provider_metrics, 0x53183523, "");
SYMBOL_CRC(handle_rate_limit, 0x0683f91a, "");
SYMBOL_CRC(adjust_scheduler_weights, 0x8aaada1c, "");
SYMBOL_CRC(scheduler_reset_metrics, 0xc78e3505, "");
SYMBOL_CRC(set_scheduler_state, 0x6733da91, "");
SYMBOL_CRC(get_scheduler_state, 0x084bcf29, "");
SYMBOL_CRC(fifo_init, 0x8b6628a6, "");
SYMBOL_CRC(fifo_add_provider, 0x66f2501d, "");
SYMBOL_CRC(fifo_cleanup, 0x0f8a994d, "");
SYMBOL_CRC(get_default_model, 0x85a03a4b, "");
SYMBOL_CRC(is_model_supported, 0xf2940b40, "");
SYMBOL_CRC(json_manager_init, 0x52a6dcfd, "");
SYMBOL_CRC(json_manager_cleanup, 0x14c955ec, "");
SYMBOL_CRC(json_manager_initialized_check, 0x125c5fb8, "");
SYMBOL_CRC(json_buffer_init, 0xd231fc54, "");
SYMBOL_CRC(json_buffer_resize, 0x734fd043, "");
SYMBOL_CRC(json_buffer_free, 0x74132eb8, "");
SYMBOL_CRC(append_json_string, 0x1697d5e1, "");
SYMBOL_CRC(append_json_value, 0x480d7e8b, "");
SYMBOL_CRC(append_json_number, 0x517f7d05, "");
SYMBOL_CRC(append_json_float, 0x54965401, "");
SYMBOL_CRC(append_json_boolean, 0x8360f463, "");
SYMBOL_CRC(extract_response_content, 0xaea64a4c, "");
SYMBOL_CRC(parse_token_count, 0x170a9610, "");
SYMBOL_CRC(establish_connection, 0xf57116b0, "");
SYMBOL_CRC(network_send_request, 0xf9797a33, "");
SYMBOL_CRC(network_init, 0xa9026e60, "");
SYMBOL_CRC(network_cleanup, 0xd0f0be56, "");
SYMBOL_CRC(setup_tls, 0x5312cb49, "");
SYMBOL_CRC(cleanup_tls, 0x6170944a, "");
SYMBOL_CRC(tls_send, 0xfa765648, "");
SYMBOL_CRC(tls_recv, 0x4a456f2a, "");
SYMBOL_CRC(tls_init, 0xe8d9f8fa, "");
SYMBOL_CRC(tls_cleanup, 0x158412f9, "");
SYMBOL_CRC(context_register_memory, 0x50ad59d4, "");
SYMBOL_CRC(context_unregister_memory, 0xf96ca2bb, "");
SYMBOL_CRC(context_get_memory_stats, 0xc157de1b, "");
SYMBOL_CRC(context_set_memory_limits, 0x3d14e57a, "");
SYMBOL_CRC(context_cleanup_memory_tracking, 0x1681e4e0, "");
SYMBOL_CRC(context_allocate_entry, 0xe6079c01, "");
SYMBOL_CRC(context_free_entry, 0x107c1391, "");
SYMBOL_CRC(memory_management_init, 0xe9be8253, "");
SYMBOL_CRC(memory_management_cleanup, 0x91581a75, "");
SYMBOL_CRC(memory_stats_show, 0x66ab2922, "");
SYMBOL_CRC(memory_limits_show, 0xcbd44ca3, "");
SYMBOL_CRC(memory_limits_store, 0x571ec088, "");

static const char ____versions[]
__used __section("__versions") =
	"\x18\x00\x00\x00\xb2\x3c\xfe\xf3"
	"try_module_get\0\0"
	"\x1c\x00\x00\x00\x2b\x2f\xec\xe3"
	"alloc_chrdev_region\0"
	"\x1c\x00\x00\x00\x48\x9f\xdb\x88"
	"__check_object_size\0"
	"\x18\x00\x00\x00\xed\x25\xcd\x49"
	"alloc_workqueue\0"
	"\x18\x00\x00\x00\x4b\x69\x80\x6f"
	"param_ops_ulong\0"
	"\x18\x00\x00\x00\xc2\x9c\xc4\x13"
	"_copy_from_user\0"
	"\x1c\x00\x00\x00\x8f\x18\x02\x7f"
	"__msecs_to_jiffies\0\0"
	"\x1c\x00\x00\x00\x91\xc9\xc5\x52"
	"__kmalloc_noprof\0\0\0\0"
	"\x14\x00\x00\x00\x6e\x4a\x6e\x65"
	"snprintf\0\0\0\0"
	"\x18\x00\x00\x00\x36\xf2\xb6\xc5"
	"queue_work_on\0\0\0"
	"\x18\x00\x00\x00\x7f\xe7\x31\xb2"
	"class_destroy\0\0\0"
	"\x14\x00\x00\x00\x86\x81\x84\x96"
	"scnprintf\0\0\0"
	"\x10\x00\x00\x00\x38\xdf\xac\x69"
	"memcpy\0\0"
	"\x14\x00\x00\x00\xea\x41\x6c\x3b"
	"kstrtouint\0\0"
	"\x10\x00\x00\x00\xba\x0c\x7a\x03"
	"kfree\0\0\0"
	"\x1c\x00\x00\x00\xdc\x90\xee\x82"
	"timer_delete_sync\0\0\0"
	"\x18\x00\x00\x00\xe7\x6f\xc5\xb2"
	"kernel_recvmsg\0\0"
	"\x20\x00\x00\x00\x0b\x05\xdb\x34"
	"_raw_spin_lock_irqsave\0\0"
	"\x18\x00\x00\x00\x64\xbd\x8f\xba"
	"_raw_spin_lock\0\0"
	"\x14\x00\x00\x00\xbb\x6d\xfb\xbd"
	"__fentry__\0\0"
	"\x10\x00\x00\x00\x7e\x3a\x2c\x12"
	"_printk\0"
	"\x14\x00\x00\x00\xc0\xce\x5f\xac"
	"in4_pton\0\0\0\0"
	"\x18\x00\x00\x00\x81\xc8\x24\x1d"
	"___ratelimit\0\0\0\0"
	"\x1c\x00\x00\x00\xcb\xf6\xfd\xf0"
	"__stack_chk_fail\0\0\0\0"
	"\x20\x00\x00\x00\xe9\x0d\x1d\x44"
	"__kmalloc_large_noprof\0\0"
	"\x18\x00\x00\x00\x4e\xba\x82\x76"
	"__copy_overflow\0"
	"\x10\x00\x00\x00\x94\xb6\x16\xa9"
	"strnlen\0"
	"\x18\x00\x00\x00\xe3\xb8\xcb\x0d"
	"const_pcpu_hot\0\0"
	"\x14\x00\x00\x00\x65\x52\x5a\xf9"
	"module_put\0\0"
	"\x28\x00\x00\x00\xb3\x1c\xa2\x87"
	"__ubsan_handle_out_of_bounds\0\0\0\0"
	"\x18\x00\x00\x00\x5a\x16\x6b\x47"
	"sized_strscpy\0\0\0"
	"\x14\x00\x00\x00\x5e\xc3\x64\xaa"
	"cdev_add\0\0\0\0"
	"\x1c\x00\x00\x00\x70\xab\x9d\x2c"
	"device_create_file\0\0"
	"\x14\x00\x00\x00\xeb\xda\x47\x70"
	"init_net\0\0\0\0"
	"\x14\x00\x00\x00\xcb\x69\x85\x8c"
	"kstrtoint\0\0\0"
	"\x14\x00\x00\x00\xb8\x83\x8c\xc3"
	"mod_timer\0\0\0"
	"\x18\x00\x00\x00\xfe\xc8\xfb\x9d"
	"device_create\0\0\0"
	"\x10\x00\x00\x00\xa8\x26\x6d\x1e"
	"strstr\0\0"
	"\x18\x00\x00\x00\x6b\xaa\x0f\xe7"
	"class_create\0\0\0\0"
	"\x1c\x00\x00\x00\x63\xa5\x03\x4c"
	"random_kmalloc_seed\0"
	"\x20\x00\x00\x00\x3d\x92\x52\x5d"
	"kernel_sock_shutdown\0\0\0\0"
	"\x1c\x00\x00\x00\x0c\xd2\x03\x8c"
	"destroy_workqueue\0\0\0"
	"\x14\x00\x00\x00\x4b\x8d\xfa\x4d"
	"mutex_lock\0\0"
	"\x10\x00\x00\x00\x11\x13\x92\x5a"
	"strncmp\0"
	"\x10\x00\x00\x00\xda\xfa\x66\x91"
	"strncpy\0"
	"\x10\x00\x00\x00\xe6\x6e\xab\xbc"
	"sscanf\0\0"
	"\x18\x00\x00\x00\x9f\x0c\xfb\xce"
	"__mutex_init\0\0\0\0"
	"\x18\x00\x00\x00\xb5\x79\xca\x75"
	"__fortify_panic\0"
	"\x28\x00\x00\x00\xc4\xcd\x36\x4e"
	"__ubsan_handle_divrem_overflow\0\0"
	"\x24\x00\x00\x00\x70\xce\x5c\xd3"
	"_raw_spin_unlock_irqrestore\0"
	"\x10\x00\x00\x00\xc7\x9a\x08\x11"
	"_ctype\0\0"
	"\x14\x00\x00\x00\x4d\xad\x4b\x12"
	"kstrtobool\0\0"
	"\x18\x00\x00\x00\x4e\x63\x34\xff"
	"kernel_connect\0\0"
	"\x10\x00\x00\x00\xc5\x8f\x57\xfb"
	"memset\0\0"
	"\x18\x00\x00\x00\xbc\x87\xef\xbf"
	"param_ops_charp\0"
	"\x1c\x00\x00\x00\x03\xfc\x66\x91"
	"__flush_workqueue\0\0\0"
	"\x1c\x00\x00\x00\xca\x39\x82\x5b"
	"__x86_return_thunk\0\0"
	"\x18\x00\x00\x00\xe1\xbe\x10\x6b"
	"_copy_to_user\0\0\0"
	"\x10\x00\x00\x00\x5a\x25\xd5\xe2"
	"strcmp\0\0"
	"\x10\x00\x00\x00\xa6\x50\xba\x15"
	"jiffies\0"
	"\x24\x00\x00\x00\x33\xb3\x91\x60"
	"unregister_chrdev_region\0\0\0\0"
	"\x18\x00\x00\x00\x38\xf0\x13\x32"
	"mutex_unlock\0\0\0\0"
	"\x1c\x00\x00\x00\x3c\x0b\x48\x2e"
	"sock_create_kern\0\0\0\0"
	"\x18\x00\x00\x00\x39\x63\xf4\xc6"
	"init_timer_key\0\0"
	"\x18\x00\x00\x00\x4b\xc2\x7c\x8c"
	"device_destroy\0\0"
	"\x1c\x00\x00\x00\x65\x62\xf5\x2c"
	"__dynamic_pr_debug\0\0"
	"\x20\x00\x00\x00\xee\xfb\xb4\x10"
	"__kmalloc_cache_noprof\0\0"
	"\x14\x00\x00\x00\x65\x93\x3f\xb4"
	"ktime_get\0\0\0"
	"\x18\x00\x00\x00\xf2\xde\xc3\x2f"
	"sock_release\0\0\0\0"
	"\x1c\x00\x00\x00\x09\x37\xed\x41"
	"get_random_bytes\0\0\0\0"
	"\x2c\x00\x00\x00\xc6\xfa\xb1\x54"
	"__ubsan_handle_load_invalid_value\0\0\0"
	"\x10\x00\x00\x00\x9c\x53\x4d\x75"
	"strlen\0\0"
	"\x18\x00\x00\x00\xa3\xe6\xaf\xef"
	"param_ops_int\0\0\0"
	"\x10\x00\x00\x00\x85\xba\x9c\x34"
	"strchr\0\0"
	"\x1c\x00\x00\x00\x34\x4b\xb5\xb5"
	"_raw_spin_unlock\0\0\0\0"
	"\x18\x00\x00\x00\xa2\xbf\x1f\x93"
	"kernel_sendmsg\0\0"
	"\x18\x00\x00\x00\xfa\x80\xfe\x6c"
	"sock_setsockopt\0"
	"\x14\x00\x00\x00\x1e\x70\xe8\x75"
	"cdev_init\0\0\0"
	"\x18\x00\x00\x00\x1d\x07\x60\x20"
	"kmalloc_caches\0\0"
	"\x14\x00\x00\x00\x69\x1b\x25\x12"
	"cdev_del\0\0\0\0"
	"\x1c\x00\x00\x00\x59\x50\xf1\xd6"
	"device_remove_file\0\0"
	"\x18\x00\x00\x00\xde\x9f\x8a\x25"
	"module_layout\0\0\0"
	"\x00\x00\x00\x00\x00\x00\x00\x00";

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "9A28325D725E4530A0ECC74");

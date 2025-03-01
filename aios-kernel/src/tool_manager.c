//
// Created by sina-mazaheri on 12/17/24.
//
/* Tool Management */
struct llm_tool *llm_tool_alloc(const char *name, const char *description)
{
    struct llm_tool *tool;

    if (!name || !description)
        return NULL;

    tool = kmalloc(sizeof(*tool), GFP_KERNEL);
    if (!tool)
        return NULL;

    strncpy(tool->name, name, MAX_TOOL_NAME - 1);
    tool->name[MAX_TOOL_NAME - 1] = '\0';

    strncpy(tool->description, description, MAX_TOOL_DESC - 1);
    tool->description[MAX_TOOL_DESC - 1] = '\0';

    INIT_LIST_HEAD(&tool->parameters);
    INIT_LIST_HEAD(&tool->list);

    return tool;
}
static void cleanup_tools(struct llm_config *config)
{
    struct llm_tool *tool, *tmp_tool;
    struct llm_tool_param *param, *tmp_param;

    list_for_each_entry_safe(tool, tmp_tool, &config->tools, list) {
        list_for_each_entry_safe(param, tmp_param, &tool->parameters, list) {
            list_del(&param->list);
            kfree(param);
        }
        list_del(&tool->list);
        llm_tool_free(tool);
    }
}
void llm_tool_free(struct llm_tool *tool)
{
    struct llm_tool_param *param, *tmp;

    if (!tool)
        return;

    list_for_each_entry_safe(param, tmp, &tool->parameters, list) {
        list_del(&param->list);
        kfree(param);
    }

    kfree(tool);
}

int llm_add_tool_param(struct llm_tool *tool, const char *name,
                      const char *description, bool required)
{
    struct llm_tool_param *param;

    if (!tool || !name || !description)
        return -EINVAL;

    param = kmalloc(sizeof(*param), GFP_KERNEL);
    if (!param)
        return -ENOMEM;

    strncpy(param->name, name, MAX_TOOL_NAME - 1);
    param->name[MAX_TOOL_NAME - 1] = '\0';

    strncpy(param->description, description, MAX_TOOL_DESC - 1);
    param->description[MAX_TOOL_DESC - 1] = '\0';

    param->required = required;
    INIT_LIST_HEAD(&param->list);

    list_add_tail(&param->list, &tool->parameters);
    return 0;
}

/* Replace global mutex with config-specific locks */
int llm_add_tool(struct llm_config *config, struct llm_tool *tool) {
    if (!config || !tool)
        return -EINVAL;

    mutex_lock(&config->tool_lock);
    list_add_tail(&tool->list, &config->tools);
    mutex_unlock(&config->tool_lock);
    return 0;
}
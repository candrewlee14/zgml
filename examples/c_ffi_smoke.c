#include "zgml.h"

#include <stdint.h>
#include <stdio.h>

static int expect_status(const char *label, zgml_status status) {
    if (status == ZGML_OK) return 0;
    fprintf(stderr, "%s: %s (%d)\n", label, zgml_status_name(status), status);
    return 1;
}

static int expect_close(float actual, float expected) {
    float diff = actual - expected;
    if (diff < 0.0f) diff = -diff;
    return diff <= 0.00001f;
}

static uint64_t command_category_total(zgml_program_inspection inspection) {
    return inspection.command_op_count +
        inspection.command_row_count +
        inspection.command_projection_count +
        inspection.command_attention_count +
        inspection.command_movement_count +
        inspection.command_elementwise_count +
        inspection.command_rope_count;
}

static void touch_current_abi_surface(void) {
    const uint64_t required_features =
        (uint64_t)ZGML_FEATURE_BUFFER_HANDLE |
        (uint64_t)ZGML_FEATURE_MODEL_AUTO |
        (uint64_t)ZGML_FEATURE_RUNTIME_PROFILE |
        (uint64_t)ZGML_FEATURE_WEBGPU_COMPILE_ONLY |
        (uint64_t)ZGML_FEATURE_WASM_EXPORTS |
        (uint64_t)ZGML_FEATURE_NATIVE_BUFFER_IO |
        (uint64_t)ZGML_FEATURE_NATIVE_ARGMAX |
        (uint64_t)ZGML_FEATURE_NATIVE_EXECUTE_ARGMAX |
        (uint64_t)ZGML_FEATURE_NATIVE_TOPK_SAMPLE |
        (uint64_t)ZGML_FEATURE_PROGRAM_REQUIREMENTS |
        (uint64_t)ZGML_FEATURE_PROGRAM_OUTPUT_BUFFER |
        (uint64_t)ZGML_FEATURE_NATIVE_GENERATE_SAMPLE |
        (uint64_t)ZGML_FEATURE_NATIVE_GENERATE_ARGMAX |
        (uint64_t)ZGML_FEATURE_SESSION_MODEL_BINDING |
        (uint64_t)ZGML_FEATURE_EXTERNAL_BUFFER |
        (uint64_t)ZGML_FEATURE_EXTERNAL_RESOURCE_BUFFER |
        (uint64_t)ZGML_FEATURE_LLAMA_KV_RESOURCE_BINDING |
        (uint64_t)ZGML_FEATURE_LLAMA_KV_CACHE_REQUIREMENTS |
        (uint64_t)ZGML_FEATURE_PROGRAM_BUFFER_FACTORY |
        (uint64_t)ZGML_FEATURE_EXTERNAL_RESOURCE_ACCESS |
        (uint64_t)ZGML_FEATURE_MODEL_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_MODEL_COMPATIBILITY |
        (uint64_t)ZGML_FEATURE_BUFFER_INSPECTION |
        (uint64_t)ZGML_FEATURE_SESSION_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_RESOURCE_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_MEMORY_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_SHAPE_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_PATCH_ENVELOPE_INSPECTION |
        (uint64_t)ZGML_FEATURE_SESSION_BINDING_SHAPE_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_DEVICE_BUFFER |
        (uint64_t)ZGML_FEATURE_PROGRAM_DEVICE_BUFFER_IMPORT |
        (uint64_t)ZGML_FEATURE_PROGRAM_DISPATCH_PLAN_INSPECTION |
        (uint64_t)ZGML_FEATURE_ABI_STRUCT_SIZE |
        (uint64_t)ZGML_FEATURE_MODEL_PATH_PROBE |
        (uint64_t)ZGML_FEATURE_SUPPORTED_CHECKPOINTS |
        (uint64_t)ZGML_FEATURE_SAFETENSORS_HEADER_PROBE |
        (uint64_t)ZGML_FEATURE_SAFETENSORS_DATA_LOAD |
        (uint64_t)ZGML_FEATURE_SAFETENSORS_DATA_PROBE |
        (uint64_t)ZGML_FEATURE_NATIVE_TINY_MLP |
        (uint64_t)ZGML_FEATURE_NATIVE_MODULE_PROGRAM |
        (uint64_t)ZGML_FEATURE_PROGRAM_BINDING_REQUIREMENTS |
        (uint64_t)ZGML_FEATURE_SESSION_PERSISTENT_UPLOAD |
        (uint64_t)ZGML_FEATURE_NATIVE_MODULE_ACTIVATION_CHAIN;
    const uint64_t optional_features =
        (uint64_t)ZGML_FEATURE_NATIVE_WGPU_EXECUTION |
        (uint64_t)ZGML_FEATURE_EXPERIMENTAL_LLAMA_WGPU_EXECUTION;
    const uint32_t abi_structs[] = {
        ZGML_ABI_STRUCT_RUNTIME_INFO,
        ZGML_ABI_STRUCT_MODEL_DESC,
        ZGML_ABI_STRUCT_MODEL_LOAD_DESC,
        ZGML_ABI_STRUCT_MODEL_INSPECTION,
        ZGML_ABI_STRUCT_PROGRAM_MODEL_COMPATIBILITY,
        ZGML_ABI_STRUCT_SESSION_INSPECTION,
        ZGML_ABI_STRUCT_COMPILE_DESC,
        ZGML_ABI_STRUCT_BIND_DESC,
        ZGML_ABI_STRUCT_BUFFER_DESC,
        ZGML_ABI_STRUCT_BUFFER_INSPECTION,
        ZGML_ABI_STRUCT_EXTERNAL_RESOURCE_DESC,
        ZGML_ABI_STRUCT_DEVICE_BUFFER_IMPORT_DESC,
        ZGML_ABI_STRUCT_BUFFER_BIND_DESC,
        ZGML_ABI_STRUCT_LLAMA_KV_CACHE_BIND_DESC,
        ZGML_ABI_STRUCT_LLAMA_BUFFER_BIND_DESC,
        ZGML_ABI_STRUCT_STEP_DESC,
        ZGML_ABI_STRUCT_STEP_RESULT,
        ZGML_ABI_STRUCT_TOKEN_STEP_DESC,
        ZGML_ABI_STRUCT_TOKEN_ADVANCE_DESC,
        ZGML_ABI_STRUCT_TOKEN_ADVANCE_TOKENS_DESC,
        ZGML_ABI_STRUCT_TOKEN_PREFILL_DESC,
        ZGML_ABI_STRUCT_TOKEN_EXECUTE_DESC,
        ZGML_ABI_STRUCT_TOKEN_ARGMAX_DESC,
        ZGML_ABI_STRUCT_TOKEN_ARGMAX_RESULT,
        ZGML_ABI_STRUCT_TOKEN_EXECUTE_ARGMAX_DESC,
        ZGML_ABI_STRUCT_TOKEN_GENERATE_ARGMAX_DESC,
        ZGML_ABI_STRUCT_TOKEN_GENERATE_ARGMAX_RESULT,
        ZGML_ABI_STRUCT_TOKEN_SAMPLE_DESC,
        ZGML_ABI_STRUCT_TOKEN_SAMPLE_RESULT,
        ZGML_ABI_STRUCT_TOKEN_EXECUTE_SAMPLE_DESC,
        ZGML_ABI_STRUCT_TOKEN_GENERATE_SAMPLE_DESC,
        ZGML_ABI_STRUCT_TOKEN_GENERATE_SAMPLE_RESULT,
        ZGML_ABI_STRUCT_PROGRAM_REQUIREMENTS,
        ZGML_ABI_STRUCT_LLAMA_KV_CACHE_REQUIREMENTS,
        ZGML_ABI_STRUCT_PROGRAM_INSPECTION,
        ZGML_ABI_STRUCT_LLAMA_PROGRAM_INSPECTION,
        ZGML_ABI_STRUCT_RUNTIME_PROFILE,
        ZGML_ABI_STRUCT_SAFETENSORS_HEADER_PROBE_DESC,
        ZGML_ABI_STRUCT_SAFETENSORS_DATA_LOAD_DESC,
        ZGML_ABI_STRUCT_MODULE_OP_DESC,
        ZGML_ABI_STRUCT_MODULE_DESC,
    };
    const uint32_t model_kinds[] = {
        ZGML_MODEL_AUTO,
        ZGML_MODEL_TINY_LINEAR,
        ZGML_MODEL_TINY_LLAMA,
        ZGML_MODEL_SMOLLM_135M,
        ZGML_MODEL_TINY_LLAMA_2LAYER,
        ZGML_MODEL_TINY_MLP,
        ZGML_MODEL_MODULE,
        ZGML_MODEL_LLAMA_FAMILY,
        ZGML_MODEL_SMOLLM2_360M,
    };
    const uint32_t tiny_mlp_activations[] = {
        ZGML_TINY_MLP_ACTIVATION_RELU,
        ZGML_TINY_MLP_ACTIVATION_GELU,
        ZGML_TINY_MLP_ACTIVATION_SILU,
        ZGML_TINY_MLP_ACTIVATION_SIGMOID,
    };
    const uint32_t module_ops[] = {
        ZGML_MODULE_OP_LINEAR,
        ZGML_MODULE_OP_ACTIVATION,
        ZGML_MODULE_OP_SOFTMAX,
        ZGML_MODULE_OP_LAYER_NORM,
        ZGML_MODULE_OP_RMS_NORM,
        ZGML_MODULE_OP_EMBEDDING,
        ZGML_MODULE_OP_LOG_SOFTMAX,
        ZGML_MODULE_OP_RESHAPE,
        ZGML_MODULE_OP_BROADCAST_TO,
        ZGML_MODULE_OP_NARROW,
        ZGML_MODULE_OP_TRANSPOSE,
        ZGML_MODULE_OP_REDUCE_SUM,
        ZGML_MODULE_OP_REDUCE_MEAN,
        ZGML_MODULE_OP_REDUCE_MAX,
        ZGML_MODULE_OP_SLICE,
        ZGML_MODULE_OP_ACTIVATION_CHAIN,
    };
    const uint32_t module_activations[] = {
        ZGML_MODULE_ACTIVATION_RELU,
        ZGML_MODULE_ACTIVATION_GELU,
        ZGML_MODULE_ACTIVATION_SILU,
        ZGML_MODULE_ACTIVATION_SIGMOID,
        ZGML_MODULE_ACTIVATION_EXP,
        ZGML_MODULE_ACTIVATION_LOG,
        ZGML_MODULE_ACTIVATION_NEG,
        ZGML_MODULE_ACTIVATION_RECIP,
        ZGML_MODULE_ACTIVATION_ABS,
        ZGML_MODULE_ACTIVATION_SQRT,
        ZGML_MODULE_ACTIVATION_SQUARE,
        ZGML_MODULE_ACTIVATION_SQR,
        ZGML_MODULE_ACTIVATION_SGN,
        ZGML_MODULE_ACTIVATION_SIGN,
        ZGML_MODULE_ACTIVATION_STEP,
    };
    const uint32_t backends[] = {
        ZGML_BACKEND_AUTO,
        ZGML_BACKEND_CPU,
        ZGML_BACKEND_METAL,
        ZGML_BACKEND_WEBGPU,
    };
    const uint32_t execute_policies[] = {
        ZGML_EXECUTE_OUTPUT_NONE,
        ZGML_EXECUTE_OUTPUT_LOGITS,
    };
    const uint32_t program_buffer_kinds[] = {
        ZGML_PROGRAM_BUFFER_WEIGHTS,
        ZGML_PROGRAM_BUFFER_BIAS,
        ZGML_PROGRAM_BUFFER_INPUT,
        ZGML_PROGRAM_BUFFER_OUTPUT,
    };
    const uint32_t resource_access_flags[] = {
        ZGML_RESOURCE_ACCESS_READ,
        ZGML_RESOURCE_ACCESS_WRITE,
        ZGML_RESOURCE_ACCESS_READ_WRITE,
    };
    const uint32_t buffer_storage_kinds[] = {
        ZGML_BUFFER_STORAGE_NONE,
        ZGML_BUFFER_STORAGE_HOST,
        ZGML_BUFFER_STORAGE_EXTERNAL_RESOURCE,
    };
    zgml_status (*model_load_path_fn)(const zgml_model_load_desc *, zgml_model **) = zgml_model_load_path;
    zgml_status (*model_load_safetensors_data_fn)(const zgml_safetensors_data_load_desc *, zgml_model **) = zgml_model_load_safetensors_data;
    zgml_status (*model_probe_path_fn)(const zgml_model_load_desc *, zgml_model_inspection *) = zgml_model_probe_path;
    zgml_status (*model_probe_safetensors_data_fn)(const zgml_safetensors_data_load_desc *, zgml_model_inspection *) = zgml_model_probe_safetensors_data;
    zgml_status (*model_probe_safetensors_header_fn)(const zgml_safetensors_header_probe_desc *, zgml_model_inspection *) = zgml_model_probe_safetensors_header;
    size_t (*supported_checkpoint_count_fn)(void) = zgml_supported_checkpoint_count;
    zgml_status (*supported_checkpoint_inspect_fn)(size_t, zgml_model_inspection *) = zgml_supported_checkpoint_inspect;
    zgml_status (*model_inspect_fn)(zgml_model *, zgml_model_inspection *) = zgml_model_inspect;
    zgml_status (*module_program_compile_fn)(const zgml_module_desc *, const zgml_compile_desc *, zgml_program **) = zgml_module_program_compile;
    zgml_status (*requirements_fn)(zgml_program *, zgml_program_requirements *) = zgml_program_get_requirements;
    zgml_status (*model_compatibility_fn)(zgml_program *, zgml_model *, zgml_program_model_compatibility *) = zgml_program_check_model_compatibility;
    zgml_status (*create_buffer_fn)(zgml_program *, uint32_t, zgml_buffer **) = zgml_program_create_buffer;
    zgml_status (*create_device_buffer_fn)(zgml_program *, uint32_t, uint32_t, zgml_buffer **) = zgml_program_create_device_buffer;
    zgml_status (*get_device_handle_fn)(zgml_program *, uint32_t, uintptr_t *) = zgml_program_get_device_handle;
    zgml_status (*import_device_buffer_fn)(zgml_program *, uint32_t, const zgml_device_buffer_import_desc *, zgml_buffer **) = zgml_program_import_device_buffer;
    zgml_status (*create_output_buffer_fn)(zgml_program *, zgml_buffer **) = zgml_program_create_output_buffer;
    zgml_status (*llama_inspect_fn)(zgml_program *, zgml_llama_program_inspection *) = zgml_llama_program_inspect;
    zgml_status (*llama_kv_requirements_fn)(zgml_program *, zgml_llama_kv_cache_requirements *) = zgml_llama_program_get_kv_cache_requirements;
    zgml_status (*program_reset_profile_fn)(zgml_program *) = zgml_program_reset_runtime_profile;
    zgml_status (*bind_model_fn)(zgml_program *, zgml_model *, const zgml_bind_desc *, zgml_session **) = zgml_session_bind_model;
    zgml_status (*bind_model_buffers_fn)(zgml_program *, zgml_model *, const zgml_buffer_bind_desc *, zgml_session **) = zgml_session_bind_model_buffers;
    zgml_status (*step_no_output_fn)(zgml_session *, const zgml_step_desc *, zgml_step_result *) = zgml_session_step_no_output;
    zgml_status (*advance_token_fn)(zgml_session *, const zgml_token_advance_desc *) = zgml_session_advance_token;
    zgml_status (*advance_tokens_fn)(zgml_session *, const zgml_token_advance_tokens_desc *) = zgml_session_advance_tokens;
    zgml_status (*prefill_tokens_fn)(zgml_session *, const zgml_token_prefill_desc *, zgml_step_result *) = zgml_session_prefill_tokens;
    zgml_status (*execute_tokens_fn)(zgml_session *, const zgml_token_execute_desc *, zgml_step_result *) = zgml_session_execute_tokens;
    zgml_status (*argmax_token_fn)(zgml_session *, const zgml_token_argmax_desc *, zgml_token_argmax_result *) = zgml_session_argmax_token;
    zgml_status (*execute_argmax_fn)(zgml_session *, const zgml_token_execute_argmax_desc *, zgml_token_argmax_result *) = zgml_session_execute_argmax_tokens;
    zgml_status (*generate_argmax_fn)(zgml_session *, const zgml_token_generate_argmax_desc *, zgml_token_generate_argmax_result *) = zgml_session_generate_argmax_tokens;
    zgml_status (*sample_token_fn)(zgml_session *, const zgml_token_sample_desc *, zgml_token_sample_result *) = zgml_session_sample_token;
    zgml_status (*execute_sample_fn)(zgml_session *, const zgml_token_execute_sample_desc *, zgml_token_sample_result *) = zgml_session_execute_sample_tokens;
    zgml_status (*generate_sample_fn)(zgml_session *, const zgml_token_generate_sample_desc *, zgml_token_generate_sample_result *) = zgml_session_generate_sample_tokens;
    zgml_status (*position_fn)(zgml_session *, size_t *) = zgml_session_position;
    zgml_status (*session_inspect_fn)(zgml_session *, zgml_session_inspection *) = zgml_session_inspect;
    zgml_status (*reset_fn)(zgml_session *) = zgml_session_reset;
    size_t (*abi_struct_size_fn)(uint32_t) = zgml_abi_struct_size;
    void *(*buffer_data_fn)(zgml_buffer *) = zgml_buffer_data;
    zgml_status (*buffer_inspect_fn)(zgml_buffer *, zgml_buffer_inspection *) = zgml_buffer_inspect;
    zgml_status (*buffer_wrap_fn)(void *, size_t, zgml_buffer **) = zgml_buffer_wrap;
    zgml_status (*buffer_wrap_resource_fn)(const zgml_external_resource_desc *, zgml_buffer **) = zgml_buffer_wrap_resource;
    size_t (*wasm_alloc_fn)(size_t) = zgml_wasm_alloc;
    void (*wasm_free_fn)(size_t, size_t) = zgml_wasm_free;
    const zgml_external_resource_desc external_resource_desc = {
        .placement = ZGML_BACKEND_WEBGPU,
        .access_flags = ZGML_RESOURCE_ACCESS_READ_WRITE,
        .handle = (uintptr_t)1,
        .byte_offset = 0,
        .byte_len = 16,
    };
    const zgml_device_buffer_import_desc device_buffer_import_desc = {
        .placement = ZGML_BACKEND_WEBGPU,
        .device_handle = (uintptr_t)1,
        .buffer_handle = (uintptr_t)2,
        .byte_offset = 0,
        .byte_len = 16,
    };
    const zgml_safetensors_header_probe_desc safetensors_header_probe_desc = {
        .kind = ZGML_MODEL_AUTO,
        .reserved = 0,
        .header = "{}",
        .header_len = 2,
    };
    const zgml_safetensors_data_load_desc safetensors_data_load_desc = {
        .kind = ZGML_MODEL_AUTO,
        .reserved = 0,
        .data = "{}",
        .data_len = 2,
    };
    const zgml_model_inspection model_inspection = {
        .model_kind = ZGML_MODEL_TINY_LINEAR,
        .input_len = 2,
        .output_len = 3,
    };
    const zgml_program_model_compatibility model_compatibility = {
        .program_model_kind = ZGML_MODEL_TINY_LINEAR,
        .model_kind = ZGML_MODEL_TINY_LINEAR,
        .compatible = 0,
    };
    const zgml_buffer_inspection buffer_inspection = {
        .storage = ZGML_BUFFER_STORAGE_EXTERNAL_RESOURCE,
        .placement = ZGML_BACKEND_WEBGPU,
        .access_flags = ZGML_RESOURCE_ACCESS_WRITE,
        .byte_len = 16,
        .handle = 1,
    };
    const zgml_session_inspection session_inspection = {
        .model_kind = ZGML_MODEL_TINY_LINEAR,
        .backend = ZGML_BACKEND_CPU,
        .output_storage = ZGML_BUFFER_STORAGE_HOST,
    };

    (void)required_features;
    (void)optional_features;
    (void)abi_structs;
    (void)model_kinds;
    (void)tiny_mlp_activations;
    (void)module_ops;
    (void)module_activations;
    (void)backends;
    (void)execute_policies;
    (void)program_buffer_kinds;
    (void)resource_access_flags;
    (void)buffer_storage_kinds;
    (void)model_load_path_fn;
    (void)model_load_safetensors_data_fn;
    (void)model_probe_path_fn;
    (void)model_probe_safetensors_data_fn;
    (void)model_probe_safetensors_header_fn;
    (void)supported_checkpoint_count_fn;
    (void)supported_checkpoint_inspect_fn;
    (void)model_inspect_fn;
    (void)module_program_compile_fn;
    (void)requirements_fn;
    (void)model_compatibility_fn;
    (void)create_buffer_fn;
    (void)create_device_buffer_fn;
    (void)get_device_handle_fn;
    (void)import_device_buffer_fn;
    (void)create_output_buffer_fn;
    (void)llama_inspect_fn;
    (void)llama_kv_requirements_fn;
    (void)program_reset_profile_fn;
    (void)bind_model_fn;
    (void)bind_model_buffers_fn;
    (void)step_no_output_fn;
    (void)advance_token_fn;
    (void)advance_tokens_fn;
    (void)prefill_tokens_fn;
    (void)execute_tokens_fn;
    (void)argmax_token_fn;
    (void)execute_argmax_fn;
    (void)generate_argmax_fn;
    (void)sample_token_fn;
    (void)execute_sample_fn;
    (void)generate_sample_fn;
    (void)position_fn;
    (void)session_inspect_fn;
    (void)reset_fn;
    (void)abi_struct_size_fn;
    (void)buffer_data_fn;
    (void)buffer_inspect_fn;
    (void)buffer_wrap_fn;
    (void)buffer_wrap_resource_fn;
    (void)external_resource_desc;
    (void)device_buffer_import_desc;
    (void)safetensors_header_probe_desc;
    (void)safetensors_data_load_desc;
    (void)model_inspection;
    (void)model_compatibility;
    (void)buffer_inspection;
    (void)session_inspection;
    (void)wasm_alloc_fn;
    (void)wasm_free_fn;
}

int main(void) {
    touch_current_abi_surface();

    zgml_model *model = NULL;
    zgml_program *program = NULL;
    zgml_session *session = NULL;
    zgml_buffer *weights_buffer = NULL;
    zgml_buffer *bias_buffer = NULL;
    zgml_buffer *input_buffer = NULL;
    zgml_buffer *output_buffer = NULL;
    int failed = 0;

    zgml_runtime_info runtime_info = {0};
    failed |= expect_status("zgml_get_runtime_info", zgml_get_runtime_info(&runtime_info));
    if (!failed && runtime_info.abi_version != ZGML_ABI_VERSION) {
        fprintf(stderr, "expected ABI version %u, got %u\n", ZGML_ABI_VERSION, runtime_info.abi_version);
        failed = 1;
    }
    if (!failed && runtime_info.size_t_bytes != sizeof(size_t)) {
        fprintf(stderr, "unexpected size_t width %u\n", runtime_info.size_t_bytes);
        failed = 1;
    }
    zgml_model_inspection path_probe = {0};
    zgml_status path_probe_status = zgml_model_probe_path(NULL, &path_probe);
    if (!failed && path_probe_status != ZGML_INVALID_ARGUMENT) {
        fprintf(stderr, "expected zgml_model_probe_path invalid_argument, got %s (%d)\n", zgml_status_name(path_probe_status), path_probe_status);
        failed = 1;
    }
    zgml_status header_probe_status = zgml_model_probe_safetensors_header(NULL, &path_probe);
    if (!failed && header_probe_status != ZGML_INVALID_ARGUMENT) {
        fprintf(stderr, "expected zgml_model_probe_safetensors_header invalid_argument, got %s (%d)\n", zgml_status_name(header_probe_status), header_probe_status);
        failed = 1;
    }
    size_t supported_count = zgml_supported_checkpoint_count();
    if (!failed && supported_count != 4) {
        fprintf(stderr, "expected 4 supported checkpoint envelopes, got %zu\n", supported_count);
        failed = 1;
    }
    zgml_model_inspection supported_checkpoint = {0};
    if (!failed) {
        failed |= expect_status("zgml_supported_checkpoint_inspect", zgml_supported_checkpoint_inspect(0, &supported_checkpoint));
    }
    if (!failed && (
        supported_checkpoint.model_kind != ZGML_MODEL_TINY_LLAMA ||
        supported_checkpoint.vocab_size != 8 ||
        supported_checkpoint.d_model != 4 ||
        supported_checkpoint.n_layers != 1 ||
        supported_checkpoint.n_kv_heads != 1 ||
        supported_checkpoint.tied_lm_head != 0
    )) {
        fprintf(stderr, "unexpected tiny LLaMA supported checkpoint envelope: kind=%u vocab=%llu d_model=%llu layers=%llu kv_heads=%llu tied=%llu\n",
            supported_checkpoint.model_kind,
            (unsigned long long)supported_checkpoint.vocab_size,
            (unsigned long long)supported_checkpoint.d_model,
            (unsigned long long)supported_checkpoint.n_layers,
            (unsigned long long)supported_checkpoint.n_kv_heads,
            (unsigned long long)supported_checkpoint.tied_lm_head);
        failed = 1;
    }
    supported_checkpoint = (zgml_model_inspection){0};
    if (!failed) {
        failed |= expect_status("zgml_supported_checkpoint_inspect", zgml_supported_checkpoint_inspect(1, &supported_checkpoint));
    }
    if (!failed && (
        supported_checkpoint.model_kind != ZGML_MODEL_TINY_LLAMA_2LAYER ||
        supported_checkpoint.vocab_size != 8 ||
        supported_checkpoint.d_model != 4 ||
        supported_checkpoint.n_layers != 2 ||
        supported_checkpoint.n_kv_heads != 1 ||
        supported_checkpoint.tied_lm_head != 0
    )) {
        fprintf(stderr, "unexpected two-layer tiny LLaMA supported checkpoint envelope: kind=%u vocab=%llu d_model=%llu layers=%llu kv_heads=%llu tied=%llu\n",
            supported_checkpoint.model_kind,
            (unsigned long long)supported_checkpoint.vocab_size,
            (unsigned long long)supported_checkpoint.d_model,
            (unsigned long long)supported_checkpoint.n_layers,
            (unsigned long long)supported_checkpoint.n_kv_heads,
            (unsigned long long)supported_checkpoint.tied_lm_head);
        failed = 1;
    }
    supported_checkpoint = (zgml_model_inspection){0};
    if (!failed) {
        failed |= expect_status("zgml_supported_checkpoint_inspect", zgml_supported_checkpoint_inspect(2, &supported_checkpoint));
    }
    if (!failed && (
        supported_checkpoint.model_kind != ZGML_MODEL_SMOLLM_135M ||
        supported_checkpoint.vocab_size != 49152 ||
        supported_checkpoint.d_model != 576 ||
        supported_checkpoint.n_layers != 30 ||
        supported_checkpoint.n_kv_heads != 3 ||
        supported_checkpoint.tied_lm_head != 1
    )) {
        fprintf(stderr, "unexpected SmolLM supported checkpoint envelope: kind=%u vocab=%llu d_model=%llu layers=%llu kv_heads=%llu tied=%llu\n",
            supported_checkpoint.model_kind,
            (unsigned long long)supported_checkpoint.vocab_size,
            (unsigned long long)supported_checkpoint.d_model,
            (unsigned long long)supported_checkpoint.n_layers,
            (unsigned long long)supported_checkpoint.n_kv_heads,
            (unsigned long long)supported_checkpoint.tied_lm_head);
        failed = 1;
    }
    supported_checkpoint = (zgml_model_inspection){0};
    if (!failed) {
        failed |= expect_status("zgml_supported_checkpoint_inspect", zgml_supported_checkpoint_inspect(3, &supported_checkpoint));
    }
    if (!failed && (
        supported_checkpoint.model_kind != ZGML_MODEL_SMOLLM2_360M ||
        supported_checkpoint.vocab_size != 49152 ||
        supported_checkpoint.d_model != 960 ||
        supported_checkpoint.n_layers != 32 ||
        supported_checkpoint.n_kv_heads != 5 ||
        supported_checkpoint.tied_lm_head != 1
    )) {
        fprintf(stderr, "unexpected SmolLM2 supported checkpoint envelope: kind=%u vocab=%llu d_model=%llu layers=%llu kv_heads=%llu tied=%llu\n",
            supported_checkpoint.model_kind,
            (unsigned long long)supported_checkpoint.vocab_size,
            (unsigned long long)supported_checkpoint.d_model,
            (unsigned long long)supported_checkpoint.n_layers,
            (unsigned long long)supported_checkpoint.n_kv_heads,
            (unsigned long long)supported_checkpoint.tied_lm_head);
        failed = 1;
    }
    supported_checkpoint.model_kind = 99;
    supported_checkpoint.vocab_size = 123;
    zgml_status invalid_supported_status = zgml_supported_checkpoint_inspect(supported_count, &supported_checkpoint);
    if (!failed && invalid_supported_status != ZGML_INVALID_ARGUMENT) {
        fprintf(stderr, "expected unsupported checkpoint invalid_argument, got %s (%d)\n", zgml_status_name(invalid_supported_status), invalid_supported_status);
        failed = 1;
    }
    if (!failed && (supported_checkpoint.model_kind != 0 || supported_checkpoint.vocab_size != 0)) {
        fprintf(stderr, "expected invalid supported checkpoint inspect to clear output\n");
        failed = 1;
    }
    if (!failed && runtime_info.token_id_bytes != sizeof(uint32_t)) {
        fprintf(stderr, "unexpected token id width %u\n", runtime_info.token_id_bytes);
        failed = 1;
    }
    const uint64_t required_features =
        (uint64_t)ZGML_FEATURE_BUFFER_HANDLE |
        (uint64_t)ZGML_FEATURE_MODEL_AUTO |
        (uint64_t)ZGML_FEATURE_RUNTIME_PROFILE |
        (uint64_t)ZGML_FEATURE_WEBGPU_COMPILE_ONLY |
        (uint64_t)ZGML_FEATURE_WASM_EXPORTS |
        (uint64_t)ZGML_FEATURE_NATIVE_BUFFER_IO |
        (uint64_t)ZGML_FEATURE_NATIVE_ARGMAX |
        (uint64_t)ZGML_FEATURE_NATIVE_EXECUTE_ARGMAX |
        (uint64_t)ZGML_FEATURE_NATIVE_TOPK_SAMPLE |
        (uint64_t)ZGML_FEATURE_PROGRAM_REQUIREMENTS |
        (uint64_t)ZGML_FEATURE_PROGRAM_OUTPUT_BUFFER |
        (uint64_t)ZGML_FEATURE_NATIVE_GENERATE_SAMPLE |
        (uint64_t)ZGML_FEATURE_NATIVE_GENERATE_ARGMAX |
        (uint64_t)ZGML_FEATURE_SESSION_MODEL_BINDING |
        (uint64_t)ZGML_FEATURE_EXTERNAL_BUFFER |
        (uint64_t)ZGML_FEATURE_EXTERNAL_RESOURCE_BUFFER |
        (uint64_t)ZGML_FEATURE_LLAMA_KV_RESOURCE_BINDING |
        (uint64_t)ZGML_FEATURE_LLAMA_KV_CACHE_REQUIREMENTS |
        (uint64_t)ZGML_FEATURE_PROGRAM_BUFFER_FACTORY |
        (uint64_t)ZGML_FEATURE_EXTERNAL_RESOURCE_ACCESS |
        (uint64_t)ZGML_FEATURE_MODEL_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_MODEL_COMPATIBILITY |
        (uint64_t)ZGML_FEATURE_BUFFER_INSPECTION |
        (uint64_t)ZGML_FEATURE_SESSION_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_RESOURCE_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_MEMORY_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_SHAPE_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_PATCH_ENVELOPE_INSPECTION |
        (uint64_t)ZGML_FEATURE_SESSION_BINDING_SHAPE_INSPECTION |
        (uint64_t)ZGML_FEATURE_PROGRAM_DEVICE_BUFFER |
        (uint64_t)ZGML_FEATURE_PROGRAM_DEVICE_BUFFER_IMPORT |
        (uint64_t)ZGML_FEATURE_PROGRAM_DISPATCH_PLAN_INSPECTION |
        (uint64_t)ZGML_FEATURE_ABI_STRUCT_SIZE |
        (uint64_t)ZGML_FEATURE_MODEL_PATH_PROBE |
        (uint64_t)ZGML_FEATURE_SUPPORTED_CHECKPOINTS |
        (uint64_t)ZGML_FEATURE_SAFETENSORS_HEADER_PROBE |
        (uint64_t)ZGML_FEATURE_SAFETENSORS_DATA_LOAD |
        (uint64_t)ZGML_FEATURE_SAFETENSORS_DATA_PROBE |
        (uint64_t)ZGML_FEATURE_NATIVE_TINY_MLP |
        (uint64_t)ZGML_FEATURE_NATIVE_MODULE_PROGRAM |
        (uint64_t)ZGML_FEATURE_PROGRAM_BINDING_REQUIREMENTS |
        (uint64_t)ZGML_FEATURE_SESSION_PERSISTENT_UPLOAD |
        (uint64_t)ZGML_FEATURE_NATIVE_MODULE_ACTIVATION_CHAIN;
    if (!failed && (runtime_info.feature_flags & required_features) != required_features) {
        fprintf(stderr, "missing runtime feature flags: expected mask %llu, got %llu\n", (unsigned long long)required_features, (unsigned long long)runtime_info.feature_flags);
        failed = 1;
    }
    const int runtime_reports_native_wgpu = (runtime_info.feature_flags & (uint64_t)ZGML_FEATURE_NATIVE_WGPU_EXECUTION) != 0;
    const int runtime_reports_experimental_llama_wgpu = (runtime_info.feature_flags & (uint64_t)ZGML_FEATURE_EXPERIMENTAL_LLAMA_WGPU_EXECUTION) != 0;
    if (!failed && runtime_reports_experimental_llama_wgpu && !runtime_reports_native_wgpu) {
        fprintf(stderr, "experimental LLaMA WebGPU feature set without native wgpu execution\n");
        failed = 1;
    }
    if (!failed && zgml_abi_struct_size(ZGML_ABI_STRUCT_RUNTIME_INFO) != sizeof(zgml_runtime_info)) {
        fprintf(stderr, "unexpected runtime info struct size\n");
        failed = 1;
    }
    if (!failed && zgml_abi_struct_size(ZGML_ABI_STRUCT_EXTERNAL_RESOURCE_DESC) != sizeof(zgml_external_resource_desc)) {
        fprintf(stderr, "unexpected external resource descriptor size\n");
        failed = 1;
    }
    if (!failed && zgml_abi_struct_size(ZGML_ABI_STRUCT_DEVICE_BUFFER_IMPORT_DESC) != sizeof(zgml_device_buffer_import_desc)) {
        fprintf(stderr, "unexpected device-buffer import descriptor size\n");
        failed = 1;
    }
    if (!failed && zgml_abi_struct_size(ZGML_ABI_STRUCT_PROGRAM_INSPECTION) != sizeof(zgml_program_inspection)) {
        fprintf(stderr, "unexpected program inspection struct size\n");
        failed = 1;
    }
    if (!failed && zgml_abi_struct_size(ZGML_ABI_STRUCT_RUNTIME_PROFILE) != sizeof(zgml_runtime_profile)) {
        fprintf(stderr, "unexpected runtime profile struct size\n");
        failed = 1;
    }
    if (!failed && zgml_abi_struct_size(ZGML_ABI_STRUCT_MODULE_OP_DESC) != sizeof(zgml_module_op_desc)) {
        fprintf(stderr, "unexpected module op descriptor size\n");
        failed = 1;
    }
    if (!failed && zgml_abi_struct_size(ZGML_ABI_STRUCT_MODULE_DESC) != sizeof(zgml_module_desc)) {
        fprintf(stderr, "unexpected module descriptor size\n");
        failed = 1;
    }
    if (!failed && zgml_abi_struct_size(0) != 0) {
        fprintf(stderr, "unknown ABI struct kind returned a size\n");
        failed = 1;
    }

    zgml_program *module_program = NULL;
    zgml_session *module_session = NULL;
    const size_t module_input_shape[] = {3};
    const zgml_module_op_desc module_ops_abs_sqrt_square_sgn_step[] = {
        {
            .kind = ZGML_MODULE_OP_ACTIVATION_CHAIN,
            .activation = ZGML_MODULE_ACTIVATION_ABS,
            .flags = 4,
            .a = ZGML_MODULE_ACTIVATION_SQRT,
            .b = ZGML_MODULE_ACTIVATION_SQUARE,
            .c = ZGML_MODULE_ACTIVATION_SGN,
        },
        {
            .kind = ZGML_MODULE_OP_ACTIVATION,
            .activation = ZGML_MODULE_ACTIVATION_STEP,
        },
    };
    const zgml_module_desc module_desc = {
        .input_shape = module_input_shape,
        .input_rank = sizeof(module_input_shape) / sizeof(module_input_shape[0]),
        .ops = module_ops_abs_sqrt_square_sgn_step,
        .op_count = sizeof(module_ops_abs_sqrt_square_sgn_step) / sizeof(module_ops_abs_sqrt_square_sgn_step[0]),
    };
    const zgml_compile_desc module_compile_desc = {0};
    if (!failed) {
        failed |= expect_status("zgml_module_program_compile abs sqrt square sgn step", zgml_module_program_compile(&module_desc, &module_compile_desc, &module_program));
    }
    zgml_program_requirements module_requirements = {0};
    if (!failed) {
        failed |= expect_status("zgml_program_get_requirements module", zgml_program_get_requirements(module_program, &module_requirements));
    }
    if (!failed && (
        module_requirements.model_kind != ZGML_MODEL_MODULE ||
        module_requirements.input_len != 3 ||
        module_requirements.output_len != 3 ||
        module_requirements.weights_len != 0 ||
        module_requirements.bias_len != 0
    )) {
        fprintf(stderr, "unexpected module requirements: kind=%u input=%llu output=%llu weights=%llu bias=%llu\n",
            module_requirements.model_kind,
            (unsigned long long)module_requirements.input_len,
            (unsigned long long)module_requirements.output_len,
            (unsigned long long)module_requirements.weights_len,
            (unsigned long long)module_requirements.bias_len);
        failed = 1;
    }
    const zgml_bind_desc module_bind_desc = {
        .weights = NULL,
        .weights_len = 0,
        .bias = NULL,
        .bias_len = 0,
    };
    if (!failed) {
        failed |= expect_status("zgml_session_bind module", zgml_session_bind(module_program, &module_bind_desc, &module_session));
    }
    const float module_input[] = {4.0f, 0.0f, -16.0f};
    float module_output[] = {0.0f, 0.0f, 0.0f};
    zgml_step_result module_result = {0};
    const zgml_step_desc module_step = {
        .input = module_input,
        .input_len = sizeof(module_input) / sizeof(module_input[0]),
        .output = module_output,
        .output_len = sizeof(module_output) / sizeof(module_output[0]),
    };
    if (!failed) {
        failed |= expect_status("zgml_session_step module", zgml_session_step(module_session, &module_step, &module_result));
    }
    if (!failed && (
        module_result.output_len != 3 ||
        !expect_close(module_output[0], 1.0f) ||
        !expect_close(module_output[1], 0.0f) ||
        !expect_close(module_output[2], 1.0f)
    )) {
        fprintf(stderr, "unexpected C ABI module abs/sqrt/square/sgn/step output: len=%zu values=%f,%f,%f\n",
            module_result.output_len,
            module_output[0],
            module_output[1],
            module_output[2]);
        failed = 1;
    }
    zgml_session_free(module_session);
    zgml_program_free(module_program);

    zgml_program *webgpu_zero_program = NULL;
    zgml_session *webgpu_zero_session = NULL;
    const size_t webgpu_zero_input_shape[] = {4};
    const zgml_module_desc webgpu_zero_module_desc = {
        .input_shape = webgpu_zero_input_shape,
        .input_rank = sizeof(webgpu_zero_input_shape) / sizeof(webgpu_zero_input_shape[0]),
        .ops = NULL,
        .op_count = 0,
    };
    const zgml_compile_desc webgpu_zero_compile_desc = {
        .backend = ZGML_BACKEND_WEBGPU,
    };
    if (!failed) {
        failed |= expect_status("zgml_module_program_compile zero-op webgpu", zgml_module_program_compile(&webgpu_zero_module_desc, &webgpu_zero_compile_desc, &webgpu_zero_program));
    }
    zgml_program_requirements webgpu_zero_requirements = {0};
    if (!failed) {
        failed |= expect_status("zgml_program_get_requirements zero-op webgpu", zgml_program_get_requirements(webgpu_zero_program, &webgpu_zero_requirements));
    }
    if (!failed && (
        webgpu_zero_requirements.model_kind != ZGML_MODEL_MODULE ||
        webgpu_zero_requirements.input_len != 4 ||
        webgpu_zero_requirements.output_len != 4 ||
        webgpu_zero_requirements.weights_len != 0 ||
        webgpu_zero_requirements.bias_len != 0
    )) {
        fprintf(stderr, "unexpected zero-op WebGPU module requirements: kind=%u input=%llu output=%llu weights=%llu bias=%llu\n",
            webgpu_zero_requirements.model_kind,
            (unsigned long long)webgpu_zero_requirements.input_len,
            (unsigned long long)webgpu_zero_requirements.output_len,
            (unsigned long long)webgpu_zero_requirements.weights_len,
            (unsigned long long)webgpu_zero_requirements.bias_len);
        failed = 1;
    }
    zgml_program_inspection webgpu_zero_inspection = {0};
    if (!failed) {
        failed |= expect_status("zgml_program_inspect zero-op webgpu", zgml_program_inspect(webgpu_zero_program, &webgpu_zero_inspection));
    }
    const int webgpu_zero_executes = webgpu_zero_inspection.execution_supported != 0;
    if (!failed && (
        webgpu_zero_inspection.backend != ZGML_BACKEND_WEBGPU ||
        webgpu_zero_executes != runtime_reports_native_wgpu ||
        webgpu_zero_inspection.external_resources_supported != 1 ||
        webgpu_zero_inspection.buffer_count != 1 ||
        webgpu_zero_inspection.buffer_element_count != 4 ||
        webgpu_zero_inspection.buffer_byte_len != 16 ||
        webgpu_zero_inspection.initial_upload_count != 1 ||
        webgpu_zero_inspection.qweight_count != 0 ||
        webgpu_zero_inspection.op_count != 0 ||
        webgpu_zero_inspection.command_count != 0 ||
        webgpu_zero_inspection.command_stencil_hash != 0 ||
        command_category_total(webgpu_zero_inspection) != 0 ||
        webgpu_zero_inspection.backend_dispatch_count != 0 ||
        webgpu_zero_inspection.persistent_requirement_count != 0 ||
        webgpu_zero_inspection.step_input_requirement_count != 1 ||
        webgpu_zero_inspection.step_output_requirement_count != 1
    )) {
        fprintf(stderr, "unexpected zero-op WebGPU module inspection: backend=%llu executes=%llu resources=%llu buffers=%llu elems=%llu bytes=%llu uploads=%llu qweights=%llu ops=%llu commands=%llu hash=%llu dispatches=%llu persistent=%llu inputs=%llu outputs=%llu\n",
            (unsigned long long)webgpu_zero_inspection.backend,
            (unsigned long long)webgpu_zero_inspection.execution_supported,
            (unsigned long long)webgpu_zero_inspection.external_resources_supported,
            (unsigned long long)webgpu_zero_inspection.buffer_count,
            (unsigned long long)webgpu_zero_inspection.buffer_element_count,
            (unsigned long long)webgpu_zero_inspection.buffer_byte_len,
            (unsigned long long)webgpu_zero_inspection.initial_upload_count,
            (unsigned long long)webgpu_zero_inspection.qweight_count,
            (unsigned long long)webgpu_zero_inspection.op_count,
            (unsigned long long)webgpu_zero_inspection.command_count,
            (unsigned long long)webgpu_zero_inspection.command_stencil_hash,
            (unsigned long long)webgpu_zero_inspection.backend_dispatch_count,
            (unsigned long long)webgpu_zero_inspection.persistent_requirement_count,
            (unsigned long long)webgpu_zero_inspection.step_input_requirement_count,
            (unsigned long long)webgpu_zero_inspection.step_output_requirement_count);
        failed = 1;
    }
    if (!failed && webgpu_zero_executes && (
        webgpu_zero_inspection.dispatch_plan_supported != 1 ||
        webgpu_zero_inspection.dispatch_plan_covered_op_count != 0 ||
        webgpu_zero_inspection.dispatch_plan_first_unsupported_op != UINT64_MAX
    )) {
        fprintf(stderr, "unexpected executable zero-op WebGPU dispatch plan: supported=%llu covered=%llu first=%llu\n",
            (unsigned long long)webgpu_zero_inspection.dispatch_plan_supported,
            (unsigned long long)webgpu_zero_inspection.dispatch_plan_covered_op_count,
            (unsigned long long)webgpu_zero_inspection.dispatch_plan_first_unsupported_op);
        failed = 1;
    }
    const zgml_bind_desc webgpu_zero_bind_desc = {
        .weights = NULL,
        .weights_len = 0,
        .bias = NULL,
        .bias_len = 0,
    };
    if (!failed && webgpu_zero_executes) {
        failed |= expect_status("zgml_session_bind zero-op webgpu", zgml_session_bind(webgpu_zero_program, &webgpu_zero_bind_desc, &webgpu_zero_session));
        const float webgpu_zero_input[] = {8.0f, 6.0f, 4.0f, 2.0f};
        float webgpu_zero_output[] = {-1.0f, -1.0f, -1.0f, -1.0f};
        const zgml_step_desc webgpu_zero_step = {
            .input = webgpu_zero_input,
            .input_len = sizeof(webgpu_zero_input) / sizeof(webgpu_zero_input[0]),
            .output = webgpu_zero_output,
            .output_len = sizeof(webgpu_zero_output) / sizeof(webgpu_zero_output[0]),
        };
        zgml_step_result webgpu_zero_result = {0};
        if (!failed) {
            failed |= expect_status("zgml_session_step zero-op webgpu", zgml_session_step(webgpu_zero_session, &webgpu_zero_step, &webgpu_zero_result));
        }
        if (!failed && (
            webgpu_zero_result.output_len != 4 ||
            !expect_close(webgpu_zero_output[0], 8.0f) ||
            !expect_close(webgpu_zero_output[1], 6.0f) ||
            !expect_close(webgpu_zero_output[2], 4.0f) ||
            !expect_close(webgpu_zero_output[3], 2.0f)
        )) {
            fprintf(stderr, "unexpected zero-op WebGPU host output: len=%zu values=%f,%f,%f,%f\n",
                webgpu_zero_result.output_len,
                webgpu_zero_output[0],
                webgpu_zero_output[1],
                webgpu_zero_output[2],
                webgpu_zero_output[3]);
            failed = 1;
        }
        zgml_runtime_profile webgpu_zero_profile = {0};
        if (!failed) {
            failed |= expect_status("zgml_session_runtime_profile zero-op webgpu", zgml_session_runtime_profile(webgpu_zero_session, &webgpu_zero_profile));
        }
        if (!failed && (
            webgpu_zero_profile.call_count != 1 ||
            webgpu_zero_profile.command_count != 0 ||
            webgpu_zero_profile.backend_op_count != 0 ||
            webgpu_zero_profile.backend_dispatch_count != 0 ||
            webgpu_zero_profile.sync_count != 1
        )) {
            fprintf(stderr, "unexpected zero-op WebGPU host profile: calls=%llu commands=%llu ops=%llu dispatches=%llu sync=%llu\n",
                (unsigned long long)webgpu_zero_profile.call_count,
                (unsigned long long)webgpu_zero_profile.command_count,
                (unsigned long long)webgpu_zero_profile.backend_op_count,
                (unsigned long long)webgpu_zero_profile.backend_dispatch_count,
                (unsigned long long)webgpu_zero_profile.sync_count);
            failed = 1;
        }
    } else if (!failed) {
        webgpu_zero_session = (zgml_session *)(uintptr_t)1;
        const zgml_status zero_bind_status = zgml_session_bind(webgpu_zero_program, &webgpu_zero_bind_desc, &webgpu_zero_session);
        if (zero_bind_status != ZGML_UNSUPPORTED || webgpu_zero_session != NULL) {
            fprintf(stderr, "expected zero-op WebGPU host bind unsupported and cleared, got %s (%d) session=%p\n",
                zgml_status_name(zero_bind_status),
                zero_bind_status,
                (void *)webgpu_zero_session);
            failed = 1;
        }
    }
    zgml_session_free(webgpu_zero_session);
    webgpu_zero_session = NULL;

    zgml_buffer *webgpu_zero_device_input = NULL;
    zgml_buffer *webgpu_zero_device_output = NULL;
    if (!failed && webgpu_zero_executes) {
        failed |= expect_status("zgml_program_create_device_buffer zero-op input", zgml_program_create_device_buffer(webgpu_zero_program, ZGML_PROGRAM_BUFFER_INPUT, ZGML_BACKEND_WEBGPU, &webgpu_zero_device_input));
        failed |= expect_status("zgml_program_create_device_buffer zero-op output", zgml_program_create_device_buffer(webgpu_zero_program, ZGML_PROGRAM_BUFFER_OUTPUT, ZGML_BACKEND_WEBGPU, &webgpu_zero_device_output));
        const float webgpu_zero_device_input_data[] = {9.0f, 7.0f, 5.0f, 3.0f};
        failed |= expect_status("zgml_buffer_write zero-op webgpu input", zgml_buffer_write(webgpu_zero_device_input, 0, webgpu_zero_device_input_data, sizeof(webgpu_zero_device_input_data)));
        const zgml_buffer_bind_desc webgpu_zero_device_bind_desc = {
            .input = webgpu_zero_device_input,
            .input_len = sizeof(webgpu_zero_device_input_data) / sizeof(webgpu_zero_device_input_data[0]),
            .output = webgpu_zero_device_output,
            .output_len = sizeof(webgpu_zero_device_input_data) / sizeof(webgpu_zero_device_input_data[0]),
        };
        failed |= expect_status("zgml_session_bind_buffers zero-op webgpu", zgml_session_bind_buffers(webgpu_zero_program, &webgpu_zero_device_bind_desc, &webgpu_zero_session));
    }
    if (!failed && webgpu_zero_executes) {
        zgml_step_result webgpu_zero_device_result = {0};
        float webgpu_zero_device_output_data[] = {-1.0f, -1.0f, -1.0f, -1.0f};
        failed |= expect_status("zgml_session_step zero-op webgpu device", zgml_session_step(webgpu_zero_session, NULL, &webgpu_zero_device_result));
        failed |= expect_status("zgml_buffer_read zero-op webgpu output", zgml_buffer_read(webgpu_zero_device_output, 0, webgpu_zero_device_output_data, sizeof(webgpu_zero_device_output_data)));
        zgml_runtime_profile webgpu_zero_device_profile = {0};
        failed |= expect_status("zgml_session_runtime_profile zero-op webgpu device", zgml_session_runtime_profile(webgpu_zero_session, &webgpu_zero_device_profile));
        if (!failed && (
            webgpu_zero_device_result.output_len != 4 ||
            !expect_close(webgpu_zero_device_output_data[0], 9.0f) ||
            !expect_close(webgpu_zero_device_output_data[1], 7.0f) ||
            !expect_close(webgpu_zero_device_output_data[2], 5.0f) ||
            !expect_close(webgpu_zero_device_output_data[3], 3.0f) ||
            webgpu_zero_device_profile.call_count != 1 ||
            webgpu_zero_device_profile.command_count != 0 ||
            webgpu_zero_device_profile.backend_op_count != 0 ||
            webgpu_zero_device_profile.backend_dispatch_count != 0 ||
            webgpu_zero_device_profile.sync_count != 0
        )) {
            fprintf(stderr, "unexpected zero-op WebGPU device result: len=%zu values=%f,%f,%f,%f calls=%llu commands=%llu ops=%llu dispatches=%llu sync=%llu\n",
                webgpu_zero_device_result.output_len,
                webgpu_zero_device_output_data[0],
                webgpu_zero_device_output_data[1],
                webgpu_zero_device_output_data[2],
                webgpu_zero_device_output_data[3],
                (unsigned long long)webgpu_zero_device_profile.call_count,
                (unsigned long long)webgpu_zero_device_profile.command_count,
                (unsigned long long)webgpu_zero_device_profile.backend_op_count,
                (unsigned long long)webgpu_zero_device_profile.backend_dispatch_count,
                (unsigned long long)webgpu_zero_device_profile.sync_count);
            failed = 1;
        }
    }
    zgml_session_free(webgpu_zero_session);
    zgml_buffer_free(webgpu_zero_device_output);
    zgml_buffer_free(webgpu_zero_device_input);
    zgml_program_free(webgpu_zero_program);

    const zgml_model_desc model_desc = {
        .kind = ZGML_MODEL_TINY_LINEAR,
        .input_len = 2,
        .output_len = 3,
    };
    failed |= expect_status("zgml_model_create", zgml_model_create(&model_desc, &model));
    zgml_model_inspection model_inspection = {0};
    if (!failed) {
        failed |= expect_status("zgml_model_inspect", zgml_model_inspect(model, &model_inspection));
    }
    if (!failed && (
        model_inspection.model_kind != ZGML_MODEL_TINY_LINEAR ||
        model_inspection.input_len != 2 ||
        model_inspection.output_len != 3 ||
        model_inspection.vocab_size != 0
    )) {
        fprintf(stderr, "unexpected model inspection: kind=%u input=%llu output=%llu vocab=%llu\n",
            model_inspection.model_kind,
            (unsigned long long)model_inspection.input_len,
            (unsigned long long)model_inspection.output_len,
            (unsigned long long)model_inspection.vocab_size);
        failed = 1;
    }

    const zgml_compile_desc compile_desc = {0};
    if (!failed) {
        failed |= expect_status("zgml_program_compile", zgml_program_compile(model, &compile_desc, &program));
    }
    zgml_program_model_compatibility compatibility = {0};
    if (!failed) {
        failed |= expect_status("zgml_program_check_model_compatibility", zgml_program_check_model_compatibility(program, model, &compatibility));
    }
    if (!failed && (
        compatibility.program_model_kind != ZGML_MODEL_TINY_LINEAR ||
        compatibility.model_kind != ZGML_MODEL_TINY_LINEAR ||
        compatibility.compatible != 0
    )) {
        fprintf(stderr, "unexpected model compatibility: program=%u model=%u compatible=%llu\n",
            compatibility.program_model_kind,
            compatibility.model_kind,
            (unsigned long long)compatibility.compatible);
        failed = 1;
    }

    zgml_program_inspection inspection = {0};
    if (!failed) {
        failed |= expect_status("zgml_program_inspect", zgml_program_inspect(program, &inspection));
    }
    if (!failed && inspection.command_count == 0) {
        fprintf(stderr, "expected non-empty compiled command shape\n");
        failed = 1;
    }
    if (!failed && inspection.command_stencil_hash == 0) {
        fprintf(stderr, "expected non-zero command stencil hash\n");
        failed = 1;
    }
    if (!failed && inspection.binding_requirement_hash == 0) {
        fprintf(stderr, "expected non-zero binding requirement hash\n");
        failed = 1;
    }
    if (!failed && (inspection.persistent_requirement_count != 2 ||
        inspection.step_input_requirement_count != 1 ||
        inspection.step_output_requirement_count != 1)) {
        fprintf(stderr, "unexpected binding requirement counts: persistent=%llu input=%llu output=%llu\n",
            (unsigned long long)inspection.persistent_requirement_count,
            (unsigned long long)inspection.step_input_requirement_count,
            (unsigned long long)inspection.step_output_requirement_count);
        failed = 1;
    }
    if (!failed && command_category_total(inspection) == 0) {
        fprintf(stderr, "expected non-empty command category counts\n");
        failed = 1;
    }
    if (!failed && inspection.external_resources_supported != 0) {
        fprintf(stderr, "expected host-only tiny-linear program resource support to be false\n");
        failed = 1;
    }
    if (!failed && (
        inspection.buffer_count != 5 ||
        inspection.buffer_element_count != 17 ||
        inspection.buffer_byte_len != 68 ||
        inspection.initial_upload_count != 1 ||
        inspection.qweight_count != 0 ||
        inspection.op_count == 0 ||
        inspection.runtime_patch_max_cache_write_pos != UINT32_MAX ||
        inspection.runtime_patch_max_attention_seq_kv != UINT32_MAX
    )) {
        fprintf(stderr, "unexpected program memory layout: buffers=%llu elements=%llu bytes=%llu uploads=%llu qweights=%llu ops=%llu max_cache=%llu max_attention=%llu\n",
            (unsigned long long)inspection.buffer_count,
            (unsigned long long)inspection.buffer_element_count,
            (unsigned long long)inspection.buffer_byte_len,
            (unsigned long long)inspection.initial_upload_count,
            (unsigned long long)inspection.qweight_count,
            (unsigned long long)inspection.op_count,
            (unsigned long long)inspection.runtime_patch_max_cache_write_pos,
            (unsigned long long)inspection.runtime_patch_max_attention_seq_kv);
        failed = 1;
    }

    zgml_program *webgpu_program = NULL;
    const zgml_compile_desc webgpu_compile_desc = {
        .backend = ZGML_BACKEND_WEBGPU,
    };
    if (!failed) {
        failed |= expect_status("zgml_program_compile webgpu", zgml_program_compile(model, &webgpu_compile_desc, &webgpu_program));
    }
    zgml_program_inspection webgpu_inspection = {0};
    if (!failed) {
        failed |= expect_status("zgml_program_inspect webgpu", zgml_program_inspect(webgpu_program, &webgpu_inspection));
    }
    if (!failed && webgpu_inspection.backend != ZGML_BACKEND_WEBGPU) {
        fprintf(stderr, "expected WebGPU backend inspection, got %llu\n", (unsigned long long)webgpu_inspection.backend);
        failed = 1;
    }
    const int webgpu_executes = webgpu_inspection.execution_supported != 0;
    if (!failed && webgpu_inspection.execution_supported > 1) {
        fprintf(stderr, "unexpected WebGPU tiny-linear execution flag: %llu\n", (unsigned long long)webgpu_inspection.execution_supported);
        failed = 1;
    }
    if (!failed && webgpu_inspection.external_resources_supported != 1) {
        fprintf(stderr, "expected WebGPU tiny-linear resource-table support, got %llu\n", (unsigned long long)webgpu_inspection.external_resources_supported);
        failed = 1;
    }
    if (!failed && webgpu_executes != runtime_reports_native_wgpu) {
        fprintf(stderr, "native wgpu runtime feature disagrees with tiny-linear WebGPU execution: feature=%d execution=%d\n",
            runtime_reports_native_wgpu,
            webgpu_executes);
        failed = 1;
    }
    if (!failed && (
        webgpu_inspection.buffer_count != 5 ||
        webgpu_inspection.buffer_element_count != 17 ||
        webgpu_inspection.buffer_byte_len != 68 ||
        webgpu_inspection.initial_upload_count != 1 ||
        webgpu_inspection.qweight_count != 0 ||
        webgpu_inspection.op_count == 0 ||
        webgpu_inspection.runtime_patch_max_cache_write_pos != UINT32_MAX ||
        webgpu_inspection.runtime_patch_max_attention_seq_kv != UINT32_MAX
    )) {
        fprintf(stderr, "unexpected WebGPU program memory layout: buffers=%llu elements=%llu bytes=%llu uploads=%llu qweights=%llu ops=%llu max_cache=%llu max_attention=%llu\n",
            (unsigned long long)webgpu_inspection.buffer_count,
            (unsigned long long)webgpu_inspection.buffer_element_count,
            (unsigned long long)webgpu_inspection.buffer_byte_len,
            (unsigned long long)webgpu_inspection.initial_upload_count,
            (unsigned long long)webgpu_inspection.qweight_count,
            (unsigned long long)webgpu_inspection.op_count,
            (unsigned long long)webgpu_inspection.runtime_patch_max_cache_write_pos,
            (unsigned long long)webgpu_inspection.runtime_patch_max_attention_seq_kv);
        failed = 1;
    }
    if (!failed && (webgpu_inspection.command_count == 0 || webgpu_inspection.command_stencil_hash == 0 || command_category_total(webgpu_inspection) == 0)) {
        fprintf(stderr, "expected WebGPU tiny-linear command evidence\n");
        failed = 1;
    }
    if (!failed && webgpu_executes) {
        if (webgpu_inspection.backend_dispatch_count == 0 ||
            webgpu_inspection.dispatch_plan_supported != 1 ||
            webgpu_inspection.dispatch_plan_covered_op_count != webgpu_inspection.op_count ||
            webgpu_inspection.dispatch_plan_first_unsupported_op != UINT64_MAX ||
            webgpu_inspection.dispatch_plan_projection_count == 0) {
            fprintf(stderr, "unexpected executable WebGPU tiny-linear dispatch plan: dispatches=%llu supported=%llu covered=%llu ops=%llu first=%llu projections=%llu\n",
                (unsigned long long)webgpu_inspection.backend_dispatch_count,
                (unsigned long long)webgpu_inspection.dispatch_plan_supported,
                (unsigned long long)webgpu_inspection.dispatch_plan_covered_op_count,
                (unsigned long long)webgpu_inspection.op_count,
                (unsigned long long)webgpu_inspection.dispatch_plan_first_unsupported_op,
                (unsigned long long)webgpu_inspection.dispatch_plan_projection_count);
            failed = 1;
        }
    }
    if (!failed && !webgpu_executes) {
        if (webgpu_inspection.backend_dispatch_count != 0 ||
            webgpu_inspection.dispatch_plan_supported != 0 ||
            webgpu_inspection.dispatch_plan_covered_op_count != 0 ||
            webgpu_inspection.dispatch_plan_first_unsupported_op != UINT64_MAX ||
            webgpu_inspection.dispatch_plan_projection_count != 0) {
            fprintf(stderr, "unexpected compile-only WebGPU tiny-linear dispatch plan: dispatches=%llu supported=%llu covered=%llu first=%llu projections=%llu\n",
                (unsigned long long)webgpu_inspection.backend_dispatch_count,
                (unsigned long long)webgpu_inspection.dispatch_plan_supported,
                (unsigned long long)webgpu_inspection.dispatch_plan_covered_op_count,
                (unsigned long long)webgpu_inspection.dispatch_plan_first_unsupported_op,
                (unsigned long long)webgpu_inspection.dispatch_plan_projection_count);
            failed = 1;
        }
    }
    const float webgpu_weights[] = {1, 0, 0, 1, 1, 1};
    const zgml_bind_desc webgpu_bind_desc = {
        .weights = webgpu_weights,
        .weights_len = sizeof(webgpu_weights) / sizeof(webgpu_weights[0]),
    };
    zgml_session *webgpu_session = NULL;
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_session_bind executable webgpu", zgml_session_bind(webgpu_program, &webgpu_bind_desc, &webgpu_session));
        const float webgpu_input[] = {2, 3};
        float webgpu_output[] = {-9, -9, -9};
        const zgml_step_desc webgpu_step_desc = {
            .input = webgpu_input,
            .input_len = sizeof(webgpu_input) / sizeof(webgpu_input[0]),
            .output = webgpu_output,
            .output_len = sizeof(webgpu_output) / sizeof(webgpu_output[0]),
        };
        zgml_step_result webgpu_step_result = {0};
        if (!failed) {
            failed |= expect_status("zgml_session_step executable webgpu", zgml_session_step(webgpu_session, &webgpu_step_desc, &webgpu_step_result));
        }
        if (!failed && (webgpu_step_result.output_len != 3 || !expect_close(webgpu_output[0], 5.0f) || !expect_close(webgpu_output[1], 3.0f) || !expect_close(webgpu_output[2], 3.0f))) {
            fprintf(stderr, "unexpected executable WebGPU tiny-linear output: len=%zu values=%f,%f,%f\n",
                webgpu_step_result.output_len,
                webgpu_output[0],
                webgpu_output[1],
                webgpu_output[2]);
            failed = 1;
        }
    } else if (!failed) {
        webgpu_session = (zgml_session *)(uintptr_t)1;
        const zgml_status bind_status = zgml_session_bind(webgpu_program, &webgpu_bind_desc, &webgpu_session);
        if (bind_status != ZGML_UNSUPPORTED) {
            fprintf(stderr, "expected WebGPU tiny-linear bind unsupported, got %s (%d)\n", zgml_status_name(bind_status), bind_status);
            failed = 1;
        }
    }
    if (!failed && !webgpu_executes && webgpu_session != NULL) {
        fprintf(stderr, "expected WebGPU tiny-linear bind to clear failed session output\n");
        failed = 1;
    }
    zgml_session_free(webgpu_session);
    webgpu_session = NULL;

    zgml_buffer *webgpu_device_weights = NULL;
    zgml_buffer *webgpu_device_bias = NULL;
    zgml_buffer *webgpu_device_input = NULL;
    zgml_buffer *webgpu_device_output = NULL;
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_program_create_device_buffer weights", zgml_program_create_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_WEIGHTS, ZGML_BACKEND_WEBGPU, &webgpu_device_weights));
        failed |= expect_status("zgml_program_create_device_buffer bias", zgml_program_create_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_BIAS, ZGML_BACKEND_WEBGPU, &webgpu_device_bias));
        failed |= expect_status("zgml_program_create_device_buffer input", zgml_program_create_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_INPUT, ZGML_BACKEND_WEBGPU, &webgpu_device_input));
        failed |= expect_status("zgml_program_create_device_buffer output", zgml_program_create_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_OUTPUT, ZGML_BACKEND_WEBGPU, &webgpu_device_output));
    }
    const float webgpu_device_weights_data[] = {1, 0, 0, 1, 1, 1};
    const float webgpu_device_bias_data[] = {10, -1, 2};
    const float webgpu_device_input_data[] = {2, 3};
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_buffer_write webgpu device weights", zgml_buffer_write(webgpu_device_weights, 0, webgpu_device_weights_data, sizeof(webgpu_device_weights_data)));
        failed |= expect_status("zgml_buffer_write webgpu device bias", zgml_buffer_write(webgpu_device_bias, 0, webgpu_device_bias_data, sizeof(webgpu_device_bias_data)));
        failed |= expect_status("zgml_buffer_write webgpu device input", zgml_buffer_write(webgpu_device_input, 0, webgpu_device_input_data, sizeof(webgpu_device_input_data)));
    }
    zgml_buffer_inspection webgpu_device_output_inspection = {0};
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_buffer_inspect webgpu device output", zgml_buffer_inspect(webgpu_device_output, &webgpu_device_output_inspection));
    }
    if (!failed && webgpu_executes && (
        webgpu_device_output_inspection.storage != ZGML_BUFFER_STORAGE_EXTERNAL_RESOURCE ||
        webgpu_device_output_inspection.placement != ZGML_BACKEND_WEBGPU ||
        webgpu_device_output_inspection.access_flags != ZGML_RESOURCE_ACCESS_WRITE ||
        webgpu_device_output_inspection.byte_len != sizeof(webgpu_device_bias_data) ||
        webgpu_device_output_inspection.handle == 0
    )) {
        fprintf(stderr, "unexpected WebGPU device output buffer inspection: storage=%u placement=%u access=%u bytes=%llu handle=%llu\n",
            webgpu_device_output_inspection.storage,
            webgpu_device_output_inspection.placement,
            webgpu_device_output_inspection.access_flags,
            (unsigned long long)webgpu_device_output_inspection.byte_len,
            (unsigned long long)webgpu_device_output_inspection.handle);
        failed = 1;
    }
    const zgml_buffer_bind_desc webgpu_device_bind_desc = {
        .weights = webgpu_device_weights,
        .weights_len = sizeof(webgpu_device_weights_data) / sizeof(webgpu_device_weights_data[0]),
        .bias = webgpu_device_bias,
        .bias_len = sizeof(webgpu_device_bias_data) / sizeof(webgpu_device_bias_data[0]),
        .input = webgpu_device_input,
        .input_len = sizeof(webgpu_device_input_data) / sizeof(webgpu_device_input_data[0]),
        .output = webgpu_device_output,
        .output_len = sizeof(webgpu_device_bias_data) / sizeof(webgpu_device_bias_data[0]),
    };
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_session_bind_buffers webgpu device", zgml_session_bind_buffers(webgpu_program, &webgpu_device_bind_desc, &webgpu_session));
    }
    zgml_session_inspection webgpu_device_session_inspection = {0};
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_session_inspect webgpu device", zgml_session_inspect(webgpu_session, &webgpu_device_session_inspection));
    }
    if (!failed && webgpu_executes && (
        webgpu_device_session_inspection.backend != ZGML_BACKEND_WEBGPU ||
        webgpu_device_session_inspection.output_storage != ZGML_BUFFER_STORAGE_EXTERNAL_RESOURCE ||
        webgpu_device_session_inspection.persistent_binding_count != 2 ||
        webgpu_device_session_inspection.step_input_count != 1 ||
        webgpu_device_session_inspection.step_output_count != 1 ||
        webgpu_device_session_inspection.host_binding_count != 0 ||
        webgpu_device_session_inspection.resource_binding_count != 4 ||
        webgpu_device_session_inspection.binding_shape_hash == 0
    )) {
        fprintf(stderr, "unexpected WebGPU device session inspection: backend=%u output_storage=%u persistent=%llu input=%llu output=%llu host=%llu resource=%llu binding_hash=%llu\n",
            webgpu_device_session_inspection.backend,
            webgpu_device_session_inspection.output_storage,
            (unsigned long long)webgpu_device_session_inspection.persistent_binding_count,
            (unsigned long long)webgpu_device_session_inspection.step_input_count,
            (unsigned long long)webgpu_device_session_inspection.step_output_count,
            (unsigned long long)webgpu_device_session_inspection.host_binding_count,
            (unsigned long long)webgpu_device_session_inspection.resource_binding_count,
            (unsigned long long)webgpu_device_session_inspection.binding_shape_hash);
        failed = 1;
    }
    zgml_step_result webgpu_device_step_result = {0};
    float webgpu_device_output_data[] = {-9, -9, -9};
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_session_step webgpu device", zgml_session_step(webgpu_session, NULL, &webgpu_device_step_result));
        failed |= expect_status("zgml_buffer_read webgpu device output", zgml_buffer_read(webgpu_device_output, 0, webgpu_device_output_data, sizeof(webgpu_device_output_data)));
    }
    if (!failed && webgpu_executes && (
        webgpu_device_step_result.output_len != 3 ||
        !expect_close(webgpu_device_output_data[0], 15.0f) ||
        !expect_close(webgpu_device_output_data[1], 2.0f) ||
        !expect_close(webgpu_device_output_data[2], 5.0f)
    )) {
        fprintf(stderr, "unexpected WebGPU device-buffer tiny-linear output: len=%zu values=%f,%f,%f\n",
            webgpu_device_step_result.output_len,
            webgpu_device_output_data[0],
            webgpu_device_output_data[1],
            webgpu_device_output_data[2]);
        failed = 1;
    }
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_session_reset_runtime_profile webgpu device", zgml_session_reset_runtime_profile(webgpu_session));
    }
    const float webgpu_device_input_data_2[] = {6, 7};
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_buffer_write webgpu device input 2", zgml_buffer_write(webgpu_device_input, 0, webgpu_device_input_data_2, sizeof(webgpu_device_input_data_2)));
        failed |= expect_status("zgml_session_step_no_output webgpu device", zgml_session_step_no_output(webgpu_session, NULL, &webgpu_device_step_result));
    }
    zgml_runtime_profile webgpu_device_profile = {0};
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_session_runtime_profile webgpu device", zgml_session_runtime_profile(webgpu_session, &webgpu_device_profile));
    }
    if (!failed && webgpu_executes && (webgpu_device_profile.call_count != 1 || webgpu_device_profile.sync_count != 0)) {
        fprintf(stderr, "expected no-readback WebGPU device profile call=1 sync=0, got call=%llu sync=%llu\n",
            (unsigned long long)webgpu_device_profile.call_count,
            (unsigned long long)webgpu_device_profile.sync_count);
        failed = 1;
    }
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_buffer_read webgpu device output after no-output step", zgml_buffer_read(webgpu_device_output, 0, webgpu_device_output_data, sizeof(webgpu_device_output_data)));
    }
    if (!failed && webgpu_executes && (
        !expect_close(webgpu_device_output_data[0], 23.0f) ||
        !expect_close(webgpu_device_output_data[1], 6.0f) ||
        !expect_close(webgpu_device_output_data[2], 9.0f)
    )) {
        fprintf(stderr, "unexpected WebGPU device-buffer no-output advance contents: values=%f,%f,%f\n",
            webgpu_device_output_data[0],
            webgpu_device_output_data[1],
            webgpu_device_output_data[2]);
        failed = 1;
    }

    uintptr_t webgpu_device_handle = 0;
    zgml_buffer_inspection webgpu_device_weights_inspection = {0};
    zgml_buffer_inspection webgpu_device_bias_inspection = {0};
    zgml_buffer_inspection webgpu_device_input_inspection = {0};
    if (!failed && webgpu_executes) {
        failed |= expect_status("zgml_program_get_device_handle webgpu", zgml_program_get_device_handle(webgpu_program, ZGML_BACKEND_WEBGPU, &webgpu_device_handle));
        failed |= expect_status("zgml_buffer_inspect webgpu device weights", zgml_buffer_inspect(webgpu_device_weights, &webgpu_device_weights_inspection));
        failed |= expect_status("zgml_buffer_inspect webgpu device bias", zgml_buffer_inspect(webgpu_device_bias, &webgpu_device_bias_inspection));
        failed |= expect_status("zgml_buffer_inspect webgpu device input", zgml_buffer_inspect(webgpu_device_input, &webgpu_device_input_inspection));
    }
    if (!failed && webgpu_executes && webgpu_device_handle == 0) {
        fprintf(stderr, "expected nonzero WebGPU device handle\n");
        failed = 1;
    }
    zgml_buffer *bad_imported_webgpu = (zgml_buffer *)(uintptr_t)1;
    if (!failed && webgpu_executes) {
        const zgml_device_buffer_import_desc bad_import_desc = {
            .placement = ZGML_BACKEND_WEBGPU,
            .device_handle = webgpu_device_handle + 1,
            .buffer_handle = (uintptr_t)webgpu_device_weights_inspection.handle,
            .byte_offset = 0,
            .byte_len = (size_t)webgpu_device_weights_inspection.byte_len,
        };
        const zgml_status bad_import_status = zgml_program_import_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_WEIGHTS, &bad_import_desc, &bad_imported_webgpu);
        if (bad_import_status != ZGML_UNSUPPORTED || bad_imported_webgpu != NULL) {
            fprintf(stderr, "expected wrong-device WebGPU import to be unsupported and clear output, got %s (%d) buffer=%p\n",
                zgml_status_name(bad_import_status),
                bad_import_status,
                (void *)bad_imported_webgpu);
            failed = 1;
        }
    }

    zgml_buffer *imported_webgpu_weights = NULL;
    zgml_buffer *imported_webgpu_bias = NULL;
    zgml_buffer *imported_webgpu_input = NULL;
    zgml_buffer *imported_webgpu_output = NULL;
    if (!failed && webgpu_executes) {
        const zgml_device_buffer_import_desc imported_weights_desc = {
            .placement = ZGML_BACKEND_WEBGPU,
            .device_handle = webgpu_device_handle,
            .buffer_handle = (uintptr_t)webgpu_device_weights_inspection.handle,
            .byte_offset = 0,
            .byte_len = (size_t)webgpu_device_weights_inspection.byte_len,
        };
        const zgml_device_buffer_import_desc imported_bias_desc = {
            .placement = ZGML_BACKEND_WEBGPU,
            .device_handle = webgpu_device_handle,
            .buffer_handle = (uintptr_t)webgpu_device_bias_inspection.handle,
            .byte_offset = 0,
            .byte_len = (size_t)webgpu_device_bias_inspection.byte_len,
        };
        const zgml_device_buffer_import_desc imported_input_desc = {
            .placement = ZGML_BACKEND_WEBGPU,
            .device_handle = webgpu_device_handle,
            .buffer_handle = (uintptr_t)webgpu_device_input_inspection.handle,
            .byte_offset = 0,
            .byte_len = (size_t)webgpu_device_input_inspection.byte_len,
        };
        const zgml_device_buffer_import_desc imported_output_desc = {
            .placement = ZGML_BACKEND_WEBGPU,
            .device_handle = webgpu_device_handle,
            .buffer_handle = (uintptr_t)webgpu_device_output_inspection.handle,
            .byte_offset = 0,
            .byte_len = (size_t)webgpu_device_output_inspection.byte_len,
        };
        failed |= expect_status("zgml_program_import_device_buffer weights", zgml_program_import_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_WEIGHTS, &imported_weights_desc, &imported_webgpu_weights));
        failed |= expect_status("zgml_program_import_device_buffer bias", zgml_program_import_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_BIAS, &imported_bias_desc, &imported_webgpu_bias));
        failed |= expect_status("zgml_program_import_device_buffer input", zgml_program_import_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_INPUT, &imported_input_desc, &imported_webgpu_input));
        failed |= expect_status("zgml_program_import_device_buffer output", zgml_program_import_device_buffer(webgpu_program, ZGML_PROGRAM_BUFFER_OUTPUT, &imported_output_desc, &imported_webgpu_output));
    }
    zgml_session_free(webgpu_session);
    webgpu_session = NULL;
    if (!failed && webgpu_executes) {
        zgml_buffer_free(webgpu_device_output);
        zgml_buffer_free(webgpu_device_input);
        zgml_buffer_free(webgpu_device_bias);
        zgml_buffer_free(webgpu_device_weights);
        webgpu_device_output = NULL;
        webgpu_device_input = NULL;
        webgpu_device_bias = NULL;
        webgpu_device_weights = NULL;

        const float imported_weights_data[] = {2, 0, 1, 0, 2, 1};
        const float imported_bias_data[] = {0, 5, -1};
        const float imported_input_data[] = {1, 4};
        failed |= expect_status("zgml_buffer_write imported webgpu weights", zgml_buffer_write(imported_webgpu_weights, 0, imported_weights_data, sizeof(imported_weights_data)));
        failed |= expect_status("zgml_buffer_write imported webgpu bias", zgml_buffer_write(imported_webgpu_bias, 0, imported_bias_data, sizeof(imported_bias_data)));
        failed |= expect_status("zgml_buffer_write imported webgpu input", zgml_buffer_write(imported_webgpu_input, 0, imported_input_data, sizeof(imported_input_data)));

        const zgml_buffer_bind_desc imported_bind_desc = {
            .weights = imported_webgpu_weights,
            .weights_len = sizeof(imported_weights_data) / sizeof(imported_weights_data[0]),
            .bias = imported_webgpu_bias,
            .bias_len = sizeof(imported_bias_data) / sizeof(imported_bias_data[0]),
            .input = imported_webgpu_input,
            .input_len = sizeof(imported_input_data) / sizeof(imported_input_data[0]),
            .output = imported_webgpu_output,
            .output_len = sizeof(imported_bias_data) / sizeof(imported_bias_data[0]),
        };
        failed |= expect_status("zgml_session_bind_buffers imported webgpu", zgml_session_bind_buffers(webgpu_program, &imported_bind_desc, &webgpu_session));
    }
    if (!failed && webgpu_executes) {
        zgml_session_inspection imported_session_inspection = {0};
        failed |= expect_status("zgml_session_inspect imported webgpu", zgml_session_inspect(webgpu_session, &imported_session_inspection));
        if (!failed && (
            imported_session_inspection.output_storage != ZGML_BUFFER_STORAGE_EXTERNAL_RESOURCE ||
            imported_session_inspection.host_binding_count != 0 ||
            imported_session_inspection.resource_binding_count != 4 ||
            imported_session_inspection.binding_shape_hash == 0
        )) {
            fprintf(stderr, "unexpected imported WebGPU session inspection: output_storage=%u host=%llu resource=%llu binding_hash=%llu\n",
                imported_session_inspection.output_storage,
                (unsigned long long)imported_session_inspection.host_binding_count,
                (unsigned long long)imported_session_inspection.resource_binding_count,
                (unsigned long long)imported_session_inspection.binding_shape_hash);
            failed = 1;
        }
    }
    if (!failed && webgpu_executes) {
        float imported_output_data[] = {-9, -9, -9};
        zgml_runtime_profile imported_profile = {0};
        failed |= expect_status("zgml_session_step imported webgpu", zgml_session_step(webgpu_session, NULL, &webgpu_device_step_result));
        failed |= expect_status("zgml_buffer_read imported webgpu output", zgml_buffer_read(imported_webgpu_output, 0, imported_output_data, sizeof(imported_output_data)));
        failed |= expect_status("zgml_session_runtime_profile imported webgpu", zgml_session_runtime_profile(webgpu_session, &imported_profile));
        if (!failed && (
            webgpu_device_step_result.output_len != 3 ||
            !expect_close(imported_output_data[0], 2.0f) ||
            !expect_close(imported_output_data[1], 13.0f) ||
            !expect_close(imported_output_data[2], 4.0f) ||
            imported_profile.call_count != 1 ||
            imported_profile.sync_count != 0
        )) {
            fprintf(stderr, "unexpected imported WebGPU result: len=%zu values=%f,%f,%f calls=%llu sync=%llu\n",
                webgpu_device_step_result.output_len,
                imported_output_data[0],
                imported_output_data[1],
                imported_output_data[2],
                (unsigned long long)imported_profile.call_count,
                (unsigned long long)imported_profile.sync_count);
            failed = 1;
        }
    }
    zgml_session_free(webgpu_session);
    webgpu_session = NULL;
    zgml_buffer_free(imported_webgpu_output);
    zgml_buffer_free(imported_webgpu_input);
    zgml_buffer_free(imported_webgpu_bias);
    zgml_buffer_free(imported_webgpu_weights);
    zgml_buffer_free(webgpu_device_output);
    zgml_buffer_free(webgpu_device_input);
    zgml_buffer_free(webgpu_device_bias);
    zgml_buffer_free(webgpu_device_weights);
    zgml_program_free(webgpu_program);

    const float weights[] = {1, 2, 3, 4, 5, 6};
    const float bias[] = {0.5f, -0.5f, 1.0f};
    const float input[] = {1, 2};
    const float zero_output[] = {0, 0, 0};
    if (!failed) {
        failed |= expect_status("zgml_program_create_buffer weights", zgml_program_create_buffer(program, ZGML_PROGRAM_BUFFER_WEIGHTS, &weights_buffer));
        failed |= expect_status("zgml_program_create_buffer bias", zgml_program_create_buffer(program, ZGML_PROGRAM_BUFFER_BIAS, &bias_buffer));
        failed |= expect_status("zgml_program_create_buffer input", zgml_program_create_buffer(program, ZGML_PROGRAM_BUFFER_INPUT, &input_buffer));
        failed |= expect_status("zgml_program_create_buffer output", zgml_program_create_buffer(program, ZGML_PROGRAM_BUFFER_OUTPUT, &output_buffer));
    }
    if (!failed && zgml_buffer_size(output_buffer) != sizeof(zero_output)) {
        fprintf(stderr, "unexpected output buffer size\n");
        failed = 1;
    }
    zgml_buffer_inspection output_buffer_inspection = {0};
    if (!failed) {
        failed |= expect_status("zgml_buffer_inspect output", zgml_buffer_inspect(output_buffer, &output_buffer_inspection));
    }
    if (!failed && (
        output_buffer_inspection.storage != ZGML_BUFFER_STORAGE_HOST ||
        output_buffer_inspection.byte_len != sizeof(zero_output) ||
        output_buffer_inspection.placement != 0 ||
        output_buffer_inspection.handle != 0
    )) {
        fprintf(stderr, "unexpected output buffer inspection: storage=%u bytes=%llu placement=%u handle=%llu\n",
            output_buffer_inspection.storage,
            (unsigned long long)output_buffer_inspection.byte_len,
            output_buffer_inspection.placement,
            (unsigned long long)output_buffer_inspection.handle);
        failed = 1;
    }
    if (!failed) {
        failed |= expect_status("zgml_buffer_write weights", zgml_buffer_write(weights_buffer, 0, weights, sizeof(weights)));
        failed |= expect_status("zgml_buffer_write bias", zgml_buffer_write(bias_buffer, 0, bias, sizeof(bias)));
        failed |= expect_status("zgml_buffer_write input", zgml_buffer_write(input_buffer, 0, input, sizeof(input)));
        failed |= expect_status("zgml_buffer_write output", zgml_buffer_write(output_buffer, 0, zero_output, sizeof(zero_output)));
    }

    const zgml_buffer_bind_desc bind_desc = {
        .weights = weights_buffer,
        .weights_len = sizeof(weights) / sizeof(weights[0]),
        .bias = bias_buffer,
        .bias_len = sizeof(bias) / sizeof(bias[0]),
        .input = input_buffer,
        .input_len = sizeof(input) / sizeof(input[0]),
        .output = output_buffer,
        .output_len = 3,
    };
    if (!failed) {
        failed |= expect_status("zgml_session_bind_buffers", zgml_session_bind_buffers(program, &bind_desc, &session));
    }
    zgml_session_inspection bound_session_inspection = {0};
    if (!failed) {
        failed |= expect_status("zgml_session_inspect", zgml_session_inspect(session, &bound_session_inspection));
    }
    if (!failed && (
        bound_session_inspection.model_kind != ZGML_MODEL_TINY_LINEAR ||
        bound_session_inspection.backend != ZGML_BACKEND_CPU ||
        bound_session_inspection.output_storage != ZGML_BUFFER_STORAGE_HOST ||
        bound_session_inspection.kv_cache_storage != ZGML_BUFFER_STORAGE_NONE ||
        bound_session_inspection.persistent_binding_count != 2 ||
        bound_session_inspection.step_input_count != 1 ||
        bound_session_inspection.step_output_count != 1 ||
        bound_session_inspection.host_binding_count != 4 ||
        bound_session_inspection.resource_binding_count != 0 ||
        bound_session_inspection.binding_shape_hash == 0
    )) {
        fprintf(stderr, "unexpected session inspection: model=%u backend=%u output_storage=%u persistent=%llu input=%llu output=%llu host=%llu resource=%llu binding_hash=%llu\n",
            bound_session_inspection.model_kind,
            bound_session_inspection.backend,
            bound_session_inspection.output_storage,
            (unsigned long long)bound_session_inspection.persistent_binding_count,
            (unsigned long long)bound_session_inspection.step_input_count,
            (unsigned long long)bound_session_inspection.step_output_count,
            (unsigned long long)bound_session_inspection.host_binding_count,
            (unsigned long long)bound_session_inspection.resource_binding_count,
            (unsigned long long)bound_session_inspection.binding_shape_hash);
        failed = 1;
    }

    zgml_step_result result = {0};
    if (!failed) {
        failed |= expect_status("zgml_session_step", zgml_session_step(session, NULL, &result));
    }

    if (!failed && result.output_len != 3) {
        fprintf(stderr, "expected 3 outputs, got %zu\n", result.output_len);
        failed = 1;
    }

    const float expected[] = {9.5f, 11.5f, 16.0f};
    float output[] = {0, 0, 0};
    if (!failed) {
        failed |= expect_status("zgml_buffer_read output", zgml_buffer_read(output_buffer, 0, output, sizeof(output)));
    }
    for (size_t i = 0; !failed && i < 3; i += 1) {
        if (!expect_close(output[i], expected[i])) {
            fprintf(stderr, "output[%zu] expected %.4f, got %.4f\n", i, expected[i], output[i]);
            failed = 1;
        }
    }

    zgml_runtime_profile program_profile = {0};
    if (!failed) {
        failed |= expect_status("zgml_program_runtime_profile", zgml_program_runtime_profile(program, &program_profile));
    }
    if (!failed && program_profile.call_count != 0) {
        fprintf(stderr, "expected program profile call_count 0, got %llu\n", (unsigned long long)program_profile.call_count);
        failed = 1;
    }

    zgml_runtime_profile session_profile = {0};
    if (!failed) {
        failed |= expect_status("zgml_session_runtime_profile", zgml_session_runtime_profile(session, &session_profile));
    }
    if (!failed && session_profile.call_count != 1) {
        fprintf(stderr, "expected session profile call_count 1, got %llu\n", (unsigned long long)session_profile.call_count);
        failed = 1;
    }
    if (!failed && session_profile.runtime_patch_call_count != 1) {
        fprintf(stderr, "expected session profile runtime_patch_call_count 1, got %llu\n", (unsigned long long)session_profile.runtime_patch_call_count);
        failed = 1;
    }
    if (!failed && session_profile.command_count == 0) {
        fprintf(stderr, "expected session profile command shape\n");
        failed = 1;
    }

    if (!failed) {
        failed |= expect_status("zgml_session_reset_runtime_profile", zgml_session_reset_runtime_profile(session));
    }
    if (!failed) {
        failed |= expect_status("zgml_session_runtime_profile", zgml_session_runtime_profile(session, &session_profile));
    }
    if (!failed && session_profile.call_count != 0) {
        fprintf(stderr, "expected reset session profile call_count 0, got %llu\n", (unsigned long long)session_profile.call_count);
        failed = 1;
    }

    if (!failed) {
        printf("zgml c ffi smoke ok: [%.1f, %.1f, %.1f]\n", output[0], output[1], output[2]);
    }

    zgml_session_free(session);
    zgml_program_free(program);
    zgml_model_free(model);
    zgml_buffer_free(output_buffer);
    zgml_buffer_free(input_buffer);
    zgml_buffer_free(bias_buffer);
    zgml_buffer_free(weights_buffer);

    if (failed) return 1;
    return 0;
}

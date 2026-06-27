#ifndef ZGML_H
#define ZGML_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define ZGML_API __declspec(dllimport)
#else
#define ZGML_API
#endif

typedef struct zgml_model zgml_model;
typedef struct zgml_program zgml_program;
typedef struct zgml_session zgml_session;
typedef struct zgml_buffer zgml_buffer;

typedef int zgml_status;

enum {
    ZGML_ABI_VERSION = 6,
};

enum {
    ZGML_OK = 0,
    ZGML_INVALID_ARGUMENT = 1,
    ZGML_OUT_OF_MEMORY = 2,
    ZGML_SHAPE_MISMATCH = 3,
    ZGML_COMPILE_FAILED = 4,
    ZGML_UNSUPPORTED = 5,
};

enum {
    ZGML_MODEL_AUTO = 0,
    ZGML_MODEL_TINY_LINEAR = 1,
    ZGML_MODEL_TINY_LLAMA = 2,
    ZGML_MODEL_SMOLLM_135M = 3,
    ZGML_MODEL_TINY_LLAMA_2LAYER = 4,
    ZGML_MODEL_TINY_MLP = 5,
    ZGML_MODEL_MODULE = 6,
};

enum {
    ZGML_TINY_MLP_ACTIVATION_RELU = 1,
    ZGML_TINY_MLP_ACTIVATION_GELU = 2,
    ZGML_TINY_MLP_ACTIVATION_SILU = 3,
    ZGML_TINY_MLP_ACTIVATION_SIGMOID = 4,
};

enum {
    ZGML_BACKEND_AUTO = 0,
    ZGML_BACKEND_CPU = 1,
    ZGML_BACKEND_METAL = 2,
    ZGML_BACKEND_WEBGPU = 3,
};

enum {
    ZGML_BUFFER_STORAGE_NONE = 0,
    ZGML_BUFFER_STORAGE_HOST = 1,
    ZGML_BUFFER_STORAGE_EXTERNAL_RESOURCE = 2,
};

enum {
    ZGML_EXECUTE_OUTPUT_NONE = 0,
    ZGML_EXECUTE_OUTPUT_LOGITS = 1,
};

enum {
    ZGML_RESOURCE_ACCESS_READ = 1u << 0,
    ZGML_RESOURCE_ACCESS_WRITE = 1u << 1,
    ZGML_RESOURCE_ACCESS_READ_WRITE = ZGML_RESOURCE_ACCESS_READ | ZGML_RESOURCE_ACCESS_WRITE,
};

enum {
    ZGML_PROGRAM_BUFFER_WEIGHTS = 1,
    ZGML_PROGRAM_BUFFER_BIAS = 2,
    ZGML_PROGRAM_BUFFER_INPUT = 3,
    ZGML_PROGRAM_BUFFER_OUTPUT = 4,
    ZGML_PROGRAM_BUFFER_LLAMA_K_CACHE = 5,
    ZGML_PROGRAM_BUFFER_LLAMA_V_CACHE = 6,
};

enum {
    ZGML_FEATURE_BUFFER_HANDLE = 1u << 0,
    ZGML_FEATURE_MODEL_AUTO = 1u << 1,
    ZGML_FEATURE_RUNTIME_PROFILE = 1u << 2,
    ZGML_FEATURE_WEBGPU_COMPILE_ONLY = 1u << 3,
    ZGML_FEATURE_WASM_EXPORTS = 1u << 4,
    ZGML_FEATURE_NATIVE_BUFFER_IO = 1u << 5,
    ZGML_FEATURE_NATIVE_ARGMAX = 1u << 6,
    ZGML_FEATURE_NATIVE_EXECUTE_ARGMAX = 1u << 7,
    ZGML_FEATURE_NATIVE_TOPK_SAMPLE = 1u << 8,
    ZGML_FEATURE_PROGRAM_REQUIREMENTS = 1u << 9,
    ZGML_FEATURE_PROGRAM_OUTPUT_BUFFER = 1u << 10,
    ZGML_FEATURE_NATIVE_GENERATE_SAMPLE = 1u << 11,
    ZGML_FEATURE_NATIVE_GENERATE_ARGMAX = 1u << 12,
    ZGML_FEATURE_SESSION_MODEL_BINDING = 1u << 13,
    ZGML_FEATURE_EXTERNAL_BUFFER = 1u << 14,
    ZGML_FEATURE_EXTERNAL_RESOURCE_BUFFER = 1u << 15,
    ZGML_FEATURE_LLAMA_KV_RESOURCE_BINDING = 1u << 16,
    ZGML_FEATURE_LLAMA_KV_CACHE_REQUIREMENTS = 1u << 17,
    ZGML_FEATURE_PROGRAM_BUFFER_FACTORY = 1u << 18,
    ZGML_FEATURE_EXTERNAL_RESOURCE_ACCESS = 1u << 19,
    ZGML_FEATURE_MODEL_INSPECTION = 1u << 20,
    ZGML_FEATURE_PROGRAM_MODEL_COMPATIBILITY = 1u << 21,
    ZGML_FEATURE_BUFFER_INSPECTION = 1u << 22,
    ZGML_FEATURE_SESSION_INSPECTION = 1u << 23,
    ZGML_FEATURE_PROGRAM_RESOURCE_INSPECTION = 1u << 24,
    ZGML_FEATURE_PROGRAM_MEMORY_INSPECTION = 1u << 25,
    ZGML_FEATURE_PROGRAM_SHAPE_INSPECTION = 1u << 26,
    ZGML_FEATURE_PROGRAM_PATCH_ENVELOPE_INSPECTION = 1u << 27,
    ZGML_FEATURE_SESSION_BINDING_SHAPE_INSPECTION = 1u << 28,
    ZGML_FEATURE_NATIVE_WGPU_EXECUTION = 1u << 29,
    ZGML_FEATURE_PROGRAM_DEVICE_BUFFER = 1u << 30,
    ZGML_FEATURE_PROGRAM_DEVICE_BUFFER_IMPORT = 1ull << 31,
    ZGML_FEATURE_PROGRAM_DISPATCH_PLAN_INSPECTION = 1ull << 32,
    ZGML_FEATURE_ABI_STRUCT_SIZE = 1ull << 33,
    ZGML_FEATURE_EXPERIMENTAL_LLAMA_WGPU_EXECUTION = 1ull << 34,
    ZGML_FEATURE_MODEL_PATH_PROBE = 1ull << 35,
    ZGML_FEATURE_SUPPORTED_CHECKPOINTS = 1ull << 36,
    ZGML_FEATURE_SAFETENSORS_HEADER_PROBE = 1ull << 37,
    ZGML_FEATURE_SAFETENSORS_DATA_LOAD = 1ull << 38,
    ZGML_FEATURE_SAFETENSORS_DATA_PROBE = 1ull << 39,
    ZGML_FEATURE_NATIVE_TINY_MLP = 1ull << 40,
    ZGML_FEATURE_NATIVE_MODULE_PROGRAM = 1ull << 41,
    ZGML_FEATURE_PROGRAM_BINDING_REQUIREMENTS = 1ull << 42,
    ZGML_FEATURE_SESSION_PERSISTENT_UPLOAD = 1ull << 43,
    ZGML_FEATURE_NATIVE_MODULE_ACTIVATION_CHAIN = 1ull << 44,
    ZGML_FEATURE_NATIVE_EAGER_LINEAR = 1ull << 45,
    ZGML_FEATURE_NATIVE_EAGER_LINEAR_ACTIVATION = 1ull << 46,
    ZGML_FEATURE_NATIVE_TRAINING_STEP = 1ull << 47,
    ZGML_FEATURE_NATIVE_EAGER_SOFTMAX = 1ull << 48,
    ZGML_FEATURE_NATIVE_EAGER_MATMUL = 1ull << 49,
    ZGML_FEATURE_NATIVE_EAGER_ACTIVATION = 1ull << 50,
    ZGML_FEATURE_NATIVE_EAGER_ELEMENTWISE = 1ull << 51,
    ZGML_FEATURE_NATIVE_EAGER_REDUCE = 1ull << 52,
    ZGML_FEATURE_NATIVE_EAGER_CONV2D = 1ull << 53,
    ZGML_FEATURE_NATIVE_EAGER_POOL2D = 1ull << 54,
};

enum {
    ZGML_MODULE_OP_LINEAR = 1,
    ZGML_MODULE_OP_ACTIVATION = 2,
    ZGML_MODULE_OP_SOFTMAX = 3,
    ZGML_MODULE_OP_LAYER_NORM = 4,
    ZGML_MODULE_OP_RMS_NORM = 5,
    ZGML_MODULE_OP_EMBEDDING = 6,
    ZGML_MODULE_OP_LOG_SOFTMAX = 7,
    ZGML_MODULE_OP_RESHAPE = 8,
    ZGML_MODULE_OP_BROADCAST_TO = 9,
    ZGML_MODULE_OP_NARROW = 10,
    ZGML_MODULE_OP_TRANSPOSE = 11,
    ZGML_MODULE_OP_REDUCE_SUM = 12,
    ZGML_MODULE_OP_REDUCE_MEAN = 13,
    ZGML_MODULE_OP_REDUCE_MAX = 14,
    ZGML_MODULE_OP_SLICE = 15,
    ZGML_MODULE_OP_ACTIVATION_CHAIN = 16,
    ZGML_MODULE_OP_MAX_POOL2D = 17,
    ZGML_MODULE_OP_AVG_POOL2D = 18,
    ZGML_MODULE_OP_CONV2D = 19,
    ZGML_MODULE_OP_ADD = 20,
    ZGML_MODULE_OP_REDUCE_MIN = 21,
    ZGML_MODULE_OP_FEATURE_AFFINE = 22,
    ZGML_MODULE_OP_DIAGONAL = 23,
    ZGML_MODULE_OP_REDUCE_ARGMAX = 24,
    ZGML_MODULE_OP_REDUCE_ARGMIN = 25,
    ZGML_MODULE_OP_REDUCE_PROD = 26,
};

enum {
    ZGML_MODULE_ACTIVATION_RELU = 1,
    ZGML_MODULE_ACTIVATION_GELU = 2,
    ZGML_MODULE_ACTIVATION_SILU = 3,
    ZGML_MODULE_ACTIVATION_SIGMOID = 4,
    ZGML_MODULE_ACTIVATION_EXP = 5,
    ZGML_MODULE_ACTIVATION_LOG = 6,
    ZGML_MODULE_ACTIVATION_NEG = 7,
    ZGML_MODULE_ACTIVATION_RECIP = 8,
    ZGML_MODULE_ACTIVATION_ABS = 9,
    ZGML_MODULE_ACTIVATION_SQRT = 10,
    ZGML_MODULE_ACTIVATION_SQUARE = 11,
    ZGML_MODULE_ACTIVATION_SQR = ZGML_MODULE_ACTIVATION_SQUARE,
    ZGML_MODULE_ACTIVATION_SGN = 12,
    ZGML_MODULE_ACTIVATION_SIGN = ZGML_MODULE_ACTIVATION_SGN,
    ZGML_MODULE_ACTIVATION_STEP = 13,
    ZGML_MODULE_ACTIVATION_TANH = 14,
};

enum {
    ZGML_MODULE_FLAG_WEIGHT = 1u << 0,
    ZGML_MODULE_FLAG_BIAS = 1u << 1,
};

enum {
    ZGML_ABI_STRUCT_RUNTIME_INFO = 1,
    ZGML_ABI_STRUCT_MODEL_DESC = 2,
    ZGML_ABI_STRUCT_MODEL_LOAD_DESC = 3,
    ZGML_ABI_STRUCT_MODEL_INSPECTION = 4,
    ZGML_ABI_STRUCT_PROGRAM_MODEL_COMPATIBILITY = 5,
    ZGML_ABI_STRUCT_SESSION_INSPECTION = 6,
    ZGML_ABI_STRUCT_COMPILE_DESC = 7,
    ZGML_ABI_STRUCT_BIND_DESC = 8,
    ZGML_ABI_STRUCT_BUFFER_DESC = 9,
    ZGML_ABI_STRUCT_BUFFER_INSPECTION = 10,
    ZGML_ABI_STRUCT_EXTERNAL_RESOURCE_DESC = 11,
    ZGML_ABI_STRUCT_DEVICE_BUFFER_IMPORT_DESC = 12,
    ZGML_ABI_STRUCT_BUFFER_BIND_DESC = 13,
    ZGML_ABI_STRUCT_LLAMA_KV_CACHE_BIND_DESC = 14,
    ZGML_ABI_STRUCT_LLAMA_BUFFER_BIND_DESC = 15,
    ZGML_ABI_STRUCT_STEP_DESC = 16,
    ZGML_ABI_STRUCT_STEP_RESULT = 17,
    ZGML_ABI_STRUCT_TOKEN_STEP_DESC = 18,
    ZGML_ABI_STRUCT_TOKEN_ADVANCE_DESC = 19,
    ZGML_ABI_STRUCT_TOKEN_ADVANCE_TOKENS_DESC = 20,
    ZGML_ABI_STRUCT_TOKEN_PREFILL_DESC = 21,
    ZGML_ABI_STRUCT_TOKEN_EXECUTE_DESC = 22,
    ZGML_ABI_STRUCT_TOKEN_ARGMAX_DESC = 23,
    ZGML_ABI_STRUCT_TOKEN_ARGMAX_RESULT = 24,
    ZGML_ABI_STRUCT_TOKEN_EXECUTE_ARGMAX_DESC = 25,
    ZGML_ABI_STRUCT_TOKEN_GENERATE_ARGMAX_DESC = 26,
    ZGML_ABI_STRUCT_TOKEN_GENERATE_ARGMAX_RESULT = 27,
    ZGML_ABI_STRUCT_TOKEN_SAMPLE_DESC = 28,
    ZGML_ABI_STRUCT_TOKEN_SAMPLE_RESULT = 29,
    ZGML_ABI_STRUCT_TOKEN_EXECUTE_SAMPLE_DESC = 30,
    ZGML_ABI_STRUCT_TOKEN_GENERATE_SAMPLE_DESC = 31,
    ZGML_ABI_STRUCT_TOKEN_GENERATE_SAMPLE_RESULT = 32,
    ZGML_ABI_STRUCT_PROGRAM_REQUIREMENTS = 33,
    ZGML_ABI_STRUCT_LLAMA_KV_CACHE_REQUIREMENTS = 34,
    ZGML_ABI_STRUCT_PROGRAM_INSPECTION = 35,
    ZGML_ABI_STRUCT_LLAMA_PROGRAM_INSPECTION = 36,
    ZGML_ABI_STRUCT_RUNTIME_PROFILE = 37,
    ZGML_ABI_STRUCT_SAFETENSORS_HEADER_PROBE_DESC = 38,
    ZGML_ABI_STRUCT_SAFETENSORS_DATA_LOAD_DESC = 39,
    ZGML_ABI_STRUCT_MODULE_OP_DESC = 40,
    ZGML_ABI_STRUCT_MODULE_DESC = 41,
};

enum {
    ZGML_SAMPLE_TOP_K_MAX = 256,
};

typedef struct zgml_runtime_info {
    uint32_t abi_version;
    uint32_t size_t_bytes;
    uint32_t pointer_bytes;
    uint32_t token_id_bytes;
    uint64_t feature_flags;
} zgml_runtime_info;

typedef struct zgml_model_desc {
    uint32_t kind;
    uint32_t activation;
    size_t input_len;
    size_t output_len;
    size_t hidden_len;
} zgml_model_desc;

typedef struct zgml_model_load_desc {
    uint32_t kind;
    const char *path;
    size_t path_len;
} zgml_model_load_desc;

typedef struct zgml_safetensors_header_probe_desc {
    uint32_t kind;
    uint32_t reserved;
    const char *header;
    size_t header_len;
} zgml_safetensors_header_probe_desc;

typedef struct zgml_safetensors_data_load_desc {
    uint32_t kind;
    uint32_t reserved;
    const void *data;
    size_t data_len;
} zgml_safetensors_data_load_desc;

typedef struct zgml_module_op_desc {
    uint32_t kind;
    uint32_t activation;
    uint32_t flags;
    /*
     * For ZGML_MODULE_OP_SLICE, reserved carries the positive step.
     * For ZGML_MODULE_OP_ACTIVATION_CHAIN, flags carries the activation count
     * and activation/a/b/c carry up to four activation ids.
     * Other ops require reserved to be zero.
     */
    uint32_t reserved;
    size_t a;
    size_t b;
    size_t c;
    double eps;
} zgml_module_op_desc;

typedef struct zgml_module_desc {
    const size_t *input_shape;
    size_t input_rank;
    const zgml_module_op_desc *ops;
    size_t op_count;
} zgml_module_desc;

typedef struct zgml_model_inspection {
    uint32_t model_kind;
    uint32_t reserved;
    uint64_t input_len;
    uint64_t output_len;
    uint64_t vocab_size;
    uint64_t max_seq_len;
    uint64_t d_model;
    uint64_t n_layers;
    uint64_t n_heads;
    uint64_t n_kv_heads;
    uint64_t d_ff;
    double rope_base;
    double rms_norm_eps;
    uint64_t tied_lm_head;
} zgml_model_inspection;

typedef struct zgml_program_model_compatibility {
    uint32_t program_model_kind;
    uint32_t model_kind;
    uint64_t compatible;
} zgml_program_model_compatibility;

typedef struct zgml_session_inspection {
    uint32_t model_kind;
    uint32_t backend;
    uint32_t output_storage;
    uint32_t kv_cache_storage;
    uint64_t position;
    uint64_t context_len;
    uint64_t persistent_binding_count;
    uint64_t step_input_count;
    uint64_t step_output_count;
    uint64_t host_binding_count;
    uint64_t resource_binding_count;
    uint64_t binding_shape_hash;
} zgml_session_inspection;

typedef struct zgml_compile_desc {
    uint32_t backend;
    uint32_t reserved;
    size_t context_len;
    size_t batch;
} zgml_compile_desc;

typedef struct zgml_bind_desc {
    const float *weights;
    size_t weights_len;
    const float *bias;
    size_t bias_len;
    const float *input;
    size_t input_len;
    float *output;
    size_t output_len;
} zgml_bind_desc;

typedef struct zgml_buffer_desc {
    size_t byte_len;
} zgml_buffer_desc;

typedef struct zgml_buffer_inspection {
    uint32_t storage;
    uint32_t placement;
    uint32_t access_flags;
    uint32_t reserved;
    uint64_t byte_len;
    uint64_t handle;
    uint64_t byte_offset;
    uint64_t resource_byte_len;
} zgml_buffer_inspection;

typedef struct zgml_external_resource_desc {
    uint32_t placement;
    uint32_t access_flags;
    uintptr_t handle;
    size_t byte_offset;
    size_t byte_len;
} zgml_external_resource_desc;

typedef struct zgml_device_buffer_import_desc {
    uint32_t placement;
    uint32_t reserved;
    uintptr_t device_handle;
    uintptr_t buffer_handle;
    size_t byte_offset;
    size_t byte_len;
} zgml_device_buffer_import_desc;

typedef struct zgml_buffer_bind_desc {
    zgml_buffer *weights;
    size_t weights_len;
    zgml_buffer *bias;
    size_t bias_len;
    zgml_buffer *input;
    size_t input_len;
    zgml_buffer *output;
    size_t output_len;
} zgml_buffer_bind_desc;

typedef struct zgml_llama_kv_cache_bind_desc {
    zgml_buffer **k;
    zgml_buffer **v;
    size_t len;
} zgml_llama_kv_cache_bind_desc;

typedef struct zgml_llama_buffer_bind_desc {
    zgml_buffer *output;
    size_t output_len;
    const zgml_llama_kv_cache_bind_desc *kv_cache;
} zgml_llama_buffer_bind_desc;

typedef struct zgml_step_desc {
    const float *input;
    size_t input_len;
    float *output;
    size_t output_len;
} zgml_step_desc;

typedef struct zgml_step_result {
    size_t output_len;
} zgml_step_result;

typedef struct zgml_token_step_desc {
    uint32_t token;
    float *output;
    size_t output_len;
} zgml_token_step_desc;

typedef struct zgml_token_advance_desc {
    uint32_t token;
} zgml_token_advance_desc;

typedef struct zgml_token_advance_tokens_desc {
    const uint32_t *tokens;
    size_t tokens_len;
} zgml_token_advance_tokens_desc;

typedef struct zgml_token_prefill_desc {
    const uint32_t *tokens;
    size_t tokens_len;
    float *output;
    size_t output_len;
} zgml_token_prefill_desc;

typedef struct zgml_token_execute_desc {
    const uint32_t *tokens;
    size_t tokens_len;
    uint32_t output_policy;
    uint32_t reserved;
    float *output;
    size_t output_len;
} zgml_token_execute_desc;

typedef struct zgml_token_argmax_desc {
    const float *logits;
    size_t logits_len;
    uint32_t reserved;
} zgml_token_argmax_desc;

typedef struct zgml_token_argmax_result {
    uint32_t token;
    float logit;
} zgml_token_argmax_result;

typedef struct zgml_token_execute_argmax_desc {
    const uint32_t *tokens;
    size_t tokens_len;
    uint32_t reserved;
} zgml_token_execute_argmax_desc;

typedef struct zgml_token_generate_argmax_desc {
    const uint32_t *tokens;
    size_t tokens_len;
    uint32_t *output_tokens;
    size_t output_tokens_len;
    uint32_t reserved;
} zgml_token_generate_argmax_desc;

typedef struct zgml_token_generate_argmax_result {
    size_t tokens_generated;
    uint32_t last_token;
    float last_logit;
} zgml_token_generate_argmax_result;

typedef struct zgml_token_sample_desc {
    const float *logits;
    size_t logits_len;
    uint32_t top_k;
    uint32_t seed;
    float temperature;
    uint32_t reserved;
} zgml_token_sample_desc;

typedef struct zgml_token_sample_result {
    uint32_t token;
    float logit;
} zgml_token_sample_result;

typedef struct zgml_token_execute_sample_desc {
    const uint32_t *tokens;
    size_t tokens_len;
    uint32_t top_k;
    uint32_t seed;
    float temperature;
    uint32_t reserved;
} zgml_token_execute_sample_desc;

typedef struct zgml_token_generate_sample_desc {
    const uint32_t *tokens;
    size_t tokens_len;
    uint32_t *output_tokens;
    size_t output_tokens_len;
    uint32_t top_k;
    uint32_t seed;
    float temperature;
    uint32_t reserved;
} zgml_token_generate_sample_desc;

typedef struct zgml_token_generate_sample_result {
    size_t tokens_generated;
    uint32_t last_token;
    float last_logit;
} zgml_token_generate_sample_result;

typedef struct zgml_program_requirements {
    uint32_t model_kind;
    uint32_t scalar_bytes;
    uint32_t token_id_bytes;
    uint32_t reserved;
    size_t input_len;
    size_t input_byte_len;
    size_t output_len;
    size_t weights_len;
    size_t weights_byte_len;
    size_t bias_len;
    size_t bias_byte_len;
    size_t parameter_len;
    size_t parameter_byte_len;
    size_t logits_len;
    size_t output_byte_len;
    size_t context_len;
    size_t batch;
    size_t max_token_window;
} zgml_program_requirements;

typedef struct zgml_llama_kv_cache_requirements {
    uint32_t model_kind;
    uint32_t scalar_bytes;
    uint32_t n_layers;
    uint32_t reserved;
    size_t context_len;
    size_t k_buffer_byte_len;
    size_t v_buffer_byte_len;
    size_t buffer_byte_len;
} zgml_llama_kv_cache_requirements;

typedef struct zgml_program_inspection {
    uint64_t backend;
    uint64_t execution_supported;
    uint64_t external_resources_supported;
    uint64_t buffer_count;
    uint64_t buffer_byte_len;
    uint64_t initial_upload_count;
    uint64_t qweight_count;
    uint64_t command_count;
    uint64_t command_stencil_hash;
    uint64_t runtime_patch_holes;
    uint64_t runtime_patch_cache_write_pos_holes;
    uint64_t runtime_patch_attention_seq_kv_holes;
    uint64_t runtime_patch_stencil_hash;
    uint64_t command_op_count;
    uint64_t command_row_count;
    uint64_t command_projection_count;
    uint64_t command_attention_count;
    uint64_t command_movement_count;
    uint64_t command_elementwise_count;
    uint64_t command_rope_count;
    uint64_t op_count;
    uint64_t buffer_element_count;
    uint64_t runtime_patch_max_cache_write_pos;
    uint64_t runtime_patch_max_attention_seq_kv;
    uint64_t backend_dispatch_count;
    uint64_t dispatch_plan_supported;
    uint64_t dispatch_plan_covered_op_count;
    uint64_t dispatch_plan_first_unsupported_op;
    uint64_t dispatch_plan_projection_count;
    uint64_t dispatch_plan_row_count;
    uint64_t dispatch_plan_attention_count;
    uint64_t dispatch_plan_movement_count;
    uint64_t dispatch_plan_elementwise_count;
    uint64_t dispatch_plan_rope_count;
    uint64_t dispatch_plan_quantized_projection_count;
    uint64_t binding_requirement_hash;
    uint64_t persistent_requirement_count;
    uint64_t step_input_requirement_count;
    uint64_t step_output_requirement_count;
} zgml_program_inspection;

typedef struct zgml_llama_program_inspection {
    uint64_t vocab_size;
    uint64_t max_seq_len;
    uint64_t context_len;
    uint64_t batch;
    uint64_t d_model;
    uint64_t n_layers;
    uint64_t n_heads;
    uint64_t n_kv_heads;
    uint64_t semantic_stage_count;
    uint64_t semantic_token_count;
    uint64_t semantic_layer_stage_count;
    uint64_t semantic_terminal_stage_count;
    uint64_t semantic_runtime_patch_holes;
    uint64_t semantic_runtime_patch_cache_write_pos_holes;
    uint64_t semantic_runtime_patch_attention_seq_kv_holes;
} zgml_llama_program_inspection;

typedef struct zgml_runtime_profile {
    uint64_t call_count;
    uint64_t backend_op_count;
    uint64_t fallback_op_count;
    uint64_t backend_dispatch_count;
    uint64_t sync_count;
    uint64_t runtime_patch_call_count;
    uint64_t runtime_patch_changed_count;
    uint64_t runtime_patch_invalid_count;
    uint64_t runtime_patch_holes;
    uint64_t runtime_patch_cache_write_pos_holes;
    uint64_t runtime_patch_attention_seq_kv_holes;
    uint64_t runtime_patch_stencil_hash;
    uint64_t command_count;
    uint64_t command_stencil_hash;
    uint64_t command_op_count;
    uint64_t command_row_count;
    uint64_t command_projection_count;
    uint64_t command_attention_count;
    uint64_t command_movement_count;
    uint64_t command_elementwise_count;
    uint64_t command_rope_count;
} zgml_runtime_profile;

ZGML_API size_t zgml_abi_struct_size(uint32_t kind);
ZGML_API zgml_status zgml_get_runtime_info(zgml_runtime_info *out_info);
ZGML_API const char *zgml_status_name(zgml_status code);
ZGML_API size_t zgml_wasm_alloc(size_t len);
ZGML_API void zgml_wasm_free(size_t ptr, size_t len);
ZGML_API zgml_status zgml_buffer_create(const zgml_buffer_desc *desc, zgml_buffer **out_buffer);
ZGML_API zgml_status zgml_buffer_wrap(void *data, size_t byte_len, zgml_buffer **out_buffer);
ZGML_API zgml_status zgml_buffer_wrap_resource(const zgml_external_resource_desc *desc, zgml_buffer **out_buffer);
ZGML_API void *zgml_buffer_data(zgml_buffer *buffer);
ZGML_API size_t zgml_buffer_size(zgml_buffer *buffer);
ZGML_API zgml_status zgml_buffer_inspect(zgml_buffer *buffer, zgml_buffer_inspection *out_inspection);
ZGML_API zgml_status zgml_buffer_write(zgml_buffer *buffer, size_t byte_offset, const void *src, size_t byte_len);
ZGML_API zgml_status zgml_buffer_read(zgml_buffer *buffer, size_t byte_offset, void *dst, size_t byte_len);
ZGML_API void zgml_buffer_free(zgml_buffer *buffer);
ZGML_API zgml_status zgml_eager_linear_f32(
    const float *input,
    size_t input_len,
    const float *weights,
    size_t weights_len,
    const float *bias,
    size_t bias_len,
    float *output,
    size_t output_len,
    size_t batch,
    size_t in_features,
    size_t out_features
);
ZGML_API zgml_status zgml_eager_matmul_f32(
    const float *lhs,
    size_t lhs_len,
    const float *rhs,
    size_t rhs_len,
    float *output,
    size_t output_len,
    size_t rows,
    size_t shared,
    size_t cols
);
ZGML_API zgml_status zgml_eager_linear_activation_f32(
    const float *input,
    size_t input_len,
    const float *weights,
    size_t weights_len,
    const float *bias,
    size_t bias_len,
    float *output,
    size_t output_len,
    size_t batch,
    size_t in_features,
    size_t out_features,
    uint32_t activation
);
ZGML_API zgml_status zgml_eager_activation_f32(
    const float *input,
    size_t input_len,
    float *output,
    size_t output_len,
    uint32_t activation
);
ZGML_API zgml_status zgml_eager_elementwise_f32(
    const float *lhs,
    size_t lhs_len,
    const float *rhs,
    size_t rhs_len,
    float *output,
    size_t output_len,
    uint32_t op
);
ZGML_API zgml_status zgml_eager_reduce_f32(
    const float *input,
    size_t input_len,
    float *output,
    size_t output_len,
    uint32_t op
);
ZGML_API zgml_status zgml_eager_conv2d_f32(
    const float *input,
    size_t input_len,
    const float *weights,
    size_t weights_len,
    const float *bias,
    size_t bias_len,
    float *output,
    size_t output_len,
    size_t batch,
    size_t in_channels,
    size_t height,
    size_t width,
    size_t out_channels,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t padding_h,
    size_t padding_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t out_h,
    size_t out_w
);
ZGML_API zgml_status zgml_eager_pool2d_f32(
    const float *input,
    size_t input_len,
    float *output,
    size_t output_len,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t padding_h,
    size_t padding_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t out_h,
    size_t out_w,
    uint32_t op,
    uint32_t ceil_mode,
    uint32_t count_include_pad
);
ZGML_API zgml_status zgml_eager_softmax_f32(
    const float *input,
    size_t input_len,
    float *output,
    size_t output_len,
    size_t rows,
    size_t cols,
    uint32_t log_softmax
);
ZGML_API zgml_status zgml_train_linear_mse_sgd_f32(
    const float *input,
    size_t input_len,
    const float *target,
    size_t target_len,
    float *weight,
    size_t weight_len,
    float *bias,
    size_t bias_len,
    float *output,
    size_t output_len,
    float *grad_weight,
    size_t grad_weight_len,
    size_t batch,
    size_t in_features,
    size_t out_features,
    float lr,
    float weight_decay,
    float *out_loss
);
ZGML_API zgml_status zgml_train_mlp_relu_cross_entropy_adam_f32(
    const float *input,
    size_t input_len,
    const uint32_t *targets,
    size_t target_len,
    float *w1,
    size_t w1_len,
    float *b1,
    size_t b1_len,
    float *w2,
    size_t w2_len,
    float *b2,
    size_t b2_len,
    float *mw1,
    size_t mw1_len,
    float *vw1,
    size_t vw1_len,
    float *mb1,
    size_t mb1_len,
    float *vb1,
    size_t vb1_len,
    float *mw2,
    size_t mw2_len,
    float *vw2,
    size_t vw2_len,
    float *mb2,
    size_t mb2_len,
    float *vb2,
    size_t vb2_len,
    float *hidden,
    size_t hidden_len,
    float *logits,
    size_t logits_len,
    float *grad_hidden,
    size_t grad_hidden_len,
    float *grad_w1,
    size_t grad_w1_len,
    float *grad_w2,
    size_t grad_w2_len,
    size_t batch,
    size_t in_features,
    size_t hidden_features,
    size_t classes,
    size_t step,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float *out_loss,
    size_t *out_correct
);
ZGML_API zgml_status zgml_train_mlp_relu_cross_entropy_adamw_f32(
    const float *input,
    size_t input_len,
    const uint32_t *targets,
    size_t target_len,
    float *w1,
    size_t w1_len,
    float *b1,
    size_t b1_len,
    float *w2,
    size_t w2_len,
    float *b2,
    size_t b2_len,
    float *mw1,
    size_t mw1_len,
    float *vw1,
    size_t vw1_len,
    float *mb1,
    size_t mb1_len,
    float *vb1,
    size_t vb1_len,
    float *mw2,
    size_t mw2_len,
    float *vw2,
    size_t vw2_len,
    float *mb2,
    size_t mb2_len,
    float *vb2,
    size_t vb2_len,
    float *hidden,
    size_t hidden_len,
    float *logits,
    size_t logits_len,
    float *grad_hidden,
    size_t grad_hidden_len,
    float *grad_w1,
    size_t grad_w1_len,
    float *grad_w2,
    size_t grad_w2_len,
    size_t batch,
    size_t in_features,
    size_t hidden_features,
    size_t classes,
    size_t step,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float *out_loss,
    size_t *out_correct
);
ZGML_API zgml_status zgml_model_create(const zgml_model_desc *desc, zgml_model **out_model);
ZGML_API zgml_status zgml_model_load_path(const zgml_model_load_desc *desc, zgml_model **out_model);
ZGML_API zgml_status zgml_model_load_safetensors_data(const zgml_safetensors_data_load_desc *desc, zgml_model **out_model);
ZGML_API zgml_status zgml_model_probe_path(const zgml_model_load_desc *desc, zgml_model_inspection *out_inspection);
ZGML_API zgml_status zgml_model_probe_safetensors_data(const zgml_safetensors_data_load_desc *desc, zgml_model_inspection *out_inspection);
ZGML_API zgml_status zgml_model_probe_safetensors_header(const zgml_safetensors_header_probe_desc *desc, zgml_model_inspection *out_inspection);
ZGML_API size_t zgml_supported_checkpoint_count(void);
ZGML_API zgml_status zgml_supported_checkpoint_inspect(size_t index, zgml_model_inspection *out_inspection);
ZGML_API zgml_status zgml_model_inspect(zgml_model *model, zgml_model_inspection *out_inspection);
ZGML_API zgml_status zgml_program_compile(zgml_model *model, const zgml_compile_desc *desc, zgml_program **out_program);
ZGML_API zgml_status zgml_module_program_compile(const zgml_module_desc *module_desc, const zgml_compile_desc *compile_desc, zgml_program **out_program);
ZGML_API zgml_status zgml_program_get_requirements(zgml_program *program, zgml_program_requirements *out_requirements);
ZGML_API zgml_status zgml_program_check_model_compatibility(zgml_program *program, zgml_model *model, zgml_program_model_compatibility *out_compatibility);
ZGML_API zgml_status zgml_program_create_buffer(zgml_program *program, uint32_t kind, zgml_buffer **out_buffer);
ZGML_API zgml_status zgml_program_create_device_buffer(zgml_program *program, uint32_t kind, uint32_t placement, zgml_buffer **out_buffer);
ZGML_API zgml_status zgml_program_get_device_handle(zgml_program *program, uint32_t placement, uintptr_t *out_handle);
ZGML_API zgml_status zgml_program_import_device_buffer(zgml_program *program, uint32_t kind, const zgml_device_buffer_import_desc *desc, zgml_buffer **out_buffer);
ZGML_API zgml_status zgml_program_create_output_buffer(zgml_program *program, zgml_buffer **out_buffer);
ZGML_API zgml_status zgml_program_inspect(zgml_program *program, zgml_program_inspection *out_inspection);
ZGML_API zgml_status zgml_llama_program_inspect(zgml_program *program, zgml_llama_program_inspection *out_inspection);
ZGML_API zgml_status zgml_llama_program_get_kv_cache_requirements(zgml_program *program, zgml_llama_kv_cache_requirements *out_requirements);
ZGML_API zgml_status zgml_program_runtime_profile(zgml_program *program, zgml_runtime_profile *out_profile);
ZGML_API zgml_status zgml_program_reset_runtime_profile(zgml_program *program);
ZGML_API zgml_status zgml_session_bind(zgml_program *program, const zgml_bind_desc *desc, zgml_session **out_session);
ZGML_API zgml_status zgml_session_bind_model(zgml_program *program, zgml_model *model, const zgml_bind_desc *desc, zgml_session **out_session);
ZGML_API zgml_status zgml_session_bind_model_buffers(zgml_program *program, zgml_model *model, const zgml_buffer_bind_desc *desc, zgml_session **out_session);
ZGML_API zgml_status zgml_session_bind_buffers(zgml_program *program, const zgml_buffer_bind_desc *desc, zgml_session **out_session);
ZGML_API zgml_status zgml_llama_session_bind_model_buffers(zgml_program *program, zgml_model *model, const zgml_llama_buffer_bind_desc *desc, zgml_session **out_session);
ZGML_API zgml_status zgml_llama_session_bind_buffers(zgml_program *program, const zgml_llama_buffer_bind_desc *desc, zgml_session **out_session);
ZGML_API zgml_status zgml_session_upload_persistent(zgml_session *session);
ZGML_API zgml_status zgml_session_upload_persistent_range(zgml_session *session, size_t first, size_t len);
ZGML_API zgml_status zgml_session_step(zgml_session *session, const zgml_step_desc *desc, zgml_step_result *out_result);
ZGML_API zgml_status zgml_session_step_direct(zgml_session *session, const float *input, size_t input_len, float *output, size_t output_len);
ZGML_API zgml_status zgml_session_step_no_output(zgml_session *session, const zgml_step_desc *desc, zgml_step_result *out_result);
ZGML_API zgml_status zgml_session_step_token(zgml_session *session, const zgml_token_step_desc *desc, zgml_step_result *out_result);
ZGML_API zgml_status zgml_session_advance_token(zgml_session *session, const zgml_token_advance_desc *desc);
ZGML_API zgml_status zgml_session_advance_tokens(zgml_session *session, const zgml_token_advance_tokens_desc *desc);
ZGML_API zgml_status zgml_session_prefill_tokens(zgml_session *session, const zgml_token_prefill_desc *desc, zgml_step_result *out_result);
ZGML_API zgml_status zgml_session_execute_tokens(zgml_session *session, const zgml_token_execute_desc *desc, zgml_step_result *out_result);
ZGML_API zgml_status zgml_session_argmax_token(zgml_session *session, const zgml_token_argmax_desc *desc, zgml_token_argmax_result *out_result);
ZGML_API zgml_status zgml_session_execute_argmax_tokens(zgml_session *session, const zgml_token_execute_argmax_desc *desc, zgml_token_argmax_result *out_result);
ZGML_API zgml_status zgml_session_generate_argmax_tokens(zgml_session *session, const zgml_token_generate_argmax_desc *desc, zgml_token_generate_argmax_result *out_result);
ZGML_API zgml_status zgml_session_sample_token(zgml_session *session, const zgml_token_sample_desc *desc, zgml_token_sample_result *out_result);
ZGML_API zgml_status zgml_session_execute_sample_tokens(zgml_session *session, const zgml_token_execute_sample_desc *desc, zgml_token_sample_result *out_result);
ZGML_API zgml_status zgml_session_generate_sample_tokens(zgml_session *session, const zgml_token_generate_sample_desc *desc, zgml_token_generate_sample_result *out_result);
ZGML_API zgml_status zgml_session_position(zgml_session *session, size_t *out_position);
ZGML_API zgml_status zgml_session_inspect(zgml_session *session, zgml_session_inspection *out_inspection);
ZGML_API zgml_status zgml_session_reset(zgml_session *session);
ZGML_API zgml_status zgml_session_runtime_profile(zgml_session *session, zgml_runtime_profile *out_profile);
ZGML_API zgml_status zgml_session_reset_runtime_profile(zgml_session *session);
ZGML_API void zgml_session_free(zgml_session *session);
ZGML_API void zgml_program_free(zgml_program *program);
ZGML_API void zgml_model_free(zgml_model *model);

#ifdef __cplusplus
}
#endif

#endif

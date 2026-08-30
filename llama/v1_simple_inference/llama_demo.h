// llama_demo.h
// 本文件为 llama.cpp C API 的"轻量级声明"。
//
// 设计思路：
//   我们不直接链接 llama.lib（Ollama 没有附带 .lib 导入库），
//   而是在运行时通过 LoadLibrary + GetProcAddress 动态加载
//   Ollama 自带的 libllama.dll。因此这里只需要：
//     1) 与 DLL 中布局一致的结构体定义
//     2) 每个 C API 函数的函数指针类型
//   结构体字段顺序、类型必须与 llama.cpp 的 llama.h 完全一致，
//   否则会出现内存布局错位导致崩溃。
//
// 对应 llama.cpp 版本：Ollama 0.32.6 内置版本（2026 年）。

#pragma once

#include <cstdint>
#include <cstddef>
#include <windows.h>

// ---------------------------------------------------------------------------
// 基础类型
// ---------------------------------------------------------------------------
typedef int32_t llama_pos;     // 序列中的位置
typedef int32_t llama_token;   // token 编号
typedef int32_t llama_seq_id;  // 序列 ID

// 不透明句柄（DLL 内部维护，我们只持有指针）
struct llama_vocab;
struct llama_model;
struct llama_context;
struct llama_sampler;

// ---------------------------------------------------------------------------
// 采样相关结构体
// ---------------------------------------------------------------------------

// 单个候选 token 的信息
typedef struct llama_token_data {
    llama_token id;   // token 编号
    float       logit; // 对数几率（未归一化）
    float       p;     // 概率
} llama_token_data;

// 候选 token 数组（采样器在其上操作）
typedef struct llama_token_data_array {
    llama_token_data * data;     // 候选数组指针
    size_t             size;     // 候选数量
    int64_t            selected; // 被选中的下标（不是 token id）
    bool               sorted;   // 是否已按概率排序
} llama_token_data_array;

// ---------------------------------------------------------------------------
// llama_batch：一次送入模型的一批 token
// ---------------------------------------------------------------------------
typedef struct llama_batch {
    int32_t          n_tokens;   // 本批 token 数量
    llama_token    * token;      // token id 数组（embd 为 NULL 时使用）
    float          * embd;       // 直接提供 embedding（token 为 NULL 时使用）
    llama_pos      * pos;        // 每个 token 的位置（NULL 则自动递增）
    int32_t        * n_seq_id;   // 每个 token 属于几个序列
    llama_seq_id   ** seq_id;    // 每个 token 的序列 id 列表
    int8_t         * logits;     // 是否输出该 token 的 logits（0=不输出）
} llama_batch;

// ---------------------------------------------------------------------------
// 模型加载参数
// ---------------------------------------------------------------------------
struct llama_model_params {
    void    * devices;                  // ggml_backend_dev_t* 设备列表
    void    * tensor_buft_overrides;    // 张量缓冲类型覆盖
    int32_t   n_gpu_layers;             // 卸载到 GPU 的层数（负数=全部）
    int32_t   split_mode;               // 多 GPU 拆分模式
    int32_t   load_mode;                // 加载方式（mmap/mlock）
    int32_t   main_gpu;                 // 主 GPU 编号
    float   * tensor_split;             // 各 GPU 张量分配比例
    void    * progress_callback;        // 加载进度回调函数指针
    void    * progress_callback_user_data;
    void    * kv_overrides;             // 模型元数据 KV 覆盖
    bool      vocab_only;               // 只加载词表不加载权重
    bool      check_tensors;            // 校验张量数据
    bool      use_extra_bufts;
    bool      no_host;
    bool      no_alloc;
    bool      load_mtp;
};

// ---------------------------------------------------------------------------
// 上下文参数（推理时的运行时配置）
// ---------------------------------------------------------------------------
struct llama_context_params {
    uint32_t n_ctx;                 // 上下文窗口大小（0=用模型默认）
    uint32_t n_batch;               // 逻辑批大小
    uint32_t n_ubatch;              // 物理批大小
    uint32_t n_seq_max;             // 最大序列数
    uint32_t n_rs_seq;
    uint32_t n_outputs_max;
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;             // 生成线程数
    int32_t  n_threads_batch;       // 批处理线程数
    int32_t  ctx_type;
    int32_t  rope_scaling_type;
    int32_t  pooling_type;
    int32_t  attention_type;
    int32_t  flash_attn_type;
    float    rope_freq_base;
    float    rope_freq_scale;
    float    yarn_ext_factor;
    float    yarn_attn_factor;
    float    yarn_beta_fast;
    float    yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float    defrag_thold;
    void   * cb_eval;               // ggml_backend_sched_eval_callback
    void   * cb_eval_user_data;
    int32_t  type_k;                // K 缓存数据类型
    int32_t  type_v;                // V 缓存数据类型
    void   * abort_callback;        // ggml_abort_callback
    void   * abort_callback_data;
    bool     embeddings;
    bool     offload_kqv;           // 将 KQV 计算和 KV 缓存卸载到 GPU
    bool     no_perf;
    bool     op_offload;
    bool     swa_full;
    bool     kv_unified;
    void   * samplers;              // llama_sampler_seq_config*
    size_t   n_samplers;
    void   * ctx_other;             // llama_context*
};

// ---------------------------------------------------------------------------
// 采样链参数
// ---------------------------------------------------------------------------
struct llama_sampler_chain_params {
    bool no_perf;
};

// ---------------------------------------------------------------------------
// 函数指针类型定义
// 每个 typedef 对应 libllama.dll 中一个导出的 C API 函数。
// ---------------------------------------------------------------------------
typedef void                 (*fn_llama_backend_init)();
typedef void                 (*fn_llama_backend_free)();

typedef llama_model_params   (*fn_llama_model_default_params)();
typedef llama_model        * (*fn_llama_model_load_from_file)(const char *, llama_model_params);
typedef void                 (*fn_llama_model_free)(llama_model *);
typedef llama_vocab        * (*fn_llama_model_get_vocab)(const llama_model *);
typedef int32_t              (*fn_llama_model_n_vocab)(const llama_model *);
typedef int32_t              (*fn_llama_model_n_ctx_train)(const llama_model *);
typedef llama_token          (*fn_llama_token_eos)(const llama_model *);
typedef llama_token          (*fn_llama_token_bos)(const llama_model *);

typedef llama_context_params (*fn_llama_context_default_params)();
typedef llama_context      * (*fn_llama_init_from_model)(llama_model *, llama_context_params);
typedef void                 (*fn_llama_free)(llama_context *);

typedef llama_batch          (*fn_llama_batch_init)(int32_t, int32_t, int32_t);
typedef void                 (*fn_llama_batch_free)(llama_batch);
typedef llama_batch          (*fn_llama_batch_get_one)(llama_token *, int32_t);

typedef int32_t              (*fn_llama_decode)(llama_context *, llama_batch);
typedef float              * (*fn_llama_get_logits)(llama_context *);
typedef float              * (*fn_llama_get_logits_ith)(llama_context *, int32_t);

typedef int32_t              (*fn_llama_tokenize)(const llama_vocab *, const char *, int32_t,
                                                    llama_token *, int32_t, bool, bool);
typedef int32_t              (*fn_llama_detokenize)(const llama_vocab *, const llama_token *, int32_t,
                                                      char *, int32_t, bool, bool);
typedef int32_t              (*fn_llama_vocab_n_tokens)(const llama_vocab *);
typedef llama_token          (*fn_llama_vocab_bos)(const llama_vocab *);
typedef llama_token          (*fn_llama_vocab_eos)(const llama_vocab *);
typedef int32_t              (*fn_llama_token_to_piece)(const llama_vocab *, llama_token, char *, int32_t);

typedef llama_sampler_chain_params (*fn_llama_sampler_chain_default_params)();
typedef llama_sampler      * (*fn_llama_sampler_chain_init)(llama_sampler_chain_params);
typedef void                 (*fn_llama_sampler_chain_add)(llama_sampler *, llama_sampler *);
typedef llama_sampler      * (*fn_llama_sampler_init_temp)(float);
typedef llama_sampler      * (*fn_llama_sampler_init_top_p)(float, size_t);
typedef llama_sampler      * (*fn_llama_sampler_init_top_k)(int32_t);
typedef llama_sampler      * (*fn_llama_sampler_init_dist)(uint32_t);
typedef llama_token          (*fn_llama_sampler_sample)(llama_sampler *, llama_context *, int32_t);
typedef void                 (*fn_llama_sampler_accept)(llama_sampler *, llama_token);
typedef void                 (*fn_llama_sampler_free)(llama_sampler *);

typedef const char         * (*fn_llama_print_system_info)();

// ---------------------------------------------------------------------------
// 所有 API 函数指针的集合
// ---------------------------------------------------------------------------
struct LlamaAPI {
    HMODULE dll;

    // backend
    fn_llama_backend_init              backend_init;
    fn_llama_backend_free              backend_free;

    // model
    fn_llama_model_default_params      model_default_params;
    fn_llama_model_load_from_file      model_load_from_file;
    fn_llama_model_free                model_free;
    fn_llama_model_get_vocab           model_get_vocab;
    fn_llama_model_n_vocab             model_n_vocab;
    fn_llama_model_n_ctx_train         model_n_ctx_train;
    fn_llama_token_eos                 token_eos;
    fn_llama_token_bos                 token_bos;

    // context
    fn_llama_context_default_params    context_default_params;
    fn_llama_init_from_model           init_from_model;
    fn_llama_free                      free;

    // batch
    fn_llama_batch_init                batch_init;
    fn_llama_batch_free                batch_free;
    fn_llama_batch_get_one             batch_get_one;

    // decode
    fn_llama_decode                    decode;
    fn_llama_get_logits                get_logits;
    fn_llama_get_logits_ith            get_logits_ith;

    // tokenizer
    fn_llama_tokenize                  tokenize;
    fn_llama_detokenize                detokenize;
    fn_llama_vocab_n_tokens            vocab_n_tokens;
    fn_llama_vocab_bos                 vocab_bos;
    fn_llama_vocab_eos                 vocab_eos;
    fn_llama_token_to_piece            token_to_piece;

    // sampler
    fn_llama_sampler_chain_default_params sampler_chain_default_params;
    fn_llama_sampler_chain_init           sampler_chain_init;
    fn_llama_sampler_chain_add            sampler_chain_add;
    fn_llama_sampler_init_temp            sampler_init_temp;
    fn_llama_sampler_init_top_p           sampler_init_top_p;
    fn_llama_sampler_init_top_k           sampler_init_top_k;
    fn_llama_sampler_init_dist            sampler_init_dist;
    fn_llama_sampler_sample               sampler_sample;
    fn_llama_sampler_accept               sampler_accept;
    fn_llama_sampler_free                 sampler_free;

    // misc
    fn_llama_print_system_info         print_system_info;
};

// 加载 libllama.dll 并解析所有函数指针
// 成功返回 true，失败返回 false
bool load_llama_api(LlamaAPI & api, const char * dll_path);

// 卸载 DLL
void unload_llama_api(LlamaAPI & api);

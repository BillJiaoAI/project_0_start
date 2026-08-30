// main.cpp
// 基于 llama.cpp 的大模型推理 Demo（逐步讲解版）
//
// 本程序演示一个典型的自回归大语言模型（LLM）推理的完整流程：
//   1. 加载模型权重（GGUF 格式）
//   2. 构建推理上下文（Context）
//   3. 对输入文本做 Tokenize（文本 → token id 序列）
//   4. Prefill：把 prompt 的所有 token 一次性送入模型
//   5. Decode 循环：每次生成一个 token，直到遇到 EOS 或达到上限
//   6. 采样（Sampling）：从 logits 概率分布中选下一个 token
//   7. Detokenize：token id → 文本片段，流式输出
//
// 运行方式：
//   llama_demo.exe <model_path> [prompt] [n_predict]
//
// 例如使用 Ollama 的 qwen3.5:27b 模型（blob 就是 GGUF 文件）：
//   llama_demo.exe "C:\Users\...\.ollama\models\blobs\sha256-xxx" "你好" 100

#include "llama_demo.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ===========================================================================
// 第一步：动态加载 libllama.dll 并解析所有 C API 函数
// ===========================================================================
bool load_llama_api(LlamaAPI & api, const char * dll_path) {
    api.dll = LoadLibraryA(dll_path);
    if (!api.dll) {
        printf("[ERROR] 无法加载 DLL: %s (错误码: %lu)\n", dll_path, GetLastError());
        return false;
    }

    // 辅助宏：按名字解析函数指针
    // name 是不含 llama_ 前缀的短名，例如 backend_init
    // 展开后：api.backend_init = (fn_llama_backend_init)GetProcAddress(dll, "llama_backend_init")
    #define LOAD(name) \
        api.name = (fn_llama_##name)GetProcAddress(api.dll, "llama_" #name); \
        if (!api.name) { printf("[ERROR] Cannot find function: llama_%s\n", #name); return false; }

    LOAD(backend_init)
    LOAD(backend_free)
    LOAD(model_default_params)
    LOAD(model_load_from_file)
    LOAD(model_free)
    LOAD(model_get_vocab)
    LOAD(model_n_vocab)
    LOAD(model_n_ctx_train)
    LOAD(token_eos)
    LOAD(token_bos)
    LOAD(context_default_params)
    LOAD(init_from_model)
    LOAD(free)
    LOAD(batch_init)
    LOAD(batch_free)
    LOAD(batch_get_one)
    LOAD(decode)
    LOAD(get_logits)
    LOAD(get_logits_ith)
    LOAD(tokenize)
    LOAD(detokenize)
    LOAD(vocab_n_tokens)
    LOAD(vocab_bos)
    LOAD(vocab_eos)
    LOAD(token_to_piece)
    LOAD(sampler_chain_default_params)
    LOAD(sampler_chain_init)
    LOAD(sampler_chain_add)
    LOAD(sampler_init_temp)
    LOAD(sampler_init_top_p)
    LOAD(sampler_init_top_k)
    LOAD(sampler_init_dist)
    LOAD(sampler_sample)
    LOAD(sampler_accept)
    LOAD(sampler_free)
    LOAD(print_system_info)

    #undef LOAD
    return true;
}

void unload_llama_api(LlamaAPI & api) {
    if (api.dll) {
        FreeLibrary(api.dll);
        api.dll = nullptr;
    }
}

// ===========================================================================
// 辅助：把 token id 转为文本并输出（处理 UTF-8 多字节拼接）
// ===========================================================================
static void print_token_piece(const LlamaAPI & api, llama_vocab * vocab,
                              llama_token token, std::string & leftover) {
    char buf[256];
    int32_t n = api.token_to_piece(vocab, token, buf, sizeof(buf));
    if (n <= 0) return;

    // token_to_piece 返回的可能是不完整的 UTF-8 字节序列，
    // 需要把上次剩余的字节和这次拼起来再输出。
    std::string piece(buf, n);
    leftover += piece;

    // 找到最后一个完整的 UTF-8 字符边界
    size_t cut = leftover.size();
    while (cut > 0) {
        unsigned char c = (unsigned char)leftover[cut - 1];
        if (c < 0x80) break;                    // ASCII
        if ((c & 0xC0) == 0xC0) break;          // 多字节起始
        cut--;                                  // 否则是续字节，回退
    }
    if (cut > 0) {
        printf("%s", leftover.substr(0, cut).c_str());
        fflush(stdout);
        leftover = leftover.substr(cut);
    }
}

// ===========================================================================
// 主推理流程
// ===========================================================================
int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("用法: %s <model_path> [prompt] [n_predict]\n", argv[0]);
        printf("  model_path : GGUF 模型文件路径\n");
        printf("  prompt     : 输入提示（默认: 你好，请介绍一下你自己。）\n");
        printf("  n_predict  : 最大生成 token 数（默认: 128）\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * prompt     = argc > 2 ? argv[2] : "你好，请介绍一下你自己。";
    int          n_predict  = argc > 3 ? atoi(argv[3]) : 128;

    // Ollama 的 libllama.dll 路径
    const char * dll_path = "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Ollama\\lib\\ollama\\libllama.dll";

    printf("=== llama.cpp 推理 Demo ===\n\n");

    // -----------------------------------------------------------------------
    // Step 1: 加载 DLL 并初始化后端
    // -----------------------------------------------------------------------
    printf("[Step 1] 加载 libllama.dll ...\n");
    LlamaAPI api;
    if (!load_llama_api(api, dll_path)) return 1;

    printf("[Step 1] 初始化 llama 后端 ...\n");
    api.backend_init();
    printf("  系统信息: %s\n\n", api.print_system_info());

    // -----------------------------------------------------------------------
    // Step 2: 加载模型
    // -----------------------------------------------------------------------
    printf("[Step 2] 加载模型: %s\n", model_path);

    llama_model_params mparams = api.model_default_params();
    mparams.n_gpu_layers = 999;   // 尽可能多层卸载到 GPU
    mparams.main_gpu     = 0;

    llama_model * model = api.model_load_from_file(model_path, mparams);
    if (!model) {
        printf("[ERROR] 模型加载失败！\n");
        api.backend_free();
        unload_llama_api(api);
        return 1;
    }

    int32_t n_vocab      = api.model_n_vocab(model);
    int32_t n_ctx_train  = api.model_n_ctx_train(model);
    llama_token token_eos = api.token_eos(model);
    llama_token token_bos = api.token_bos(model);
    printf("  词表大小 n_vocab = %d\n", n_vocab);
    printf("  训练上下文 n_ctx_train = %d\n", n_ctx_train);
    printf("  BOS token = %d, EOS token = %d\n\n", token_bos, token_eos);

    // -----------------------------------------------------------------------
    // Step 3: 创建推理上下文（Context）
    // -----------------------------------------------------------------------
    printf("[Step 3] 创建推理上下文 ...\n");
    llama_context_params cparams = api.context_default_params();
    cparams.n_ctx       = 2048;   // 上下文窗口（prompt + 生成）
    cparams.n_threads   = 8;      // CPU 线程数
    cparams.offload_kqv = true;   // KV 缓存卸载到 GPU

    llama_context * ctx = api.init_from_model(model, cparams);
    if (!ctx) {
        printf("[ERROR] 上下文创建失败！\n");
        api.model_free(model);
        api.backend_free();
        unload_llama_api(api);
        return 1;
    }
    printf("  上下文创建成功 (n_ctx=%u)\n\n", cparams.n_ctx);

    // -----------------------------------------------------------------------
    // Step 4: Tokenize —— 把文本转为 token id 序列
    // -----------------------------------------------------------------------
    printf("[Step 4] Tokenize 输入文本 ...\n");
    printf("  原文: \"%s\"\n", prompt);

    llama_vocab * vocab = api.model_get_vocab(model);

    // 第一次调用：传 n_tokens_max=0，获取所需 token 数
    int32_t n_tokens = api.tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                    nullptr, 0, true, true);
    if (n_tokens <= 0) {
        printf("[ERROR] Tokenize 失败，返回 %d\n", n_tokens);
        api.free(ctx);
        api.model_free(model);
        api.backend_free();
        unload_llama_api(api);
        return 1;
    }

    std::vector<llama_token> tokens(n_tokens);
    // 第二次调用：实际写入 token
    api.tokenize(vocab, prompt, (int32_t)strlen(prompt),
                 tokens.data(), n_tokens, true, true);

    printf("  Token 数量: %d\n", n_tokens);
    printf("  Token IDs: ");
    for (int i = 0; i < n_tokens; i++) printf("%d ", tokens[i]);
    printf("\n\n");

    // -----------------------------------------------------------------------
    // Step 5: 构建采样器链（Sampler Chain）
    // -----------------------------------------------------------------------
    // 采样器链按顺序对 logits 进行处理：
    //   temp   → 温度缩放（控制随机性）
    //   top_k  → 只保留概率最高的 k 个候选
    //   top_p  → 核采样（累积概率阈值）
    //   dist   → 最终按概率分布随机抽取
    printf("[Step 5] 构建采样器链 ...\n");
    auto sparams = api.sampler_chain_default_params();
    llama_sampler * sampler = api.sampler_chain_init(sparams);

    api.sampler_chain_add(sampler, api.sampler_init_temp(0.7f));   // 温度
    api.sampler_chain_add(sampler, api.sampler_init_top_k(40));     // Top-K
    api.sampler_chain_add(sampler, api.sampler_init_top_p(0.9f, 1));// Top-P
    api.sampler_chain_add(sampler, api.sampler_init_dist(1234));    // 随机采样
    printf("  temp=0.7, top_k=40, top_p=0.9, seed=1234\n\n");

    // -----------------------------------------------------------------------
    // Step 6: Prefill —— 把 prompt 的所有 token 一次性送入模型
    // -----------------------------------------------------------------------
    printf("[Step 6] Prefill（处理 prompt）...\n");
    llama_batch batch = api.batch_get_one(tokens.data(), n_tokens);
    int ret = api.decode(ctx, batch);
    if (ret != 0) {
        printf("[ERROR] Prefill decode 失败，返回 %d\n", ret);
        api.sampler_free(sampler);
        api.batch_free(batch);
        api.free(ctx);
        api.model_free(model);
        api.backend_free();
        unload_llama_api(api);
        return 1;
    }
    api.batch_free(batch);
    printf("  Prefill 完成，开始生成 ...\n\n");

    // -----------------------------------------------------------------------
    // Step 7: 生成循环（Decode Loop）—— 自回归生成
    // -----------------------------------------------------------------------
    printf("[Step 7] 生成结果:\n");
    printf("---\n");

    std::string leftover;  // 用于拼接不完整的 UTF-8 字节
    int n_generated = 0;

    for (int i = 0; i < n_predict; i++) {
        // 7a. 采样：从当前 logits 中选出下一个 token
        llama_token new_token = api.sampler_sample(sampler, ctx, -1);

        // 7b. 通知采样器接受了这个 token（用于重复惩罚等）
        api.sampler_accept(sampler, new_token);

        // 7c. 检查是否到达 EOS（结束符）
        if (new_token == token_eos) {
            printf("\n[达到 EOS，生成结束]\n");
            break;
        }

        // 7d. 把 token 转为文本并输出（流式）
        print_token_piece(api, vocab, new_token, leftover);
        n_generated++;

        // 7e. 把刚生成的 token 送入模型，更新 KV 缓存，为下一步做准备
        llama_batch one = api.batch_get_one(&new_token, 1);
        ret = api.decode(ctx, one);
        api.batch_free(one);
        if (ret != 0) {
            printf("\n[ERROR] Decode 失败，返回 %d\n", ret);
            break;
        }
    }

    // 输出剩余的 UTF-8 字节
    if (!leftover.empty()) {
        printf("%s", leftover.c_str());
    }
    printf("\n---\n");
    printf("  共生成 %d 个 token\n\n", n_generated);

    // -----------------------------------------------------------------------
    // Step 8: 释放资源
    // -----------------------------------------------------------------------
    printf("[Step 8] 释放资源 ...\n");
    api.sampler_free(sampler);
    api.free(ctx);
    api.model_free(model);
    api.backend_free();
    unload_llama_api(api);

    printf("  完成。\n");
    return 0;
}

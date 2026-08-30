// ============================================================================
// llama.cpp 推理 Demo（v2 - 基于 llama.cpp 源码静态链接）
//
// 本 demo 演示一个典型大语言模型(LLM)的完整推理流程：
//   1. 加载后端(CPU/CUDA)  2. 加载模型(GGUF)  3. 分词(tokenize)
//   4. 创建推理上下文      5. 创建采样器       6. 预填充(prefill)
//   7. 自回归解码循环(采样→输出→再解码)        8. 资源释放
//
// 编译方式: 见同目录 build.bat
// ============================================================================

#include "llama.h"   // llama.cpp 官方 C API 头文件
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    // 让 printf 的数字格式用 C locale（避免某些 locale 下小数点变逗号）
    std::setlocale(LC_NUMERIC, "C");

    // ---------- 命令行参数解析 ----------
    std::string model_path;                       // GGUF 模型文件路径
    std::string prompt = "你好，请介绍一下你自己。"; // 提示词
    int n_gpu_layers = 99;                        // 卸载到 GPU 的层数(99=全部)
    int n_predict    = 128;                       // 最多生成的 token 数
    float temp       = 0.7f;                      // 采样温度
    int   top_k      = 40;                        // top-k 采样
    float top_p      = 0.9f;                      // top-p (nucleus) 采样

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else {
            prompt = argv[i];
            for (int j = i + 1; j < argc; j++) { prompt += " "; prompt += argv[j]; }
            break;
        }
    }
    if (model_path.empty()) {
        printf("Usage: %s -m model.gguf [-n N] [-ngl N] [-t T] [prompt]\n", argv[0]);
        return 1;
    }

    // ========================================================================
    // 步骤 1：加载所有动态后端（CPU / CUDA / Metal 等）
    // ------------------------------------------------------------------------
    // ggml_backend_load_all() 会扫描并注册所有可用的计算后端。
    // llama.cpp 的"新后端架构"核心就是 ggml_backend：
    //   - ggml-cpu   : CPU 后端（含 AVX2/AVX512 等指令集优化）
    //   - ggml-cuda  : NVIDIA GPU 后端（CUDA kernels）
    //   - ggml-metal : Apple GPU 后端
    //   - ggml-vulkan, ggml-sycl, ggml-opencl ...
    // 每个后端实现同一套 ggml 算子接口，模型会按层分配到不同后端上执行。
    // ========================================================================
    printf("[1/8] Loading all backends (CPU/CUDA/...) ...\n");
    ggml_backend_load_all();

    // ========================================================================
    // 步骤 2：加载模型（GGUF 格式）
    // ------------------------------------------------------------------------
    // GGUF 是 llama.cpp 定义的模型存储格式，包含权重张量 + 元数据(架构/词表等)。
    // llama_model_load_from_file 会：
    //   - 解析 GGUF 文件头
    //   - 根据 n_gpu_layers 把部分/全部层的张量分配到指定后端(GPU/CPU)
    //   - 加载分词器(vocab)
    // ========================================================================
    printf("[2/8] Loading model: %s\n", model_path.c_str());
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;  // 99 = 尽量把所有层都放到 GPU

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "[ERROR] Failed to load model\n");
        return 1;
    }
    // 获取词表对象（分词/反分词都需要它）
    const llama_vocab * vocab = llama_model_get_vocab(model);
    printf("      Model loaded. n_vocab=%d, n_ctx_train=%d\n",
           llama_vocab_n_tokens(vocab), llama_model_n_ctx_train(model));

    // ========================================================================
    // 步骤 3：分词（Tokenize）
    // ------------------------------------------------------------------------
    // 文本 → token id 序列。
    // 例如 "你好" 可能被切成 ["你","好"] 两个 token，对应 id 比如 [321, 543]。
    // llama_tokenize 用法：
    //   - 传 tokens=NULL 时返回所需的 token 数量（负值表示需要 N 个）
    //   - 传 tokens=buf 时实际填充，返回填充数量
    // 参数 add_special=true  会自动加 BOS(开头) 等特殊 token
    //      parse_special=true 允许把 <|im_start|> 等特殊标记解析成特殊 token
    // ========================================================================
    printf("[3/8] Tokenizing prompt: \"%s\"\n", prompt.c_str());

    // 第一次调用：获取需要多少个 token
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                   NULL, 0, /*add_special=*/true, /*parse_special=*/true);
    printf("      Prompt tokens: %d\n", n_prompt);

    // 第二次调用：实际填充 token
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                   prompt_tokens.data(), n_prompt, true, true);

    // ========================================================================
    // 步骤 4：创建推理上下文（Context）
    // ------------------------------------------------------------------------
    // llama_context 是一次推理会话的状态容器，包含：
    //   - KV cache（键值缓存，存储已计算 token 的注意力 K/V 张量）
    //   - 计算图（graph）与后端缓冲区
    //   - 线程池等
    // n_ctx   : 上下文窗口大小（prompt + 生成的 token 总数上限）
    // n_batch : 一次 llama_decode 最多处理多少 token（prefill 时用到）
    // n_threads: CPU 推理线程数
    // ========================================================================
    printf("[4/8] Creating context (n_ctx=%d, n_threads=8) ...\n", n_prompt + n_predict);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_prompt + n_predict;  // 上下文窗口
    cparams.n_batch   = n_prompt;              // prefill 批大小
    cparams.n_threads = 8;                     // CPU 线程数
    cparams.no_perf   = false;                 // 开启性能计数

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[ERROR] Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // ========================================================================
    // 步骤 5：创建采样器链（Sampler Chain）
    // ------------------------------------------------------------------------
    // 模型输出的是下一个 token 的 logits（未归一化分数）。
    // 采样器链按顺序对 logits 做变换，最后从概率分布中抽一个 token：
    //   1. top_k  : 只保留概率最高的 k 个 token
    //   2. top_p  : 保留累计概率达到 p 的最小集合（nucleus sampling）
    //   3. temp   : 温度缩放（temp 越小越确定，越大越随机）
    //   4. dist   : 从最终分类分布中按概率采样
    // ========================================================================
    printf("[5/8] Creating sampler chain (top_k=%d, top_p=%.2f, temp=%.2f) ...\n",
           top_k, top_p, temp);
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp)); // \(logit' = logit / temp\)
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));  // 固定种子可复现
/*notes
// LLM单步生成完整链路：token输入 → decode得到logits → sampler采样 → token_id → detokenize
//
// 【名词速记】
// tokenize   : 文本字符串 → token_id数组（喂给模型）
// detokenize : 输出token_id → 人类可读字符串
// logits     : 模型输出原始分数，大小[n_vocab]，可正/负/-INF，不是概率
// softmax    : logits → 概率分布；exp转非负 + 归一化总和=1
// 轮盘赌采样 : 根据概率做加权随机抽取token id
//
// 📌采样器链执行顺序【添加顺序 = 执行顺序，不可乱序】
//    1. top_k : 按数量截断，只保留分数最高K个token，其余logits置‑INF
//    2. top_p : 核采样；内部softmax算概率，累加概率到top_p阈值，淘汰剩余token(logits置‑INF)
//    3. temp  : logits = logits / temp；玻尔兹曼温度缩放，只修改logits，不做exp
//    4. dist  : 【最后一步！真正产出token_id】
//         └─① safe‑softmax：对经过前面处理的logits重新计算概率
//         └─② 轮盘赌(roulette‑wheel)加权随机采样，输出1个token_id
//
// 🔔关键知识点
// 1. top_k：按候选数量过滤；top_p：按累计概率过滤，二者不是重复，经常联合使用
// 2. temp本身只是确定性除法；随机性来自dist内部的随机数，temp只改变概率分布形态
// 3. softmax不能省：轮盘赌要求输入非负、归一化概率；原始logits有负数/-INF，不能直接当权重
// 4. top_p内部会跑一次softmax，但之后会经过temp缩放，旧概率失效，dist必须重新softmax
// 5. sampler全部运行在CPU；decode(Transformer计算)跑在GPU，logits会拷贝回CPU做采样后处理
// 6. 贪心生成(greedy)：直接argmax(logits)，不需要softmax、不需要dist采样器
*/


    // ========================================================================
    // 步骤 6：预填充（Prefill）—— 一次性处理整个 prompt
    // ------------------------------------------------------------------------
    // llama_batch 是传给 llama_decode 的输入批次：
    //   - token[] : 要处理的 token id
    //   - pos[]   : 每个 token 在序列中的位置（用于 RoPE 位置编码）
    //   - logits[]: 哪些位置需要输出 logits（通常只需最后一个）
    // llama_batch_get_one 是便捷构造函数：创建单序列 batch。
    // 这次 decode 会把所有 prompt token 的 K/V 写入 KV cache。
    // ========================================================================
    printf("[6/8] Prefill (processing %d prompt tokens) ...\n", n_prompt);
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "[ERROR] Prefill decode failed\n");
        return 1;
    }

    // ========================================================================
    // 步骤 7：自回归解码循环（Autoregressive Decode）
    // ------------------------------------------------------------------------
    // 大模型生成文本的核心循环：
    //   1. 从 logits 采样出下一个 token
    //   2. 检查是否是结束符(EOG)
    //   3. 把 token 转成文本片段并打印
    //   4. 把这个 token 再喂给模型，得到下一个 token 的 logits
    //   5. 重复，直到遇到 EOG 或达到 n_predict 上限
    //
    // 这就是"自回归"(autoregressive)：每一步的输出是下一步的输入。
    // ========================================================================
    printf("[7/8] Generating ...\n\n");
    fflush(stdout);

    llama_token new_token;
    int n_decoded = 0;

    while (n_decoded < n_predict) {
        // 7a. 采样下一个 token（-1 表示取最后一个位置的 logits）
        new_token = llama_sampler_sample(smpl, ctx, -1);

        // 7b. 检查是否结束生成（EOS / EOT 等）
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // 7c. 反分词：token id → 文本片段（UTF-8 字节）
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (n > 0) {
            printf("%s", std::string(buf, n).c_str());
            fflush(stdout);
        }

        // 7d. 准备下一个 batch：只有一个 token（刚采样出来的）
        batch = llama_batch_get_one(&new_token, 1);

        // 7e. 解码：模型根据新 token + KV cache 生成下一个 logits
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\n[ERROR] Decode failed\n");
            break;
        }
        n_decoded++;
    }
    printf("\n\n");

    // ========================================================================
    // 步骤 8：打印性能统计 + 释放资源
    // ========================================================================
    printf("[8/8] Performance stats:\n");
    llama_perf_context_print(ctx);

    // 释放顺序：sampler → context → model（与创建顺序相反）
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    printf("\nDone. Generated %d tokens.\n", n_decoded);
    return 0;
}

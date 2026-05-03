---
title: DeepSeek-V4 深度解读：迈向高效百万 Token 上下文智能
date: 2026-05-03
tags:
  - LLM
  - DeepSeek
  - MoE
  - 长上下文
  - 注意力机制
categories:
  - 论文阅读
---

## 引言

2026 年 4 月，DeepSeek 团队发布了 DeepSeek-V4 系列预览版的技术报告，目标直指当前大语言模型最棘手的瓶颈之一：**超长上下文的计算效率**。

推理模型（reasoning models）的兴起带来了 test-time scaling 的新范式，但 vanilla attention 的 O(n²) 复杂度成为通往更长上下文、更深推理链路的核心阻碍。Agent 工作流、跨文档分析、长程任务这些前沿场景，都极度依赖高效的超长上下文支持。

DeepSeek-V4 的回答是：通过架构层面的彻底重构，让百万 token 上下文从"勉强能跑"变成"可日常生产部署"。本文系统梳理论文中的关键设计与技术细节。

## 模型概览

DeepSeek-V4 系列包含两个 MoE 模型：

| 模型 | 总参数 | 激活参数 | Transformer 层数 | 隐藏维度 | 上下文长度 |
|------|--------|----------|------------------|----------|-----------|
| **DeepSeek-V4-Pro** | 1.6T | 49B | 61 | 7168 | 1M tokens |
| **DeepSeek-V4-Flash** | 284B | 13B | 43 | 4096 | 1M tokens |

效率方面（vs DeepSeek-V3.2，1M token 场景）：

- **V4-Pro**：单 token 推理 FLOPs 仅为 V3.2 的 **27%**，KV cache 仅为 **10%**
- **V4-Flash**：FLOPs 仅为 V3.2 的 **10%**，KV cache 仅为 **7%**
- 对比经典 BF16 GQA-8（head dim=128）基线：V4 系列的 KV cache 大约只有其 **2%**

## 核心架构创新

### 一、混合注意力：CSA + HCA

百万 token 场景下，attention 是绝对的计算瓶颈。V4 设计了两种互补的高效注意力——**Compressed Sparse Attention（CSA）** 和 **Heavily Compressed Attention（HCA）**——并交替堆叠形成混合架构。

#### CSA：温和压缩 + 稀疏选择

CSA 是"先压缩、再稀疏"的两阶段方案。

**Step 1 — Token 级 KV 压缩（重叠版）**

设输入 H ∈ ℝ^(n×d)，CSA 先生成两组 KV 条目 C^a, C^b 和压缩权重 Z^a, Z^b：

$$
C^a = H \cdot W^{aKV},\quad C^b = H \cdot W^{bKV}
$$

每 m 个 token 压缩为 1 个条目，但**第 i 个压缩块由 2m 个原 KV 条目得到**——使用 C^a 当前窗口的 m 个 + C^b 前一窗口的 m 个，加权求和：

$$
C_i^{\text{Comp}} = \sum_{j=mi}^{m(i+1)-1} S_j^a \odot C_j^a + \sum_{j=m(i-1)}^{mi-1} S_j^b \odot C_j^b
$$

权重 S^a, S^b 由 Z^a, Z^b 加上可学习位置偏置 B^a, B^b 后通过 softmax 归一化得到。这种**重叠压缩**让相邻压缩块共享信息，避免了硬切分造成的边界信息丢失。最终序列长度缩减到 1/m。

**Step 2 — Lightning Indexer 稀疏选择**

压缩后还有 n/m 个条目，对百万 token 来说仍然太多。CSA 用 indexer 进一步做 top-k 选择：

- Indexer query 通过低秩生成（先 down-project 到 d_c，再 up-project）
- 与共享的 query 压缩向量 c_t^Q 共用 latent
- Index score 计算用 ReLU（而非 softmax）：

$$
I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \text{ReLU}\left(q_{t,h}^I \cdot K_s^{\text{IComp}}\right)
$$

- 用 top-k 选择器只保留 k 个压缩 KV 条目（V4-Pro: k=1024，V4-Flash: k=512）

**Step 3 — Shared KV MQA**

选出的 k 个压缩条目同时作为 key 和 value，多个 query head 共享一组 KV，做核心 attention。Query 与 indexer query 共享低秩压缩向量 c_t^Q。

**Step 4 — Grouped Output Projection**

由于 c × n_h 维度很大，直接投影到 d 维太贵。V4 把 n_h 个输出分成 g 组，每组先投影到较低维度 d_g，再拼接投影到 d。

#### HCA：极致压缩，密集 attention

HCA 用更激进的策略：

- 压缩率 m' >> m（V4 中 m=4，**m'=128**），是 CSA 的 32 倍
- **不做重叠压缩**，每 m' 个原始 KV 简单加权压缩为 1 个：

$$
C_i^{\text{Comp}} = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j
$$

- 压缩后条目极少，**直接做密集 attention**，不需要 top-k 稀疏选择
- 同样使用 Shared KV MQA + Grouped Output Projection

**CSA vs HCA 对比**：

| 维度 | CSA | HCA |
|------|-----|-----|
| 压缩率 | 1/4（温和） | 1/128（激进） |
| 重叠压缩 | 有（2m 参与） | 无 |
| 稀疏选择 | top-k | 无（密集） |
| 信息保留 | 更精细 | 更粗糙但更高效 |
| 每 token KV 体积 | 较大 | 极小 |

两者交替堆叠，形成"细看一眼、粗看一眼"的层次化结构——这是 V4 长上下文效率的核心来源。

#### 关键辅助技巧

**1. Sliding Window Attention 分支**

CSA 和 HCA 的因果约束让 query 无法访问当前压缩块内其他 token 的细粒度信息。V4 给每个 query 额外提供 n_win=128 个最近的**未压缩** KV 条目，弥补块内局部依赖。

**2. Partial RoPE（部分旋转位置编码）**

仅对 query 和 KV entry 的**最后 64 维**施加 RoPE。由于 KV entry 同时作为 key 和 value，naive 做法会让 attention 输出携带绝对位置信息。解决方案：对 attention 输出 o_{t,i} 也施加位置为 -i 的 RoPE，使最终输出携带相对位置信息——KV 条目对输出的贡献只与其与 query 的距离有关。

**3. Query/KV RMSNorm**

核心 attention 前对 query 每个 head 和压缩 KV 条目做 RMSNorm，防止 attention logits 爆炸。这也让 V4 的 Muon 优化器**无需 QK-Clip**。

**4. Attention Sink**

可学习 sink logit z'_h 加入 softmax 分母：

$$
s_{h,i,j} = \frac{\exp(z_{h,i,j})}{\sum_k \exp(z_{h,i,k}) + \exp(z'_h)}
$$

让每个 head 的 attention 权重总和**可以小于 1**，模型能选择性"不关注"任何 token。

#### 精度优化

- **KV 存储**：RoPE 维度 BF16，其余 FP8 → KV cache 体积减半
- **Lightning Indexer 计算**：FP4 精度
- **Routed expert 参数**：FP4 精度
- **Index scores**：从 FP32 量化到 BF16，top-k 选择器 2× 加速，KV recall 仍达 99.7%

### 二、流形约束超连接（mHC）

mHC 强化传统残差连接，相比 naive Hyper-Connections 具有更好的数值稳定性。

**标准 HC 的问题**：HC 把残差流的宽度从 ℝ^d 扩展到 ℝ^(n_hc × d)，残差更新形式为：

$$
X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_l X_l)
$$

但实际训练中，多层堆叠时频繁出现数值不稳定，难以 scale。

**mHC 的核心创新**：把残差映射 B_l **约束到双随机矩阵流形**（Birkhoff polytope）M：

$$
B_l \in \mathcal{M} := \{ M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^T M = \mathbf{1}_n^T,\ M \geq 0 \}
$$

**关键性质**：

- 谱范数 ‖B_l‖₂ ≤ 1，残差变换非扩张 → 前后向都数值稳定
- 流形 M 在矩阵乘法下封闭 → 深层堆叠依然稳定

**实现细节**：

- A_l, C_l 通过 Sigmoid 约束为非负有界
- B_l 通过 **Sinkhorn-Knopp 迭代**（20 次）投影到双随机矩阵：交替做行归一化和列归一化
- 三个映射均用 **动态 + 静态分解** 的参数化方式：动态部分由输入 X 经 RMSNorm 后线性变换得到；静态部分是可学习偏置；前面有可学习门控因子

V4 中 mHC 扩展因子 n_hc = 4。配合精心设计的融合内核与重计算策略，整体 wall-time 开销仅占 1F1B pipeline 阶段的 **6.7%**。

### 三、Muon 优化器

Muon 替代 AdamW 作为主优化器，带来更快收敛和更好的训练稳定性：

- **大部分参数**：Muon（动量 0.95，weight decay 0.1，update RMS rescale 0.18）
- **Embedding / 预测头 / RMSNorm / mHC 静态偏置和门控因子**：保留 AdamW

**Hybrid Newton-Schulz 迭代**：用于近似正交化。共 10 步两阶段：

- 前 8 步：系数 (a, b, c) = (3.4445, -4.7750, 2.0315)，激进系数快速收敛
- 后 2 步：系数 (2, -1.5, 0.5)，把奇异值精确稳定到 1

**工程优化**：

- 利用 ZeRO 分桶 + knapsack 算法平衡负载
- MoE 参数：所有 expert 的 down/up/gate 投影矩阵 flatten 后均分到所有 rank，避免拆分逻辑独立矩阵
- Newton-Schulz 在 BF16 下稳定 → MoE 梯度同步用 BF16 + stochastic rounding，通信量减半
- 用 all-to-all + 本地 FP32 求和，规避低精度累加误差

## 基础设施亮点

### 1. MoE 单融合内核（MegaMoE）

把 MoE 层分解为 4 个阶段：Dispatch、Linear-1、Linear-2、Combine（其中 Dispatch 和 Combine 是通信瓶颈）。V4 把它们融合到**单一流水线 kernel** 中：

- **Wave 调度**：把 expert 分成多个 wave，每个 wave 完成通信即可立即开始计算，不等其他 expert
- **稳态时**：当前 wave 计算 + 下个 wave token 传输 + 已完成 expert 结果回传 **三者并发**
- 相比强非融合基线，通用推理工作负载 **1.50~1.73× 加速**；RL rollout 等延迟敏感场景最高 **1.96×**
- 已开源为 DeepGEMM 的 MegaMoE 组件

论文还给出一个有趣的硬件设计建议——通信-计算比临界值：当 C/B ≤ V_comp/V_comm 时通信可被完全隐藏。对 V4-Pro 而言，每 GBps 互联带宽足够隐藏 6.1 TFLOP/s 的计算，超过这个比例继续堆带宽收益递减。

### 2. TileLang DSL

V4 用 TileLang（领域特定语言）替代了大量原本要用上百个 Torch ATen 算子写的细粒度操作：

- **Host Codegen**：在 IR 层共生成设备 kernel + 轻量 host launcher，把 Python 端的运行时检查降到 **<1 微秒/次**（原本几十~几百微秒）
- **Z3 SMT-Solver 辅助形式化整数分析**：用于布局推断、内存冲突检测、边界分析；编译时间开销仅几秒
- **数值精度可控**：默认禁用 fast-math，可显式选择 IEEE-754 严格语义；支持与手写 CUDA 基线 bitwise 对齐

### 3. 批不变 + 确定性内核库

V4 实现了端到端 bitwise 批不变与确定性内核，跨 pre-training / post-training / inference 一致：

- **Attention 批不变**：双 kernel 策略——主 kernel 单 SM 算整序列保证吞吐，辅助 kernel 多 SM + 分布式 shared memory 处理尾部 wave，两者精心设计累加顺序保持 bitwise 一致
- **GEMM 批不变**：用 DeepGEMM 替代 cuBLAS；放弃 split-k，靠优化补回性能
- **确定性反向**：
  - Attention backward 用每 SM 独立累加 buffer + 全局确定性求和
  - MoE backward 用 token order 预处理 + buffer 隔离
  - mHC 中 24 维输出 GEMM 用确定性 split-k 归约

### 4. FP4 量化感知训练（QAT）

应用 MXFP4 到两个组件：

- **MoE expert 权重**：GPU 显存大头
- **CSA 中 indexer 的 QK 路径**：QK 激活在 FP4 缓存、加载、相乘

技巧亮点——**FP4→FP8 反量化无损**：FP8 (E4M3) 比 FP4 (E2M1) 多 2 位指数，动态范围足够吸收 FP4 子块（1×32 tile）相对 FP8 块（128×128 tile）的精细 scale 信息。这让 QAT 完全复用现有 FP8 训练框架，等价于在量化操作上施加 STE（Straight-Through Estimator）。

### 5. 长上下文 Contextual Parallelism

传统 Context Parallelism（CP）在压缩 attention 下失效——压缩需要 m 个连续 KV，可能跨 rank 边界。V4 设计了**两阶段通信**：

1. 第一阶段：每个 rank i 把最后 m 个未压缩 KV 发给 rank i+1；rank i+1 用收到的 + 本地的 s 个一起压缩
2. 第二阶段：all-gather 收集所有压缩 KV；fused select-and-pad 算子重组为完整序列

### 6. 推理框架：异构 KV Cache + 磁盘存储

混合 attention 让 KV cache 异构（CSA / HCA / SWA / 未压缩状态各不同）。V4 设计了定制 KV cache 布局，其中：

- **State Cache**：固定大小池，存 SWA 和压缩分支的"未达压缩条件"的尾部 token
- **Classical KV Cache**：每 cache block 覆盖 lcm(m, m') 个原 token，产出 k1=lcm(m,m')/m 个 CSA 压缩条目和 k2=lcm(m,m')/m' 个 HCA 压缩条目

**On-Disk KV Cache** 实现共享前缀的高效复用。SWA KV 体积约是压缩 KV 的 8 倍，论文给出三种策略：

- **Full SWA Caching**：存全部，零计算冗余；但 SSD 访问模式不友好
- **Periodic Checkpointing**：每 p 个 token 存一次，按需 trade-off
- **Zero SWA Caching**：完全不存，靠最后 n_win·L 个 token 重计算复原

## 预训练

### 数据

- **总规模**：32T+ tokens
- **重点强化**：
  - 长文档（**科学论文、技术报告**等学术高价值材料）
  - Agentic 数据（mid-training 阶段引入，增强代码 Agent 能力）
  - 多语言数据（提升不同文化的长尾知识覆盖）
- **网页数据过滤**：移除批量自动生成和模板化内容，防止 model collapse
- 跨源文档打包减少截断；预训练采用 **sample-level attention masking**

### 渐进训练

序列长度阶梯式扩展：**4K → 16K → 64K → 1M**

稀疏 attention 两阶段引入：

1. **前 1T tokens** 用密集 attention 热身
2. 在 64K 序列长度时引入稀疏 attention：先短暂热身 lightning indexer（让 indexer 学会选择），再全面切换到稀疏 attention

### 训练超参

| | V4-Flash | V4-Pro |
|--|----------|--------|
| 训练 tokens | 32T | 33T |
| 最大 batch size | 75.5M tokens | 94.4M tokens |
| 峰值学习率 | 2.7×10⁻⁴ | 2.0×10⁻⁴ |
| 末期学习率 | 2.7×10⁻⁵ | 2.0×10⁻⁵ |
| MTP loss 权重 | 0.3 → 0.1（衰减期） | 同左 |

学习率线性 warmup 2000 步 → 恒定 → cosine decay。

### 训练稳定性：两个实战技巧

万亿参数 MoE 训练中，loss spike 始终与 MoE 层 outlier 相关，且路由机制会放大 outlier。V4 提出两个实用但理论尚不完全理解的技巧：

**1. Anticipatory Routing（预见性路由）**

解耦 backbone 和 routing 网络的同步更新：第 t 步用当前参数 θ_t 算特征，但路由索引用历史参数 θ_{t-Δt} 计算。

工程实现：

- 在第 t-Δt 步预取第 t 步的数据，"预见性"地预计算并缓存路由索引
- 通过流水线执行 + 与 EP 通信重叠，额外开销控制在 ~20%
- **自动检测机制**：仅当 loss spike 时触发 Anticipatory Routing；稳定后恢复正常训练
- 整体几乎零额外训练开销，不损失性能

**2. SwiGLU Clamping**

- 线性分量 clamp 到 [-10, 10]
- gate 分量上界设为 10
- 有效消除 outlier，不影响性能

### 基础模型评估

| 模型 | 激活参数 | MMLU-Pro | SimpleQA verified | LongBench-V2 |
|------|---------|----------|-------------------|--------------|
| V3.2-Base | 37B | 65.5 | 28.3 | 40.2 |
| V4-Flash-Base | 13B | 68.3 | 30.1 | 44.7 |
| V4-Pro-Base | 49B | **73.5** | **55.2** | **51.5** |

V4-Flash-Base 用更少的激活参数全面超越 V3.2-Base；V4-Pro-Base 设立 DeepSeek 系列基础模型的新天花板。

## 后训练

V4 后训练采用 **两阶段范式**：

### Stage 1：Specialist Training（专家训练）

为每个目标领域（数学、代码、Agent、指令遵循等）独立训练专家模型：

- 先在领域数据上做 **SFT**
- 再用 **GRPO** 做 RL，引入领域奖励信号

**三种推理强度模式**（通过不同的长度惩罚和上下文窗口配置训练得到）：

| 模式 | 特征 | 典型场景 | 响应格式 |
|------|------|---------|---------|
| Non-think | 快速直觉响应 | 日常对话、低风险决策 | `</think> summary` |
| Think High | 显式逻辑分析 | 复杂问题、规划 | `<think> ... </think> summary` |
| Think Max | 最大化推理强度 | 极限挑战 | 特殊 system prompt + thinking |

Think Max 模式的 system prompt 注入：

> Reasoning Effort: Absolute maximum with no shortcuts permitted. You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause...

### 创新点：Generative Reward Model（GRM）

V4 摒弃传统标量 reward model：

- **Actor 网络本身充当 GRM**
- 评估能力（judging）与生成能力**联合优化**
- 模型的内部推理能力直接融入评估过程
- 仅需少量人工标注，模型用自身逻辑泛化到复杂任务

### Stage 2：On-Policy Distillation（OPD）

将多个专家通过 on-policy 蒸馏统一到单一模型，最小化反向 KL loss。这一步替代了 V3.2 的混合 RL 阶段。

### 其他增强

- **新工具调用 schema**：使用 `|DSML|` 特殊 token 和 XML 格式，减少 escaping 失败和工具调用错误
- **Interleaved Thinking**：
  - 工具调用场景：完整保留所有 round 的推理内容（V3.2 在每次新用户消息时都会丢弃，V4 利用 1M 上下文优势保留完整推理链）
  - 普通对话场景：保留原策略，新用户消息时丢弃前 round 推理
- **Quick Instruction**：在输入序列附加专用 special token，复用 KV cache 处理触发 web search、意图识别等辅助任务，避免重复 prefill

## 评估亮点

### 知识

- **SimpleQA / Chinese-SimpleQA**：V4-Pro-Max 显著超越所有开源模型，接近 Gemini-3.1-Pro
- **MMLU-Pro / HLE / GPQA**：领先开源同行；落后 Gemini-3.1-Pro 但差距收窄

### 推理

- 超越 GPT-5.2 和 Gemini-3.0-Pro
- **Codeforces** 排名 23 位（人类候选）
- **Putnam-2025 形式化数学：120/120 满分**
- 与 GPT-5.4 / Gemini-3.1-Pro 仍有 3~6 个月的差距

### Agent

- 公开 benchmark 与 Kimi-K2.6、GLM-5.1 持平，略逊于闭源前沿模型
- **MCPAtlas / Toolathlon** 表现出色——这两个 benchmark 包含大量真实 MCP 服务，证明 V4 不只对内部框架过拟合
- 内部测试中 V4-Pro-Max 超过 Claude Sonnet 4.5，接近 Opus 4.5

### 长上下文

- **MRCR 8-needle**：1M token 仍保持 0.59~0.66，超越 Gemini-3.1-Pro，略逊于 Claude Opus 4.6
- 128K 内检索性能高度稳定
- **CorpusQA** (1M context)：超越 Gemini-3.1-Pro

### 真实任务

**中文写作**（vs Gemini-3.1-Pro）：

- 功能写作综合胜率 **62.7% vs 34.1%**
- 创意写作：指令遵循胜率 60.0%，**写作质量胜率 77.5%**
- 高复杂度场景下 Claude Opus 4.5 仍领先（52.0% vs 45.9%）

**白领任务**（30 项中文专业任务，vs Opus-4.6-Max）：

- 综合非负胜率 **63%**
- Task Completion 和 Content Quality 维度领先
- Instruction Following 和 Formatting Aesthetics 仍有差距

**Search**：

- Agentic search 在 869 道题中胜 RAG 61.7% vs 18.3%
- 成本仅略高于 RAG（output token 数：1526 vs 1308）

## 总结与思考

DeepSeek-V4 的核心贡献可以分三个层次：

**架构层**：CSA + HCA 的"细粒度温和压缩 + 极致稀疏选择"和"粗粒度激进压缩 + 密集 attention"两种策略交替使用，让模型在不同层用不同分辨率"看"长上下文。这是百万 token 不再昂贵的根本原因。

**训练层**：mHC 通过双随机矩阵流形约束保证数值稳定性；Muon + Hybrid Newton-Schulz 加速收敛；Anticipatory Routing 与 SwiGLU Clamping 解决万亿 MoE 的 loss spike 顽疾——这些都是工程上的硬突破。

**基础设施层**：MegaMoE 单融合内核、TileLang DSL、FP4 QAT、批不变确定性内核、磁盘 KV Cache——这些不是 paper 里点缀的"系统工作"，而是让前面的架构创新真正落地为生产级服务的核心。

论文坦承的局限也值得关注：

- 架构相对复杂，保留了大量验证过的组件，未来要做"架构蒸馏"
- Anticipatory Routing 与 SwiGLU Clamping 的理论原理仍未完全理解
- 多模态尚未集成
- 与前沿闭源模型仍有 3~6 个月差距

我个人最感兴趣的是 **CSA 重叠压缩 + HCA 粗粒度压缩** 的层次化设计——它暗示着未来超长上下文模型的通用范式：不是一刀切的稀疏，而是多分辨率、互补粒度的层次化抽取。结合 GRM 这样的"模型即评估器"思路，DeepSeek 已经在为下一阶段（online learning、long-horizon agent）铺路了。

---

**参考资料**：DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*. 2026.

**模型地址**：https://huggingface.co/collections/deepseek-ai/deepseek-v4

**开源参考实现**：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference

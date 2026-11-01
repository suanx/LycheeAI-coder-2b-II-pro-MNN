# LycheeAI-coder-2b-II-pro → MNN Chat 转换工具箱

把魔搭社区的模型 [`whcl412/LycheeAI-coder-2b-II-pro`](https://www.modelscope.cn/models/whcl412/LycheeAI-coder-2b-II-pro)
转换成 **MNN 格式**，放进手机上的 **MNN Chat** 离线运行。

> 模型基座为 MiniCPM5-2B（面壁智能 OpenBMB），Apache 2.0 许可，允许修改、重新训练与再分发。
> 这是一个 2B 的**工具调用（function calling）专用**微调模型，主打本地小助手场景。

---

## ⚠️ 开工前必读的三件事

**1. MNN Chat 只认 `.mnn` 格式。**
它不认 GGUF、不认 safetensors、不认 ONNX。原仓库现成的 GGUF 版本在这条链路上**完全用不上**，必须自己转一次。

**2. 必须下载无损（bf16）权重，不能用 4bit 版本。**
魔搭上 `LycheeAI-coder-2b-II-pro` 这个默认仓库是 **MLX 4bit 量化版（1.43GB）**，有损压缩，
作者明确说明它只能跑推理，**不能作为继续微调或二次转换的基础**。
要加工请用 `LycheeAI-coder-2b-II-pro-f16`（bf16，5.03GB）。

**3. 先拿原版跑通转换，再去做微调。**
MiniCPM5 是较新的架构，MNN 官方仓库目前只公开验证过 MiniCPM**4** 系列。
把"能不能转"这个不确定性放在最前面验证，避免训完之后才发现转不出来。

---

## 硬件与耗时预估

| 环节 | 需求 | 耗时 |
|---|---|---|
| 编译 MNN | Linux/Mac，4 核+，需 cmake + C++ 编译器 | 15–30 分钟 |
| 下载权重 | 5GB 磁盘 | 5–20 分钟 |
| 转 MNN（4bit） | 16GB+ 内存（CPU 即可，无需 GPU） | 20–40 分钟 |
| 产物体积 | 4bit 约 1.4–1.6GB | — |

没有本地机器？用**魔搭免费 Notebook**（阿里云 PAI-DSW）：
8 核 32GB 内存、CPU 环境长期免费，装环境和洗数据都够用；
GPU 环境 24GB 显存约 36 小时额度。注意只有 `/mnt/workspace` 目录持久化，
单实例最长 10 小时、空闲 1 小时自动关机，长任务记得后台跑 + 断点续训。

### 长时间任务：挂上心跳

编译和转换都要跑几十分钟，而 Notebook 空闲 1 小时会自动关机。跑长任务时顺手挂个心跳：

```bash
nohup ./scripts/heartbeat.sh >/dev/null 2>&1 &
tail -f heartbeat.log
```

每 60 秒追加一行，含时间戳、内存、磁盘、各阶段进度、编译/转换进程是否存活。
回来 `tail -20 heartbeat.log` 即可掌握进度；**若实例被自动关停，日志时间戳会出现断层**，一眼可辨——
此时重新拉起实例、`git pull` 后从断点继续即可。

---

## 快速开始（三步）

```bash
git clone https://github.com/suanx/LycheeAI-coder-2b-II-pro-MNN.git
cd LycheeAI-coder-2b-II-pro-MNN

./scripts/build_mnn.sh          # ① 编译 MNN，产出 MNNConvert
./scripts/download_model.sh     # ② 从魔搭下载 bf16 权重
./scripts/convert_to_mnn.sh     # ③ 转 MNN
./scripts/push_to_phone.sh      # ④ 推到手机
```

可选：想先微调再转，在 ② 和 ③ 之间插入 `./scripts/finetune_lora.sh` 与 `./scripts/merge_lora.sh`。

---

## 完整流程

### ① 编译 MNN

`MNNConvert` 是整个链路的必需工具，靠 `-DMNN_BUILD_CONVERTER=ON` 这个开关产出，**漏了就转不了**。

```bash
git clone https://github.com/alibaba/MNN.git
cmake -S MNN -B MNN/build -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
cmake --build MNN/build --parallel "$(nproc)"
```

平台可选加速项：x86 加 `-DMNN_AVX512=true`，Android 加 `-DMNN_OPENCL=true`，iOS 加 `-DMNN_METAL=ON`。

### ② 下载权重

```bash
pip install modelscope -U
python -c "from modelscope import snapshot_download; \
  snapshot_download('whcl412/LycheeAI-coder-2b-II-pro-f16', local_dir='./original')"
```

下完先确认不是空壳：`ls -lh ./original/model.safetensors`，应约 5.0GB。

### ③ 转 MNN

```bash
pip install -r MNN/transformers/llm/export/requirements.txt

python MNN/transformers/llm/export/llmexport.py \
  --path ./original \
  --export mnn \
  --quant_bit 4 --quant_block 64 --hqq \
  --lm_quant_bit 8 \
  --mnnconvert ./MNN/build/MNNConvert \
  --dst_path ./lychee-mnn \
  --test "帮我写个快速排序"
```

参数讲究：

| 参数 | 说明 |
|---|---|
| `--quant_bit 4` | 4bit 量化。手机上必须量化，2B 模型 4bit 后约 1.5GB |
| `--hqq` | 建议加，能明显改善低比特下的精度损失 |
| `--lm_quant_bit 8` | lm_head 层保持 8bit。该模型词表 130560 偏大，这里掉精度很明显 |
| `--quant_block 64` | 量化分组大小，越小精度越好、体积略大 |
| `--test "..."` | 直接输出一段对话结果，用来验证转换没跑歪 |

产物目录必须包含：`llm.mnn`、`llm.mnn.weight`、`tokenizer.txt`、`llm_config.json`、`config.json`
（可能还有 `embeddings_bf16.bin`）。

### ④ 部署到 MNN Chat

```bash
adb push ./lychee-mnn /sdcard/MNN/chat/lychee-mnn/
```

然后在 App 里点 **「添加本地模型」**，选中该目录加载。

也可在 App 内设置：计算后端选 CPU（兼容最好）或 OpenCL（GPU 加速，旗舰机更快）；
精度 Low 更快、High 质量更好；系统提示词里显式声明身份，例如
`你是 LycheeAI-coder-2b-II-pro，由 MiniCPM5-2B 通过 LoRA 微调而来的编程助手。`

### ⑤ （可选）先微调再转换

```bash
# LoRA 微调
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model whcl412/LycheeAI-coder-2b-II-pro-f16 \
  --train_type lora \
  --dataset my_data.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --lora_rank 8 --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 2048 \
  --output_dir output

# 合并回完整模型（建议先合并再转换，少排查一层）
swift export --adapters output/checkpoint-xxx --merge_lora true
```

数据集标准格式（`my_data.jsonl`，每行一条，system 可选）：

```json
{"messages":[{"role":"system","content":"你是 LycheeAI-coder-2b-II-pro..."},{"role":"user","content":"帮我写个快排"},{"role":"assistant","content":"```python\n...\n```"}]}
```

> 作者原配方：LoRA rank 8、scale 20、16 层、lr 5e-5、batch 2 / seq 2048、980 条 × 1 epoch。
> 他的经验是**"精选少量数据 × 1 epoch"远好过"全量数据 × 0.22 epoch"**——
> 行为惯性按样本条数加权，数据集里纯聊天占比过高会把梯度拉偏。

---

## 故障排查

**MiniCPM5 架构不被识别**
先试自动识别；失败则手动指定 `--type llama`（该模型 `architectures` 确实是 `LlamaForCausalLM`）。
关键配置供核对：42 层、hidden 2048、FFN 6144、16 注意力头 / 2 KV 头、词表 130560、RoPE base 5,000,000、上下文 131072。

**对话格式错乱（不停自我介绍 / 复读）**
多半是 chat template 没带过去。该模型的工具调用能力全靠 `chat_template.jinja`，
而 MNN 的 config 里有 `chat_template`、`use_template` 字段，转换后需确认它们正确写入。

**词表大导致内存吃紧**
词表 130560，embedding 单独 bf16 存约 500MB。低端机可用 `--seperate_embed` 分离、
或用 mmap 方式加载（`use_mmap`）。

**直接转 MNN 失败**
回退到两段式：先 `--export onnx` 导出 `llm.onnx`，再手动转换——
```bash
./MNNConvert --modelFile ./onnx/llm.onnx --MNNModel llm.mnn \
  --keepInputFormat --weightQuantBits=4 --weightQuantBlock=128 \
  -f ONNX --transformerFuse=1 --allowCustomOp --saveExternalData
```

**工具调用（function calling）在 MNN Chat 上大概率失效**
MNN Chat 是纯聊天 App，没有工具执行层。模型能保留的只是"输出裸 JSON"这个格式习惯，
实际调用不了任何工具。如果核心诉求是工具调用，需自己基于 MNN-LLM 写一层壳。

---

## 关于把模型权重放进 GitHub

转换产物约 1.5GB，而 GitHub 单文件上限 100MB、免费 LFS 仅 1GB 存储 + 1GB/月带宽——
**GitHub 并不适合托管模型权重**。

本仓库的定位是**转换工具箱**（文档 + 脚本），不含权重。
如需分发成品模型，推荐发布到魔搭社区或 Hugging Face：

```bash
swift export --adapters output/checkpoint-xxx --merge_lora true \
  --push_to_hub true --hub_model_id '你的账号/模型名' --hub_token '你的token'
```

---

## 许可

转换脚本沿用 Apache 2.0。模型权重继承原始模型的 Apache 2.0 许可，
再分发时请保留许可声明与版权声明，并注明你所做的修改。

基座 MiniCPM5-2B 版权归面壁智能（OpenBMB）所有。

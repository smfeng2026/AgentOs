# MNN Pocket · HarmonyOS 端侧对话 + 决策

把 **MNN-Pocket-0.3B（INT4，约 179 MB）** 和 **MNN-Pocket-0.3B-Dec（W8 决策图，约 341 MB）** 跑在 HarmonyOS 设备本地的双模式应用：

- **对话模式**：界面参照微信 / 小艺的聊天布局，输入框固定在底部，模型在设备上流式生成回答，并在每条回答下方给出 **tokens 数、推理毫秒数、tok/s、首字延迟**。
- **决策模式**：非自回归决策模型，输入「场景状态 + 问题」，**单次前向**返回 choice / noul / score 三种题型的结构化答案与校准概率。内置 7 个多场景示例（客服路由、内容审核、Agent 工具选择、满意度评分等），支持表单 / JSON 双模式自定义场景并本地保存。

顶部分段控件一键切换两种模式，会话与决策状态互不丢失。推理全程离线，不联网、不上传：权重随 HAP 一起安装，第一次启动时从 rawfile 拷进应用沙箱，之后由 MNN 的 CPU 后端在本地计算。

> **⚠️ 权重不入库**：`llm.mnn.weight`（163.5 MB）与 `decision-w8-q8-embedding.mnn`（341 MB）超过 GitHub 单文件 100 MB 限制，未随仓库提供。clone 后请先执行 `scripts/fetch_model.sh` 下载模型，详见 **[SETUP.md](SETUP.md)**（下载方式、校验值、构建与运行步骤）。

| 依赖 | 版本 |
| --- | --- |
| DevEco Studio / SDK | 6.0.2（API 22） |
| MNN | 3.6.2（源码编译，LLM + TransformerFuse 打开） |
| 对话模型 | taobao-mnn/MNN-Pocket-0.3B-MNN（qwen3 架构，28 层 / hidden 896 / INT4 W4A16） |
| 决策模型 | alivecat/MNN-Pocket-0.3B-Dec-MNN（decision-w8-q8-embedding，LoRA 已合并，W8 主干 + Q8 词表 + FP32 指针头） |
| ABI | arm64-v8a（HarmonyOS 真机；模拟器需要 x86_64 版 libMNN.so） |

---

## 1. 界面与交互

```
┌──────────────────────────────────────────┐
│ [P] Pocket  0.3B        ● 已就绪   [清空] │  顶部：模型状态、加载耗时、清空会话
├──────────────────────────────────────────┤
│  [P]  你好，我是 Pocket……                 │  AI 气泡（左，白底）
│        输入 56 tok · 输出 84 tok · 上下文…│  指标只在回答下方出现
│        推理 557 ms · 173.4 tok/s · 首字…  │
│                                          │
│                  帮我写一句朋友圈文案 [我] │  用户气泡（右，微信绿）
├──────────────────────────────────────────┤
│  ┌────────────────────────────┐   ( ↑ )   │  底部输入框 + 右侧上传/发送按钮
│  └────────────────────────────┘           │
└──────────────────────────────────────────┘
```

- **发送**：输入文字 → 右侧绿色圆形按钮亮起 → 点击后用户气泡入列、输入框清空、开始流式生成。
- **流式**：首个 token 到达前显示「正在思考…」，之后文字逐字追加，带 `▍` 光标与实时毫秒计时。
- **指标**：回答结束后气泡下方两行小字
  - `输入 56 tok · 输出 84 tok · 上下文 139 tok`
  - `推理 557 ms · 173.4 tok/s · 首字 66 ms`
- **首启**：显示「首次启动准备中 xx%」进度（把 166 MB 权重从 HAP 拷进沙箱），随后「模型加载中」，完成后进入对话。
- **多轮**：保留会话历史（MNN 复用 KV / prompt 前缀），顶部按钮可一键清空。
- **异常**：加载失败有独立错误页 + 重新加载；生成出错用 toast 提示。

### 1.1 模式切换与决策模式（⚖️）

顶部中央的分段控件在 **对话（绿）/ 决策（靛蓝）** 之间切换：单页双视图，聊天历史与决策结果都保留，切换不重新加载已就绪的模型。决策模型首次进入时懒加载（341 MB 拷贝进度 + 加载提示），之后常驻。

决策模式三个视图：

1. **场景库**：7 个内置场景卡片，题型徽章标注 choice / noul / score——
   客服工单路由（官方示例）、退换货分流（中文）、内容审核、Agent 工具选择、满意度评分、线索优先级分级、Email triage（英文）。
2. **运行视图**：场景状态全文 + 问题预览 →「运行决策」→ 单次前向（实时毫秒计时）→ 结果卡片：
   - `choice`：胜者高亮 ✓ + 各候选概率条；
   - `noul`：是/否结论 + P(是) 仪表；
   - `score`：概率加权分 + 各级分布。
   底部指标：`输入 tok · 前向 ms · 编码 ms · T`。
3. **自定义编辑器**：表单模式（state + 问题列表：题型切换、choice 键值对、noul 描述、score 量表）或 JSON 模式（直接编辑官方协议请求），实时校验；保存后进入「我的场景」分组（preferences 持久化），可编辑/删除，内置场景可「编辑副本」。

决策协议要点（与官方 `inference_mnn.py` 逐行对齐，见 `decision_encoder.cpp`）：

- 请求 = `state` + 有序 `questions`；控制 token 转义 `<|x|>` → `<¦x¦>` 防注入；
- 每个问题独立 attention 分支，position 从 state 末尾重新计数；稠密 L×L 掩码由段号生成；
- 图输出 `logits[Q,C]`，图外做温度 softmax（T=1.1112， checkpoint 校准值）；
- **端侧请求上限 2048 tokens**（掩码内存约束，可在 `DecisionViewModel.maxTokens` 调整）；超长报错不截断；
- tokenizer 复用聊天包 `tokenizer.mtok`（同一底座、同一词表），已通过与官方 `tokenizer.json` 的逐 ID 金标对齐。

---

## 2. 工程结构

```
entry/
├── libs/arm64-v8a/                      # 随 HAP 打包的动态库
│   ├── libMNN.so                        #   预编译的 MNN（含 LLM，4.1 MB）
│   └── libc++_shared.so                 #   SDK 同版本 C++ 运行库
├── src/main/cpp/                        # NAPI 模块（libllmbridge.so）
│   ├── CMakeLists.txt
│   ├── llm_engine.h / llm_engine.cpp    #   ChatEngine：加载、流式生成、token/耗时统计
│   ├── decision_json.h/.cpp             #   决策请求/结果 JSON（保插入序）
│   ├── decision_encoder.h/.cpp          #   官方协议移植：render/转义/分支编码/掩码
│   ├── decision_engine.h/.cpp           #   Module 加载 + 单次前向 + 温度 softmax
│   ├── napi_init.cpp                    #   worker 线程 + 任务队列（chat 与 decide 串行）
│   ├── types/libllmbridge/index.d.ts    #   ArkTS 侧类型声明
│   └── third_party/mnn/                 #   MNN 公共头文件 + tokenizer 源码（3.6.2，未导出故随桥编译）
├── src/main/ets/
│   ├── pages/Index.ets                  # 壳：模式切换、聊天状态机、事件路由
│   ├── pages/DecisionPage.ets           # 决策模式：场景库 / 运行 / 编辑器容器
│   ├── view/ModeSwitch.ets              # 顶部分段控件
│   ├── view/MessageBubble.ets           # 气泡 + 流式气泡（含指标行）
│   ├── view/InputBar.ets                # 底部输入框 + 发送按钮
│   ├── view/ScenarioCard.ets            # 场景卡片（题型徽章）
│   ├── view/ResultPanel.ets             # 概率条 / 是否仪表 / 评分分布
│   ├── view/ScenarioEditor.ets          # 自定义场景：表单 + JSON 双模式
│   ├── model/LlmRuntime.ets             # native 模块的 TS 单例封装（chat + decide）
│   ├── model/ModelManager.ets           # rawfile → 沙箱的一次性拷贝（双模型包）
│   ├── model/ChatTypes.ets              # 消息、指标解析、格式化
│   ├── model/DecisionTypes.ets          # 场景/请求构造/结果解析
│   ├── model/DecisionViewModel.ets      # 决策状态机（@Observed 单例）
│   ├── model/ScenarioStore.ets          # 内置场景 + 自定义持久化
│   ├── common/Theme.ets                 # 配色（对话绿 / 决策靛蓝）
│   └── entryability/EntryAbility.ets
└── src/main/resources/rawfile/
    ├── model/                           # 聊天包（llm.mnn / .weight / tokenizer / 配置）
    └── decision/                        # 决策包（w8 图 / tokenizer.mtok / inference_config.json）
scripts/build_mnn_ohos.sh                # 重新交叉编译 MNN 并装进工程
scripts/fetch_model.sh                   # 重新下载两个模型包
tools/engine_cli/                        # macOS 验证工具：chat / decide / encode 三模式
tools/dec_ref/                           # 官方参考实现 + 金标对齐脚本（parity.py）
```

数据流：

```
Index.ets ──LlmRuntime──► libllmbridge.so
                              │  ① 任务入队（init / chat / reset / dec_init / decide）
                              ▼
                       worker 线程（唯一持有 ChatEngine + DecisionEngine）
                              │  ② ChatEngine::chat → Llm::response(流式 ostream)
                              │     DecisionEngine::predict → Module::onForward(单次前向)
                              ▼
                        libMNN.so（CPU）
                              │  ③ delta / done / error · dec_ready / dec_done / dec_error
                              ▼
                 napi_threadsafe_function ──► ArkTS 事件回调（UI 线程）
```

MNN 的 LLM 运行时不是线程安全的，单轮推理也会阻塞数百毫秒到数秒，所以 **native 侧只用一个专用 worker 线程**持有两个引擎，JS 只投递任务、通过一个 threadsafe function 接收事件；UI 线程永远不被推理阻塞。对话生成与决策前向互斥排队，不会并发占用 CPU。

---

## 3. 指标口径（都来自 MNN 自己的 `LlmContext`）

| 展示项 | 来源 | 说明 |
| --- | --- | --- |
| 输入 tokens | `context->prompt_len` | 本轮 prefill 的 token 数（含系统提示与历史） |
| 输出 tokens | `context->output_tokens.size()` | 本轮生成的 token 数 |
| 上下文 tokens | `context->all_seq_len` | KV cache 中当前总长度 |
| 推理 ms | 引擎外部墙钟 | 本轮端到端耗时（含采样） |
| prefill / decode ms | `prefill_us` / `decode_us` | 只用于计算速度，未直接显示 |
| tok/s | `outputTokens / decode_us` | 生成速度 |
| 首字延迟 | 第一次流式回调时刻 | MNN 只多模态才填 `ttfa_us`，这里自己测 |

`ChatStats::toJson()` 会把上述字段一起回传，`parseStats()` 在 ArkTS 侧反序列化。

---

## 4. 构建与运行

### 4.1 用 DevEco Studio（推荐）

1. `File → Open` 打开本工程，等待 Sync。
2. `File → Project Structure → Signing Configs` 勾选 **Automatically generate signature**（需要登录华为开发者账号）。**不做签名生成的 HAP 无法安装**。
3. 连接 HarmonyOS 6.0.2 及以上的 arm64 真机（模拟器需要另行编译 x86_64 版 `libMNN.so`），点击 Run。

首次构建会编译 `libllmbridge.so`（几秒），`libMNN.so` 已预编译好随包携带。

### 4.2 命令行

```bash
export DEVECO_SDK_HOME=/Applications/DevEco-Studio.app/Contents/sdk
export JAVA_HOME=/Applications/DevEco-Studio.app/Contents/jbr/Contents/Home
export PATH=/Applications/DevEco-Studio.app/Contents/tools/node/bin:$JAVA_HOME/bin:$PATH

/Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw \
  --mode module -p product=default assembleHap --no-daemon
```

产物：`entry/build/default/outputs/default/entry-default-unsigned.hap`（约 525 MB，含两个模型包与三个 .so）。未配置签名时只能得到 unsigned 包，需用 DevEco 的自动签名或自行 `hap-sign-tool` 签名后 `hdc install`。

### 4.3 重新编译 MNN（可选）

`libMNN.so` 由下面的脚本交叉编译（约 8 分钟），会同时更新头文件：

```bash
scripts/build_mnn_ohos.sh /path/to/MNN/source
```

### 4.4 下载模型（fresh checkout 必需，详见 [SETUP.md](SETUP.md)）

```bash
scripts/fetch_model.sh                    # 默认走 hf-mirror.com
```

### 4.5 桌面端验证引擎

`tools/engine_cli` 与 app 编译的是**同一份**引擎代码（`llm_engine.cpp` + `decision_*.cpp` + vendored tokenizer），三种模式：

```bash
# 先构建 macOS 版 MNN（只需一次）：
cmake -S .mnn-src/MNN -B .mnn-build/mac -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_TOOLS=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_SHARED_LIBS=ON
cmake --build .mnn-build/mac --target MNN -j 8

# 编译 CLI（工程根目录）：
clang++ -std=c++17 -O2 -Wno-deprecated-declarations \
  -I entry/src/main/cpp -I entry/src/main/cpp/third_party/mnn/include \
  -I entry/src/main/cpp/third_party/mnn \
  tools/engine_cli/main.cpp entry/src/main/cpp/llm_engine.cpp \
  entry/src/main/cpp/decision_json.cpp entry/src/main/cpp/decision_encoder.cpp \
  entry/src/main/cpp/decision_engine.cpp \
  entry/src/main/cpp/third_party/mnn/tokenizer/tokenizer.cpp \
  entry/src/main/cpp/third_party/mnn/tokenizer/unicode.cpp \
  entry/src/main/cpp/third_party/mnn/tokenizer/unicode_data.cpp \
  -L .mnn-build/mac -lMNN -Wl,-rpath,"$PWD/.mnn-build/mac" -o tools/engine_cli/engine_cli

./tools/engine_cli/engine_cli entry/src/main/resources/rawfile/model/config.json /tmp/mnn "介绍一下你自己"
./tools/engine_cli/engine_cli decide entry/src/main/resources/rawfile/decision tools/dec_ref/model/example_request.json
./tools/engine_cli/engine_cli encode entry/src/main/resources/rawfile/decision request.json   # 编码金标 dump
```

金标对齐（C++ vs 官方 `inference_mnn.py`，含 tokenizer 逐 ID、掩码哈希、w8 概率/胜者）：

```bash
tools/dec_ref/py tools/dec_ref/parity.py all     # encode / predict / all
```

本机（Apple M5，CPU 4 线程）实测：

```
[chat]  prompt=58 tok, output=44 tok, total=453 ms, 129 tok/s, 首字 110 ms
[decide] 官方示例：input=85 tok, questions=3, encode=4 ms, forward=132 ms, T=1.111183
```

手机端 decode 预计落在 25–60 tok/s 量级（0.3B INT4 / 4 线程）；决策单次前向预计数百毫秒到 1–2 秒（W8 / 4 线程），具体取决于芯片与散热。

---

## 5. 验证状态

已完成的验证：

- `libMNN.so` 由 `scripts/build_mnn_ohos.sh`（HarmonyOS NDK + SDK 自带 cmake 3.28.2）交叉编译通过，导出 `MNN::Transformer::Llm::createLLM / load / response` 与 `MNN::Express::Module::load` 等符号（决策图直接复用，无需重编 MNN）。
- `libllmbridge.so` 交叉编译 + 链接通过（含 vendored MNN tokenizer 与决策引擎），`NEEDED` 为 `libace_napi.z.so / libhilog_ndk.z.so / libMNN.so / libc++_shared.so`。
- `hvigorw assembleHap` 全量构建通过（ArkTS 无 error），产出 525 MB HAP；`libs/arm64-v8a/` 三个 .so 与两个模型包全部文件均为 `Stored`（不压缩，保证 `getRawFd` 可用）。
- 同一份 `ChatEngine` 在 macOS 上跑通单轮 / 多轮真实推理，流式输出与 token/耗时统计如上表。
- **决策金标对齐全部通过**（`tools/dec_ref/parity.py all`）：6 组用例（官方示例、中文售后、控制 token 注入、结构化 state、emoji/空白压力、单候选+5 级量表）编码逐 ID 一致、掩码 FNV 哈希一致；w8 图前向概率误差 < 2e-3、胜者/评分一致。
- 注意：W8 量化会改变部分决策（官方已说明）。例如官方"重复扣款"示例在 w8 下三个部门概率接近（returns 0.406 / shipping 0.404 / billing 0.191），与 fp16 基准可能有出入；实现与官方参考逐位对齐，属量化漂移而非移植误差。

**未在本机验证**：页面实际渲染与真机推理——当前机器没有连接 HarmonyOS 设备，也没有可用的模拟器镜像（模拟器还需要 x86_64 版 libMNN.so）。真机首次运行如遇问题，见下一节。

---

## 6. 排错

```bash
hdc list targets
hdc shell hilog -r | grep -i PocketLLM        # native 日志（TAG=PocketLLM）
hdc shell hilog -r | grep -i com.example.mnnpocket
```

- **一直停在「首次启动准备中」**：`getRawFd` 在部分打包方式下不可用，代码会自动回退到 `getRawFileContent`；若都失败，检查 HAP 内 `resources/rawfile/model/` 是否存在。
- **「模型没能启动」**：多为沙箱空间不足或权重未拷全。用 `hdc shell ls -l /data/storage/el2/base/haps/entry/files/model` 对照大小（`llm.mnn.weight` 应为 171481330 字节），删除该目录后重启应用会重新拷贝。
- **回答很慢**：确认设备不在省电模式；`config.json` 中 `thread_num` 可上调到 6（真机大核数量因芯片而异）。
- **想要更长的回答**：`Index.ets` 里 `runtime.load(..., 512)` 的 512 即单轮 `max_new_tokens`。
- **决策模式停在「决策模型准备中」**：检查 HAP 内 `resources/rawfile/decision/` 三个文件；沙箱目录为 `/data/storage/el2/base/haps/entry/files/decision`（`decision-w8-q8-embedding.mnn` 应为 341250856 字节）。
- **决策报「超过端侧上限」**：稠密掩码内存约束，默认 2048 tokens；精简场景或调高 `DecisionViewModel.maxTokens`（4096 时掩码约 64 MB）。
- **决策结果与预期不符**：先确认是 W8 量化漂移（见验证状态）还是协议问题——用 `engine_cli encode` 与 `parity.py` 对比编码。

---

## 7. 许可

- MNN：Apache-2.0（<https://github.com/alibaba/MNN>）
- 对话模型：taobao-mnn/MNN-Pocket-0.3B-MNN（<https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-MNN>）
- 决策模型：alivecat/MNN-Pocket-0.3B-Dec-MNN（<https://huggingface.co/alivecat/MNN-Pocket-0.3B-Dec-MNN>），上游 taobao-mnn/MNN-Pocket-0.3B-Dec，Apache-2.0；使用前请阅读各自模型卡许可。

# MNN Pocket · HarmonyOS 端侧对话

把 **MNN-Pocket-0.3B（INT4，约 179 MB）** 跑在 HarmonyOS 设备本地的对话应用：界面参照微信 / 小艺的聊天布局，输入框固定在底部，点击右侧圆形按钮后消息进入对话框，模型在设备上流式生成回答，并在每条回答下方给出 **tokens 数、推理毫秒数、tok/s、首字延迟**。

推理全程离线，不联网、不上传：权重随 HAP 一起安装，第一次启动时从 rawfile 拷进应用沙箱，之后由 MNN 的 CPU 后端（ARMv8.2 fp16）在本地计算。

> **⚠️ 权重不入库**：`llm.mnn.weight`（163.5 MB）超过 GitHub 单文件 100 MB 限制，未随仓库提供。clone 后请先执行 `scripts/fetch_model.sh` 下载模型，详见 **[SETUP.md](SETUP.md)**（下载方式、校验值、构建与运行步骤）。

| 依赖 | 版本 |
| --- | --- |
| DevEco Studio / SDK | 6.0.2（API 22） |
| MNN | 3.6.2（源码编译，LLM + TransformerFuse 打开） |
| 模型 | taobao-mnn/MNN-Pocket-0.3B-MNN（qwen3 架构，28 层 / hidden 896 / INT4 W4A16） |
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
│   ├── napi_init.cpp                    #   worker 线程 + 任务队列 + threadsafe function
│   ├── types/libllmbridge/index.d.ts    #   ArkTS 侧类型声明
│   └── third_party/mnn/include/         #   MNN 公共头文件（include/MNN + llm/）
├── src/main/ets/
│   ├── pages/Index.ets                  # 页面：状态机、布局、事件处理
│   ├── view/MessageBubble.ets           # 气泡 + 流式气泡（含指标行）
│   ├── view/InputBar.ets                # 底部输入框 + 发送按钮
│   ├── model/LlmRuntime.ets             # native 模块的 TS 单例封装
│   ├── model/ModelManager.ets           # rawfile → 沙箱的一次性拷贝（带进度/校验）
│   ├── model/ChatTypes.ets              # 消息、指标解析、格式化
│   ├── common/Theme.ets                 # 配色
│   └── entryability/EntryAbility.ets
└── src/main/resources/rawfile/model/    # 模型包（llm.mnn / .weight / tokenizer / 配置）
scripts/build_mnn_ohos.sh                # 重新交叉编译 MNN 并装进工程
scripts/fetch_model.sh                   # 重新下载模型
tools/engine_cli/                        # macOS 上验证同一份引擎代码的命令行工具
```

数据流：

```
Index.ets ──LlmRuntime──► libllmbridge.so
                              │  ① 任务入队（init / chat / reset）
                              ▼
                       worker 线程（唯一持有 MNN Llm）
                              │  ② ChatEngine::chat → Llm::response(流式 ostream)
                              ▼
                        libMNN.so（CPU + fp16）
                              │  ③ delta / done / error
                              ▼
                 napi_threadsafe_function ──► ArkTS 事件回调（UI 线程）
```

MNN 的 LLM 运行时不是线程安全的，单轮推理也会阻塞数百毫秒到数秒，所以 **native 侧只用一个专用 worker 线程**持有模型，JS 只投递任务、通过一个 threadsafe function 接收事件；UI 线程永远不被推理阻塞，token 才能像打字一样逐字出现。

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

产物：`entry/build/default/outputs/default/entry-default-unsigned.hap`（约 172 MB，含模型与三个 .so）。未配置签名时只能得到 unsigned 包，需用 DevEco 的自动签名或自行 `hap-sign-tool` 签名后 `hdc install`。

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

`tools/engine_cli` 与 app 编译的是**同一份** `llm_engine.cpp`，可在 macOS 上直接跑真实模型，用来确认 token/耗时统计与流式输出：

```bash
MNN=/path/to/model/MNN-Pocket-0.3B/MNN
clang++ -std=c++17 -O2 -I entry/src/main/cpp \
  -I "$MNN/include" -I "$MNN/transformers/llm/engine/include" \
  tools/engine_cli/main.cpp entry/src/main/cpp/llm_engine.cpp \
  -L "$MNN/build" -lMNN -lllm -Wl,-rpath,"$MNN/build" -o engine_cli

./engine_cli /path/to/model/config.json /tmp/mnn "介绍一下你自己" "用一句话解释量子纠缠"
```

本机（Apple M5，CPU 4 线程，Metal 未启用）实测：

```
[load] ok in 171 ms
[ask1] 介绍一下你自己
[stat1] prompt=56 tok, output=71 tok, prefill=65 ms, decode=355 ms,
        total=424 ms, 199.5 tok/s, 首字 66 ms
[ask2] 用一句话解释什么是量子纠缠        （第二轮带历史，prompt=143 tok）
[stat2] prompt=143 tok, output=32 tok, prefill=144 ms, decode=165 ms,
        total=312 ms, 193.2 tok/s, 首字 145 ms
```

手机端 decode 预计落在 25–60 tok/s 量级（0.3B INT4 / 4 线程），具体取决于芯片与散热。

---

## 5. 验证状态

已完成的验证：

- `libMNN.so` 由 `scripts/build_mnn_ohos.sh`（HarmonyOS NDK + SDK 自带 cmake 3.28.2）交叉编译通过，导出 `MNN::Transformer::Llm::createLLM / load / response` 等符号。
- `libllmbridge.so` 交叉编译 + 链接通过，`NEEDED` 为 `libace_napi.z.so / libhilog_ndk.z.so / libMNN.so / libc++_shared.so`。
- `hvigorw clean + assembleHap` 全量构建通过（ArkTS 无 error），产出 172 MB HAP；`libs/arm64-v8a/` 三个 .so 与 5 个模型文件均为 `Stored`（不压缩，保证 `getRawFd` 可用）。
- 同一份 `ChatEngine` 在 macOS 上跑通单轮 / 多轮真实推理，流式输出与 token/耗时统计如上表。

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

---

## 7. 许可

- MNN：Apache-2.0（<https://github.com/alibaba/MNN>）
- 模型：taobao-mnn/MNN-Pocket-0.3B-MNN（<https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-MNN>），使用前请阅读其模型卡许可。

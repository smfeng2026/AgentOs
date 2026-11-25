# MNN Pocket · 模型下载与运行说明

> **仓库不包含大权重文件。** `entry/src/main/resources/rawfile/model/llm.mnn.weight`（约 163.5 MB）超过 GitHub 单文件 100 MB 限制，已在 `.gitignore` 中排除。全新 clone 后必须先下载权重，否则无法构建出可运行的 HAP。其余 4 个模型文件（`llm.mnn` / `config.json` / `llm_config.json` / `tokenizer.mtok`）已随仓库提供。

---

## 1. 模型信息

| 项 | 值 |
| --- | --- |
| 模型 | MNN-Pocket-0.3B（qwen3 架构，28 层 / hidden 896，INT4 W4A16） |
| 上游仓库 | [taobao-mnn/MNN-Pocket-0.3B-MNN](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-MNN) |
| 缺失文件 | `llm.mnn.weight` |
| 文件大小 | 171,481,330 字节（约 163.5 MB） |
| MD5 | `6bb79e589438a30eb20a68e80344d600` |
| 放置路径 | `entry/src/main/resources/rawfile/model/llm.mnn.weight` |

## 2. 下载方式

### 方式 A：一键脚本（推荐）

在工程根目录（`demo/MNNPocket/`）执行：

```bash
scripts/fetch_model.sh
```

- 默认从 **hf-mirror.com**（HuggingFace 国内镜像）下载全部 5 个模型文件到 `entry/src/main/resources/rawfile/model/`，已存在的文件会被覆盖为上游最新版。
- 可指定其他 endpoint，例如官方源：

```bash
scripts/fetch_model.sh https://huggingface.co
```

### 方式 B：只手动下载缺失的权重

```bash
cd entry/src/main/resources/rawfile/model

# 国内镜像
curl -fL --retry 3 -O \
  https://hf-mirror.com/taobao-mnn/MNN-Pocket-0.3B-MNN/resolve/main/llm.mnn.weight

# 或官方源
curl -fL --retry 3 -O \
  https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-MNN/resolve/main/llm.mnn.weight
```

也可以直接在浏览器打开上游仓库的 [Files 页面](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-MNN/tree/main) 下载 `llm.mnn.weight` 后放入上述目录。

### 下载后校验

```bash
cd entry/src/main/resources/rawfile/model
ls -l llm.mnn.weight          # 应为 171481330 字节
md5 llm.mnn.weight            # macOS；应为 6bb79e589438a30eb20a68e80344d600
# Linux 用 md5sum llm.mnn.weight
```

> 注意：上游若更新权重，大小/MD5 可能变化；以能正常加载为准，本表数值对应仓库当前使用的版本。

## 3. 工程运行

环境要求：**DevEco Studio / SDK 6.0.2（API 22）**，HarmonyOS 6.0.2+ 的 **arm64 真机**（模拟器需另行编译 x86_64 版 `libMNN.so`）。`libMNN.so`（3.6.2，含 LLM 运行时）与 `libc++_shared.so` 已预编译在 `entry/libs/arm64-v8a/`，无需重新编译 MNN。

### 3.1 DevEco Studio（推荐）

1. 确认权重已按第 2 节下载到位。
2. `File → Open` 打开 `demo/MNNPocket/`，等待 Sync 完成。
3. `File → Project Structure → Signing Configs` 勾选 **Automatically generate signature**（需登录华为开发者账号）。**未签名的 HAP 无法安装。**
4. 连接真机，点击 **Run**。首次构建会编译 NAPI 桥 `libllmbridge.so`（几秒）。
5. 应用首次启动会把权重从 rawfile 拷入沙箱（界面显示「首次启动准备中 xx%」），随后加载模型进入对话。

### 3.2 命令行构建

```bash
export DEVECO_SDK_HOME=/Applications/DevEco-Studio.app/Contents/sdk
export JAVA_HOME=/Applications/DevEco-Studio.app/Contents/jbr/Contents/Home
export PATH=/Applications/DevEco-Studio.app/Contents/tools/node/bin:$JAVA_HOME/bin:$PATH

/Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw \
  --mode module -p product=default assembleHap --no-daemon
```

产物：`entry/build/default/outputs/default/entry-default-unsigned.hap`（约 172 MB，含模型与 .so）。命令行只能产出 unsigned 包，需签名后安装：

```bash
hdc install entry-default-signed.hap
```

签名可用 DevEco 自动签名，或自行 `hap-sign-tool`。

### 3.3 运行验证与排错

```bash
hdc list targets
hdc shell hilog -r | grep -i PocketLLM                 # native 日志
hdc shell ls -l /data/storage/el2/base/haps/entry/files/model   # 沙箱内权重应为 171481330 字节
```

- 构建报错找不到 `llm.mnn.weight` → 权重未下载，回到第 2 节。
- 一直停在「首次启动准备中」/「模型没能启动」→ 多为沙箱空间不足或权重未拷全，删除上述沙箱 model 目录后重启应用。
- 更多细节（界面交互、指标口径、重新编译 MNN、桌面端引擎验证）见 [README.md](README.md)。

## 4. 许可

- MNN：Apache-2.0（<https://github.com/alibaba/MNN>）
- 模型：taobao-mnn/MNN-Pocket-0.3B-MNN，使用前请阅读其[模型卡](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-MNN)许可。
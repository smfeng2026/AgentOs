# AgentOs
基于鸿蒙框架做的 一些场景 demo 原型

## 目录
- `demo/` 鸿蒙应用原型(noteApp、messageApp)
- `models/` 决策模型
  - `laya/` [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) 上游源码副本(pip 包 `laya` 同源)
  - `laya-en/` 英文 checkpoint([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya),权重需单独下载,见下)
  - 规划中:laya-omni / laya-multilingual / siglip2-base-patch16-256
- `web/` laya 决策模型测试网站(FastAPI + 网页)

## 运行 web 测试站(laya 决策模型)

### 环境要求
- Python 3.10 ~ 3.13(3.14 尚无 torch 轮子,实测用 3.12 可用)
- 内存 ≥ 2GB,无 GPU 可运行(CPU fp16 推理,峰值约 650MB,3 个问题约 5 秒)

### 步骤(以下命令均在仓库根目录执行)

**1. 创建虚拟环境**
```bash
python3.12 -m venv .venv        # 若系统默认 python3 是 3.14,需显式指定 3.10~3.13
source .venv/bin/activate
```

**2. 安装依赖**
```bash
pip install fastapi uvicorn laya python-multipart
```
- `python-multipart` 必装:表单接口 `/api/predict` 依赖它,缺失时服务直接启动失败
- 国内网络可加镜像加速:`-i https://pypi.tuna.tsinghua.edu.cn/simple`

**3. 下载模型权重(约 803MB,`.gitignore` 已排除不入库)**
```bash
curl -L -C - -o models/laya-en/model.safetensors \
  https://huggingface.co/convaiinnovations/laya/resolve/main/model.safetensors
```
- 国内直连 HuggingFace 易断,可走本地代理(端口按实际调整):
  `curl -L -C - -x http://127.0.0.1:7890 -o models/laya-en/model.safetensors ...`
- `-C -` 支持断点续传:中断后重复执行同一命令即可接着下
- 下载完成后可校验完整性:文件应为 842,609,210 字节,文件头前 8 字节为小端 uint64 长度 21536,后跟 JSON 元数据(含 206 个张量)

**4. 启动服务**
```bash
cd web
python server.py               # 默认 http://localhost:8030
python server.py --port 8031   # 自定义端口
```

**5. 测试**

浏览器打开 http://localhost:8030。页面功能:
- **文本 / 图片 / 音频 / 视频** 四种输入(后三者为多模态,见下节)
- **6 个一键填充的中文示例**:客服工单分流、重复扣款、威胁升级、退货审核、内容审核、租约条款
- 问题类型 `choice`(单选)/ `score`(评分)/ `noul`(是非),支持中文指令与中文候选
- 结果区展示:概率条形图(最高项高亮)、答案置信度条、vega 路径额外显示 route(auto/review)与语音转写文本
- 「原始 JSON」可展开查看完整返回

或直接调 API:
```bash
curl -X POST http://localhost:8030/api/predict \
  -F 'state=客户来信:我的3月份发票被重复扣款了4521元,要求今天立即退款,否则我们就取消套餐。' \
  -F 'questions={"q1":{"type":"choice","instructions":"这封邮件应该由哪个部门处理?","criteria":["账务","技术","其他"]},"q3":{"type":"noul","instructions":"用户是否威胁要取消服务?"}}'

# 图片决策:
curl -X POST http://localhost:8030/api/predict \
  -F 'file=@screenshot.png;type=image/png' \
  -F 'questions={"q1":{"type":"choice","instructions":"Which department?","criteria":["billing","technical","other"]}}'
```

问题格式与返回字段说明见 [web/README.md](web/README.md)。

## 多模态运行(图片 / 音频 / 视频)

laya 本体是纯文本决策模型;图片/音频/视频由 **vega 引擎**(`laya.Agent(vegaml=True)`,基于 Qwen3.5-0.8B 多模态 reader)支持。网页端切换到对应标签页上传文件即可;首次上传时自动加载(本机约 5 秒,自动使用 Apple MPS 加速)。

**一次性准备**(下载约 2.7GB,之后完全离线):
```bash
# 1. vega-U 引擎(约 66MB)+ Qwen3.5-0.8B reader(约 1.7GB)
mkdir -p models/vega-U/backbone
for f in config.json vega_config.json engine.safetensors port.safetensors stats.npz; do
  curl -L -C - -o "models/vega-U/$f" "https://hf-mirror.com/nandakishorm/vega-U/resolve/main/$f"
done
for f in config.json generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json preprocessor_config.json chat_template.jinja merges.txt vocab.json model.safetensors-00001-of-00001.safetensors; do
  curl -L -C - -o "models/vega-U/backbone/$f" "https://hf-mirror.com/Qwen/Qwen3.5-0.8B/resolve/main/$f"
done

# 2. 中文语音转写 whisper-small(约 920MB;仅音频/视频需要)
mkdir -p models/whisper-small
for f in config.json generation_config.json model.safetensors preprocessor_config.json tokenizer.json tokenizer_config.json; do
  curl -L -C - -o "models/whisper-small/$f" "https://hf-mirror.com/openai/whisper-small/resolve/main/$f"
done
```
- 国内直连 HuggingFace 易断:镜像站 `hf-mirror.com` 一般可直连;也可走代理(`-x http://127.0.0.1:7890`)
- `-C -` 断点续传:中断后重复同一命令
- `models/vega-U/`、`models/whisper-small/` 均已在 `.gitignore` 中,不会入库
- 依赖追加:`pip install torchvision`(Qwen 视觉处理需要)

**多模态依赖检查**:`GET /api/info` 返回 `"multimodal": true` 即就绪。音频走「whisper 转写 → vega 决策」两段式,中文转写默认 whisper-small;视频有声轨则转写决策、无声则抽 4 帧走图片路径。实测速度(Apple Silicon,文本 3.8s / 图片 5.2s / 含转写音频更久)。

### 常见问题
| 现象 | 解决 |
|---|---|
| `pip install` 装 torch 失败 | Python 版本太新(如 3.14),换 3.10~3.13 建 venv |
| 启动报 `Form data requires "python-multipart"` | `pip install python-multipart` |
| 图片/音频报 `requires torchvision` | `pip install torchvision` |
| git/curl 访问 GitHub 或 HuggingFace 卡住、报 `HTTP2 framing layer` | 走本地代理:`git -c http.proxy=http://127.0.0.1:7890 ...`,curl 加 `-x http://127.0.0.1:7890`;或用 `hf-mirror.com` 镜像 |
| 下载中断 | 同一命令重复执行(`-C -` 断点续传) |

## 运行 demo/(鸿蒙应用)
1. 用 DevEco Studio 打开 `demo/messageApp` 或 `demo/noteApp` 目录
2. 等待 Sync 自动安装 `oh_modules` 依赖
3. Previewer(预览器)/ 模拟器 / 真机 Run ▶ 运行

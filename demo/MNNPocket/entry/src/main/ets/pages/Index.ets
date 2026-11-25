import { common } from '@kit.AbilityKit';
import {
  ChatMessage,
  ChatStats,
  Role,
  nowLabel,
  parseDelta,
  parseDoneText,
  parseError,
  parseInit,
  parseStats
} from '../model/ChatTypes';
import { LlmRuntime } from '../model/LlmRuntime';
import { ModelManager } from '../model/ModelManager';
import { DecisionViewModel } from '../model/DecisionViewModel';
import { MessageBubble, StreamingBubble } from '../view/MessageBubble';
import { InputBar } from '../view/InputBar';
import { ModeSwitch } from '../view/ModeSwitch';
import { DecisionPage } from './DecisionPage';
import { Palette } from '../common/Theme';

enum Stage {
  BOOT = 0,
  COPYING = 1,
  LOADING = 2,
  READY = 3,
  FAILED = 4
}

@Entry
@Component
struct Index {
  @State stage: Stage = Stage.BOOT;
  @State stageText: string = '正在准备模型…';
  @State progress: number = 0;
  @State errorText: string = '';
  @State messages: ChatMessage[] = [];
  @State inputText: string = '';
  @State streaming: boolean = false;
  @State streamingText: string = '';
  @State streamingElapsed: number = 0;
  @State loadMs: number = 0;
  @State sessionTokens: number = 0;
  @State mode: string = 'chat'; // chat | decide
  @State decVm: DecisionViewModel = DecisionViewModel.get();

  private scroller: Scroller = new Scroller();
  private runtime: LlmRuntime = LlmRuntime.get();
  private messageSeq: number = 0;
  private tickTimer: number = -1;
  private tickStart: number = 0;
  private scrollPending: boolean = false;

  aboutToAppear(): void {
    this.runtime.bind((name: string, payload: string) => {
      this.onNativeEvent(name, payload);
    });
    this.bootstrap();
  }

  aboutToDisappear(): void {
    this.stopTick();
  }

  // -------------------------------------------------------------------------
  // model bootstrap
  // -------------------------------------------------------------------------

  /**
   * UIAbility context, with a fallback: getUIContext() may not be attached yet
   * when aboutToAppear() fires on some builds.
   */
  private abilityContext(): common.UIAbilityContext {
    try {
      const host = this.getUIContext().getHostContext();
      if (host !== undefined && host !== null) {
        return host as common.UIAbilityContext;
      }
    } catch (error) {
      console.warn(`getHostContext failed, falling back to getContext: ${JSON.stringify(error)}`);
    }
    return getContext(this) as common.UIAbilityContext;
  }

  private async bootstrap(): Promise<void> {
    this.stage = Stage.BOOT;
    this.stageText = '正在准备模型…';
    this.progress = 0;
    this.errorText = '';
    try {
      const context = this.abilityContext();
      const configPath = await ModelManager.ensure(context, (copied: number, total: number) => {
        const percent = total > 0 ? Math.floor(copied * 100 / total) : 0;
        this.progress = percent;
        if (percent < 100) {
          this.stage = Stage.COPYING;
          this.stageText = `首次启动：正在写入模型文件 ${percent}%`;
        } else {
          this.stage = Stage.LOADING;
          this.stageText = '模型文件就绪，正在加载到内存…';
        }
      });
      this.stage = Stage.LOADING;
      this.stageText = '模型文件就绪，正在加载到内存…';
      this.runtime.load(configPath, ModelManager.tmpDir(context), 512);
    } catch (error) {
      this.stage = Stage.FAILED;
      this.errorText = `${error}`;
      console.error(`bootstrap failed: ${JSON.stringify(error)}`);
    }
  }

  private onNativeEvent(name: string, payload: string): void {
    if (name.startsWith('dec_')) {
      this.decVm.handleEvent(name, payload);
      return;
    }
    switch (name) {
      case 'init_progress':
        this.stageText = '正在初始化推理引擎…';
        break;
      case 'ready': {
        const info = parseInit(payload);
        this.loadMs = info === null ? 0 : info.loadMs;
        this.stage = Stage.READY;
        this.stageText = '已就绪';
        break;
      }
      case 'start':
        this.beginStreaming();
        break;
      case 'delta': {
        const chunk = parseDelta(payload);
        if (chunk.length > 0) {
          this.streamingText += chunk;
          this.scrollToBottom();
        }
        break;
      }
      case 'done':
        this.finishTurn(payload);
        break;
      case 'error':
        this.failTurn(parseError(payload));
        break;
      default:
        break;
    }
  }

  // -------------------------------------------------------------------------
  // turn lifecycle
  // -------------------------------------------------------------------------

  private beginStreaming(): void {
    this.streaming = true;
    this.streamingText = '';
    this.streamingElapsed = 0;
    this.tickStart = Date.now();
    this.stopTick();
    this.tickTimer = setInterval(() => {
      this.streamingElapsed = Date.now() - this.tickStart;
    }, 100);
    this.scrollToBottom();
  }

  private finishTurn(payload: string): void {
    this.stopTick();
    const stats = parseStats(payload);
    const done = parseDoneText(payload);
    let text = done === null ? '' : done.text;
    if (text.length === 0) {
      text = this.streamingText;
    }
    if (text.trim().length === 0) {
      text = '（模型没有返回内容，可以再试一次）';
    }
    this.messages.push(new ChatMessage(++this.messageSeq, Role.ASSISTANT, text.trim(), stats, nowLabel()));
    this.streaming = false;
    this.streamingText = '';
    this.sessionTokens += stats.outputTokens;
    this.scrollToBottom();
  }

  private failTurn(message: string): void {
    this.stopTick();
    this.streaming = false;
    this.streamingText = '';
    if (this.stage === Stage.LOADING || this.stage === Stage.BOOT) {
      this.stage = Stage.FAILED;
      this.errorText = message;
      return;
    }
    this.toast(message);
  }

  private send(): void {
    const text = this.inputText.trim();
    if (text.length === 0 || this.stage !== Stage.READY || this.streaming) {
      return;
    }
    this.messages.push(new ChatMessage(++this.messageSeq, Role.USER, text, new ChatStats(), nowLabel()));
    this.inputText = '';
    this.streaming = true;
    this.streamingText = '';
    this.streamingElapsed = 0;
    this.scrollToBottom();
    this.runtime.ask(text);
  }

  private clearChat(): void {
    if (this.streaming) {
      this.toast('正在生成回答，请稍候再清空');
      return;
    }
    this.runtime.clear();
    this.messages = [];
    this.sessionTokens = 0;
  }

  private scrollToBottom(): void {
    if (this.scrollPending) {
      return;
    }
    this.scrollPending = true;
    // The list needs a layout pass after the new bubble is appended, and a
    // second nudge once the streaming text has grown the bubble.
    setTimeout(() => {
      this.scroller.scrollEdge(Edge.Bottom);
      setTimeout(() => {
        this.scrollPending = false;
        this.scroller.scrollEdge(Edge.Bottom);
      }, 120);
    }, 60);
  }

  private headerSubtitle(): string {
    if (this.mode === 'decide') {
      return '决策模式 · 单次前向 · 概率输出';
    }
    if (this.stage !== Stage.READY) {
      return this.stageText;
    }
    const parts: string[] = ['端侧推理'];
    if (this.loadMs > 0) {
      parts.push(`加载 ${this.loadMs} ms`);
    }
    if (this.sessionTokens > 0) {
      parts.push(`本次 ${this.sessionTokens} tok`);
    }
    return parts.join(' · ');
  }

  private stopTick(): void {
    if (this.tickTimer !== -1) {
      clearInterval(this.tickTimer);
      this.tickTimer = -1;
    }
  }

  private toast(message: string): void {
    try {
      this.getUIContext().getPromptAction().showToast({ message: message, duration: 2200 });
    } catch (error) {
      console.warn(`toast failed: ${JSON.stringify(error)}`);
    }
  }

  // -------------------------------------------------------------------------
  // pieces
  // -------------------------------------------------------------------------

  @Builder
  statusPill() {
    Row({ space: 5 }) {
      Circle()
        .width(6)
        .height(6)
        .fill(this.stage === Stage.READY ? Palette.ACCENT :
          (this.stage === Stage.FAILED ? Palette.DANGER : Palette.WARNING))
      Text(this.stage === Stage.READY ? '已就绪' :
        (this.stage === Stage.FAILED ? '加载失败' : '加载中'))
        .fontSize(11)
        .fontColor(this.stage === Stage.READY ? '#0B8043' :
          (this.stage === Stage.FAILED ? Palette.DANGER : '#9A6700'))
    }
    .padding({ left: 8, right: 8, top: 4, bottom: 4 })
    .backgroundColor(this.stage === Stage.READY ? '#E8F8EE' :
      (this.stage === Stage.FAILED ? '#FDECEC' : '#FFF7E6'))
    .borderRadius(20)
  }

  @Builder
  header() {
    Column() {
      Row({ space: 10 }) {
        Stack() {
          Text('MNN')
            .fontSize(11)
            .fontWeight(FontWeight.Bold)
            .fontColor(Color.White)
        }
        .width(38)
        .height(38)
        .borderRadius(12)
        .linearGradient({
          angle: 135,
          colors: this.mode === 'chat'
            ? [[Palette.AVATAR_AI_START, 0.0], [Palette.AVATAR_AI_END, 1.0]]
            : [[Palette.DEC_GRADIENT_START, 0.0], [Palette.DEC_GRADIENT_END, 1.0]]
        })

        Column({ space: 2 }) {
          Row({ space: 6 }) {
            Text('Pocket')
              .fontSize(17)
              .fontWeight(FontWeight.Bold)
              .fontColor(Palette.TEXT_PRIMARY)
            Text(this.mode === 'chat' ? '0.3B' : '0.3B-Dec')
              .fontSize(9)
              .fontColor(Color.White)
              .padding({ left: 4, right: 4, top: 1, bottom: 1 })
              .backgroundColor(this.mode === 'chat' ? '#7C5CFF' : Palette.DEC_ACCENT)
              .borderRadius(4)
          }

          Text(this.headerSubtitle())
            .fontSize(11)
            .fontColor(Palette.TEXT_SECONDARY)
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)

        if (this.mode === 'chat') {
          this.statusPill()

          Button({ type: ButtonType.Circle }) {
            Image($r('app.media.icon_clear'))
              .width(18)
              .height(18)
              .objectFit(ImageFit.Contain)
          }
          .width(34)
          .height(34)
          .backgroundColor(Palette.CHIP_BG)
          .enabled(this.stage === Stage.READY && this.messages.length > 0)
          .onClick(() => {
            this.clearChat();
          })
        }
      }
      .width('100%')
      .padding({ left: 14, right: 14, top: 8, bottom: 8 })

      Row() {
        ModeSwitch({
          mode: this.mode,
          onSwitch: (next: string) => {
            this.mode = next;
          }
        })
      }
      .width('100%')
      .justifyContent(FlexAlign.Center)
      .padding({ bottom: 9 })

      // Mode accent hairline: green for chat, indigo for decision.
      Row()
        .width('100%')
        .height(2)
        .linearGradient({
          angle: 90,
          colors: this.mode === 'chat'
            ? [[Palette.ACCENT, 0.0], ['#95EC69', 1.0]]
            : [[Palette.DEC_GRADIENT_START, 0.0], [Palette.DEC_GRADIENT_END, 1.0]]
        })
        .opacity(0.55)
    }
    .width('100%')
    .backgroundColor(Palette.HEADER_BG)
  }

  @Builder
  suggestion(text: string) {
    Row({ space: 8 }) {
      Text(text)
        .fontSize(13.5)
        .fontColor(Palette.TEXT_PRIMARY)
        .layoutWeight(1)
        .textAlign(TextAlign.Start)
        .maxLines(1)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
      Text('›')
        .fontSize(16)
        .fontColor(Palette.TEXT_MUTED)
    }
    .width('100%')
    .padding({ left: 14, right: 14, top: 11, bottom: 11 })
    .backgroundColor(Palette.CARD_BG)
    .borderRadius(12)
    .onClick(() => {
      this.inputText = text;
    })
  }

  @Builder
  welcome() {
    Column() {
      Column({ space: 12 }) {
        Text('有什么可以帮你？')
          .fontSize(21)
          .fontWeight(FontWeight.Bold)
          .fontColor(Palette.TEXT_PRIMARY)
        Text('MNN-Pocket-0.3B · INT4 · 171 MB 权重\n推理全部在设备本地完成，不联网、不上传')
          .fontSize(12.5)
          .fontColor(Palette.TEXT_SECONDARY)
          .textAlign(TextAlign.Center)
          .lineHeight(19)

        Column({ space: 9 }) {
          this.suggestion('介绍一下你自己')
          this.suggestion('用一句话解释什么是量子纠缠')
          this.suggestion('帮我写一条周末露营的朋友圈文案')
        }
        .width('100%')
        .margin({ top: 10 })
      }
      .width('100%')
      .constraintSize({ maxWidth: 420 })
    }
    .width('100%')
    .layoutWeight(1)
    .justifyContent(FlexAlign.Center)
    .padding({ left: 22, right: 22, bottom: 24 })
  }

  @Builder
  chatList() {
    List({ space: 14, scroller: this.scroller }) {
      ForEach(this.messages, (item: ChatMessage) => {
        ListItem() {
          MessageBubble({ role: item.role, text: item.text, stats: item.stats })
        }
      }, (item: ChatMessage) => item.id.toString())

      if (this.streaming) {
        ListItem() {
          StreamingBubble({ text: this.streamingText, elapsedMs: this.streamingElapsed })
        }
      }
    }
    .width('100%')
    .layoutWeight(1)
    .padding({ left: 12, right: 12, top: 12, bottom: 6 })
    .scrollBar(BarState.Off)
    .edgeEffect(EdgeEffect.Spring)
  }

  @Builder
  loadingPanel() {
    Column({ space: 18 }) {
      LoadingProgress()
        .width(46)
        .height(46)
        .color(Palette.ACCENT)

      Column({ space: 6 }) {
        Text(this.stage === Stage.COPYING ? '首次启动准备中' : '模型加载中')
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor(Palette.TEXT_PRIMARY)
        Text(this.stageText)
          .fontSize(12.5)
          .fontColor(Palette.TEXT_SECONDARY)
          .textAlign(TextAlign.Center)
      }

      if (this.stage === Stage.COPYING) {
        Column({ space: 8 }) {
          Progress({ value: this.progress, total: 100, type: ProgressType.Linear })
            .width(220)
            .color(Palette.ACCENT)
            .backgroundColor('#E4E7EB')
            .borderRadius(4)
          Text(`${this.progress}%`)
            .fontSize(11)
            .fontColor(Palette.TEXT_MUTED)
        }
      } else {
        Text('0.3B INT4 · 通常 1–2 秒')
          .fontSize(11)
          .fontColor(Palette.TEXT_MUTED)
      }
    }
    .width('100%')
    .layoutWeight(1)
    .justifyContent(FlexAlign.Center)
  }

  @Builder
  errorPanel() {
    Column({ space: 14 }) {
      Text('⚠️')
        .fontSize(34)
      Text('模型没能启动')
        .fontSize(17)
        .fontWeight(FontWeight.Medium)
        .fontColor(Palette.TEXT_PRIMARY)
      Text(this.errorText)
        .fontSize(12)
        .fontColor(Palette.TEXT_SECONDARY)
        .textAlign(TextAlign.Center)
        .maxLines(6)
        .padding({ left: 24, right: 24 })
      Button('重新加载')
        .fontSize(14)
        .height(38)
        .padding({ left: 22, right: 22 })
        .backgroundColor(Palette.ACCENT)
        .onClick(() => {
          this.bootstrap();
        })
    }
    .width('100%')
    .layoutWeight(1)
    .justifyContent(FlexAlign.Center)
  }

  build() {
    Column() {
      this.header()

      if (this.mode === 'decide') {
        DecisionPage({ vm: this.decVm })
      } else if (this.stage === Stage.READY) {
        if (this.messages.length === 0 && !this.streaming) {
          this.welcome()
        } else {
          this.chatList()
        }
        InputBar({
          text: $inputText,
          canSend: this.stage === Stage.READY && !this.streaming && this.inputText.trim().length > 0,
          onSend: () => {
            this.send();
          }
        })
      } else if (this.stage === Stage.FAILED) {
        this.errorPanel()
      } else {
        this.loadingPanel()
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Palette.PAGE_BG)
  }
}

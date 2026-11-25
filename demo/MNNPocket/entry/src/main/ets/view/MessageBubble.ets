import { ChatStats, Role, formatTokens } from '../model/ChatTypes';
import { Palette } from '../common/Theme';

@Extend(Text)
function metaLine() {
  .fontSize(10.5)
  .fontColor(Palette.TEXT_MUTED)
  .lineHeight(15)
}

@Extend(Text)
function bubbleText() {
  .fontSize(15.5)
  .lineHeight(23)
  .fontColor(Palette.TEXT_PRIMARY)
  .copyOption(CopyOptions.LocalDevice)
}

@Component
export struct MessageBubble {
  // Finished bubbles never change, so plain members are enough: no deep copy of
  // the stats object (which would drop its methods) and no needless observation.
  role: Role = Role.ASSISTANT;
  text: string = '';
  stats: ChatStats = new ChatStats();

  private isUser(): boolean {
    return this.role === Role.USER;
  }

  @Builder
  avatar() {
    Stack() {
      Text(this.isUser() ? '我' : 'P')
        .fontSize(13)
        .fontWeight(FontWeight.Medium)
        .fontColor(Color.White)
    }
    .width(34)
    .height(34)
    .borderRadius(10)
    .linearGradient(this.isUser()
      ? { angle: 135, colors: [[Palette.AVATAR_USER, 0.0], ['#2F7BEA', 1.0]] }
      : { angle: 135, colors: [[Palette.AVATAR_AI_START, 0.0], [Palette.AVATAR_AI_END, 1.0]] })
  }

  @Builder
  metrics() {
    Column({ space: 3 }) {
      Text(`输入 ${formatTokens(this.stats.promptTokens)} tok  ·  输出 ${formatTokens(this.stats.outputTokens)} tok  ·  上下文 ${formatTokens(this.stats.contextTokens)} tok`)
        .metaLine()
      Text(`推理 ${this.stats.totalMs} ms  ·  ${this.stats.decodeTps.toFixed(1)} tok/s  ·  首字 ${this.stats.ttfaMs} ms`)
        .metaLine()
    }
    .alignItems(HorizontalAlign.Start)
    .padding({ left: 10, right: 10, top: 7, bottom: 7 })
    .backgroundColor(Palette.CHIP_BG)
    .borderRadius(10)
    .margin({ top: 1 })
  }

  build() {
    Row({ space: 8 }) {
      if (!this.isUser()) {
        this.avatar()
      }
      Column({ space: 5 }) {
        Text(this.text)
          .bubbleText()
          .padding({ left: 13, right: 13, top: 10, bottom: 10 })
          .backgroundColor(this.isUser() ? Palette.BUBBLE_USER : Palette.BUBBLE_AI)
          .borderRadius(this.isUser()
            ? { topLeft: 16, topRight: 5, bottomLeft: 16, bottomRight: 16 }
            : { topLeft: 5, topRight: 16, bottomLeft: 16, bottomRight: 16 })
          .constraintSize({ maxWidth: '78%' })
        if (!this.isUser() && this.stats.outputTokens > 0) {
          this.metrics()
        }
      }
      .alignItems(this.isUser() ? HorizontalAlign.End : HorizontalAlign.Start)

      if (this.isUser()) {
        this.avatar()
      }
    }
    .width('100%')
    .alignItems(VerticalAlign.Top)
    .justifyContent(this.isUser() ? FlexAlign.End : FlexAlign.Start)
  }
}

/** Assistant bubble shown while tokens are still arriving. */
@Component
export struct StreamingBubble {
  @Prop text: string = '';
  @Prop elapsedMs: number = 0;

  @Builder
  avatar() {
    Stack() {
      Text('P')
        .fontSize(13)
        .fontWeight(FontWeight.Medium)
        .fontColor(Color.White)
    }
    .width(34)
    .height(34)
    .borderRadius(10)
    .linearGradient({
      angle: 135,
      colors: [[Palette.AVATAR_AI_START, 0.0], [Palette.AVATAR_AI_END, 1.0]]
    })
  }

  @Builder
  thinking() {
    Row({ space: 5 }) {
      LoadingProgress()
        .width(18)
        .height(18)
        .color(Palette.TEXT_MUTED)
      Text('正在思考…')
        .fontSize(14)
        .fontColor(Palette.TEXT_MUTED)
    }
    .padding({ left: 14, right: 16, top: 11, bottom: 11 })
    .backgroundColor(Palette.BUBBLE_AI)
    .borderRadius({ topLeft: 5, topRight: 16, bottomLeft: 16, bottomRight: 16 })
  }

  build() {
    Row({ space: 8 }) {
      this.avatar()
      Column({ space: 5 }) {
        if (this.text.length === 0) {
          this.thinking()
        } else {
          Text(`${this.text}▍`)
            .bubbleText()
            .padding({ left: 13, right: 13, top: 10, bottom: 10 })
            .backgroundColor(Palette.BUBBLE_AI)
            .borderRadius({ topLeft: 5, topRight: 16, bottomLeft: 16, bottomRight: 16 })
            .constraintSize({ maxWidth: '78%' })
            .copyOption(CopyOptions.LocalDevice)
        }
        Text(`${this.elapsedMs} ms  ·  生成中…`)
          .metaLine()
          .padding({ left: 4 })
      }
      .alignItems(HorizontalAlign.Start)
    }
    .width('100%')
    .alignItems(VerticalAlign.Top)
    .justifyContent(FlexAlign.Start)
  }
}

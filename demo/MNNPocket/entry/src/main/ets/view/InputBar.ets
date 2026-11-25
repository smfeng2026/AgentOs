import { Palette } from '../common/Theme';

/** Bottom composer: multiline field on the left, round send button on the right. */
@Component
export struct InputBar {
  @Link text: string;
  @Prop canSend: boolean = false;
  onSend: () => void = () => {
  };

  build() {
    Row({ space: 10 }) {
      TextArea({ text: this.text, placeholder: '说说你的诉求，例如：帮我写一段自我介绍' })
        .placeholderColor(Palette.TEXT_MUTED)
        .fontSize(15)
        .fontColor(Palette.TEXT_PRIMARY)
        .backgroundColor(Palette.INPUT_BG)
        .borderRadius(20)
        .padding({ left: 14, right: 14, top: 9, bottom: 9 })
        .maxLines(4)
        .maxLength(2000)
        .layoutWeight(1)
        .onChange((value: string) => {
          this.text = value;
        })

      Button({ type: ButtonType.Circle, stateEffect: true }) {
        Image($r('app.media.icon_send'))
          .width(20)
          .height(20)
          .objectFit(ImageFit.Contain)
      }
      .width(42)
      .height(42)
      .backgroundColor(this.canSend ? Palette.ACCENT : Palette.ACCENT_DISABLED)
      .enabled(this.canSend)
      .onClick(() => {
        if (this.canSend) {
          this.onSend();
        }
      })
    }
    .width('100%')
    .alignItems(VerticalAlign.Bottom)
    .padding({ left: 12, right: 12, top: 8, bottom: 10 })
    .backgroundColor(Palette.HEADER_BG)
  }
}

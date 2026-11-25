/**
 * Top segmented control: 对话 / 决策. The active pill carries the mode accent
 * (WeChat green vs indigo) so switching "brains" is visually unmistakable.
 */

import { Palette } from '../common/Theme';

@Component
export struct ModeSwitch {
  @Prop mode: string = 'chat';
  onSwitch: (mode: string) => void = () => {
  };

  @Builder
  segment(label: string, icon: string, value: string, accent: string) {
    Row({ space: 4 }) {
      Text(icon)
        .fontSize(12)
      Text(label)
        .fontSize(13)
        .fontWeight(this.mode === value ? FontWeight.Bold : FontWeight.Medium)
        .fontColor(this.mode === value ? accent : Palette.TEXT_SECONDARY)
    }
    .justifyContent(FlexAlign.Center)
    .height(30)
    .layoutWeight(1)
    .borderRadius(15)
    .backgroundColor(this.mode === value ? Palette.CARD_BG : Color.Transparent)
    .shadow(this.mode === value
      ? { radius: 6, color: '#1A000000', offsetX: 0, offsetY: 1 }
      : { radius: 0, color: Color.Transparent, offsetX: 0, offsetY: 0 })
    .onClick(() => {
      if (this.mode !== value) {
        animateTo({ duration: 180, curve: Curve.EaseOut }, () => {
          this.onSwitch(value);
        });
      }
    })
  }

  build() {
    Row() {
      this.segment('对话', '💬', 'chat', '#0B8043')
      this.segment('决策', '⚖️', 'decide', Palette.DEC_ACCENT)
    }
    .width(172)
    .padding(3)
    .borderRadius(18)
    .backgroundColor(Palette.INPUT_BG)
  }
}
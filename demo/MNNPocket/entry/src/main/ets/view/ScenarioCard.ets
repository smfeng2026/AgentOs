/**
 * Scenario gallery card: title, one-line summary and question-type badges.
 */

import { Scenario, typeLabel } from '../model/DecisionTypes';
import { Palette, typeTagColor } from '../common/Theme';

@Component
export struct ScenarioCard {
  scenario: Scenario = new Scenario('', '', '', true, '', []);
  onOpen: () => void = () => {
  };
  onEdit: () => void = () => {
  };
  onDelete: (() => void) | null = null;

  @Builder
  typeTag(type: string) {
    Text(`${type} · ${typeLabel(type)}`)
      .fontSize(10)
      .fontColor(typeTagColor(type))
      .padding({ left: 7, right: 7, top: 2.5, bottom: 2.5 })
      .backgroundColor(Palette.CHIP_BG)
      .borderRadius(9)
  }

  build() {
    Column({ space: 8 }) {
      Row({ space: 8 }) {
        Column({ space: 3 }) {
          Text(this.scenario.title)
            .fontSize(15.5)
            .fontWeight(FontWeight.Bold)
            .fontColor(Palette.TEXT_PRIMARY)
          Text(this.scenario.subtitle)
            .fontSize(12)
            .fontColor(Palette.TEXT_SECONDARY)
            .maxLines(2)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)

        Text('›')
          .fontSize(20)
          .fontColor(Palette.TEXT_MUTED)
      }
      .width('100%')

      Row({ space: 6 }) {
        ForEach(this.scenario.typeTags(), (tag: string) => {
          this.typeTag(tag)
        }, (tag: string) => tag)
        Blank()
        if (!this.scenario.builtin) {
          Text('编辑')
            .fontSize(12)
            .fontColor(Palette.DEC_ACCENT)
            .padding({ left: 8, right: 8, top: 3, bottom: 3 })
            .onClick(() => {
              this.onEdit();
            })
          if (this.onDelete !== null) {
            Text('删除')
              .fontSize(12)
              .fontColor(Palette.DANGER)
              .padding({ left: 8, right: 8, top: 3, bottom: 3 })
              .onClick(() => {
                if (this.onDelete !== null) {
                  this.onDelete();
                }
              })
          }
        }
      }
      .width('100%')
    }
    .width('100%')
    .padding(14)
    .backgroundColor(Palette.CARD_BG)
    .borderRadius(14)
    .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })
    .onClick(() => {
      this.onOpen();
    })
  }
}
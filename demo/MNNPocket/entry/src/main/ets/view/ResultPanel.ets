/**
 * Result cards for one decision outcome: probability bars for choice,
 * a yes/no gauge for noul, weighted score + distribution for score.
 */

import { AnswerCard, DecisionOutcome, ProbEntry, formatPercent, typeLabel } from '../model/DecisionTypes';
import { Palette, typeTagColor } from '../common/Theme';

@Component
export struct ResultPanel {
  outcome: DecisionOutcome = new DecisionOutcome();

  @Builder
  cardHeader(card: AnswerCard) {
    Row({ space: 8 }) {
      Text(card.id)
        .fontSize(13.5)
        .fontWeight(FontWeight.Bold)
        .fontColor(Palette.TEXT_PRIMARY)
        .layoutWeight(1)
        .maxLines(1)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
      Text(`${card.type} · ${typeLabel(card.type)}`)
        .fontSize(10)
        .fontColor(typeTagColor(card.type))
        .padding({ left: 7, right: 7, top: 2.5, bottom: 2.5 })
        .backgroundColor(Palette.CHIP_BG)
        .borderRadius(9)
    }
    .width('100%')
  }

  @Builder
  probRow(entry: ProbEntry, accent: string) {
    Row({ space: 8 }) {
      Text(entry.label)
        .fontSize(12)
        .fontColor(entry.winner ? Palette.TEXT_PRIMARY : Palette.TEXT_SECONDARY)
        .fontWeight(entry.winner ? FontWeight.Bold : FontWeight.Normal)
        .width(108)
        .maxLines(2)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
      Stack({ alignContent: Alignment.Start }) {
        Row()
          .width('100%')
          .height(8)
          .borderRadius(4)
          .backgroundColor(Palette.DEC_BAR_TRACK)
        Row()
          .width(`${Math.max(entry.probability * 100, 1.5)}%`)
          .height(8)
          .borderRadius(4)
          .backgroundColor(entry.winner ? accent : '#B9BEC9')
      }
      .layoutWeight(1)
      Text(formatPercent(entry.probability))
        .fontSize(11.5)
        .fontColor(entry.winner ? accent : Palette.TEXT_MUTED)
        .fontWeight(entry.winner ? FontWeight.Bold : FontWeight.Normal)
        .width(52)
        .textAlign(TextAlign.End)
    }
    .width('100%')
    .padding({ top: 5, bottom: 5, left: 6, right: 6 })
    .borderRadius(8)
    .backgroundColor(entry.winner ? Palette.DEC_WINNER_BG : Color.Transparent)
  }

  @Builder
  choiceCard(card: AnswerCard) {
    Column({ space: 4 }) {
      this.cardHeader(card)
      Row({ space: 5 }) {
        Text('✓')
          .fontSize(13)
          .fontColor(Palette.DEC_ACCENT)
        Text(`决策：${card.choice}`)
          .fontSize(13)
          .fontWeight(FontWeight.Bold)
          .fontColor(Palette.DEC_ACCENT)
      }
      .width('100%')
      .margin({ top: 2, bottom: 2 })
      ForEach(card.entries, (entry: ProbEntry) => {
        this.probRow(entry, Palette.DEC_ACCENT)
      }, (entry: ProbEntry) => entry.key)
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
    .padding(14)
    .backgroundColor(Palette.CARD_BG)
    .borderRadius(14)
    .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })
  }

  @Builder
  noulCard(card: AnswerCard) {
    Column({ space: 10 }) {
      this.cardHeader(card)
      Row({ space: 10 }) {
        Text(card.decision ? '是' : '否')
          .fontSize(26)
          .fontWeight(FontWeight.Bold)
          .fontColor(card.decision ? Palette.DEC_ACCENT : Palette.TEXT_PRIMARY)
        Column({ space: 2 }) {
          Text(`判定结论 · 置信 ${formatPercent(card.decision ? card.noul : 1 - card.noul)}`)
            .fontSize(12)
            .fontColor(Palette.TEXT_SECONDARY)
          Text(`P(是) = ${formatPercent(card.noul)}`)
            .fontSize(11)
            .fontColor(Palette.TEXT_MUTED)
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
      }
      .width('100%')

      // "是" segment grows from the LEFT; labels below read 左是右否.
      Stack({ alignContent: Alignment.Start }) {
        Row()
          .width('100%')
          .height(10)
          .borderRadius(5)
          .backgroundColor('#E4E6EB')
        Row()
          .width(`${Math.max(card.noul * 100, 1.5)}%`)
          .height(10)
          .borderRadius(5)
          .linearGradient({
            angle: 90,
            colors: [[Palette.DEC_GRADIENT_START, 0.0], [Palette.DEC_GRADIENT_END, 1.0]]
          })
      }
      .width('100%')

      Row() {
        Text(`是 ${formatPercent(card.noul)}`)
          .fontSize(11)
          .fontColor(Palette.DEC_ACCENT)
        Blank()
        Text(`否 ${formatPercent(1 - card.noul)}`)
          .fontSize(11)
          .fontColor(Palette.TEXT_MUTED)
      }
      .width('100%')
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
    .padding(14)
    .backgroundColor(Palette.CARD_BG)
    .borderRadius(14)
    .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })
  }

  @Builder
  scoreCard(card: AnswerCard) {
    Column({ space: 10 }) {
      this.cardHeader(card)
      Row({ space: 10 }) {
        Text(card.score.toFixed(2))
          .fontSize(26)
          .fontWeight(FontWeight.Bold)
          .fontColor(Palette.TAG_SCORE)
        Text(`/ ${card.entries.length - 1}`)
          .fontSize(13)
          .fontColor(Palette.TEXT_MUTED)
        Blank()
        Text('概率加权分')
          .fontSize(11)
          .fontColor(Palette.TEXT_SECONDARY)
      }
      .width('100%')
      .alignItems(VerticalAlign.Bottom)

      ForEach(card.entries, (entry: ProbEntry) => {
        this.probRow(entry, Palette.TAG_SCORE)
      }, (entry: ProbEntry) => entry.key)
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
    .padding(14)
    .backgroundColor(Palette.CARD_BG)
    .borderRadius(14)
    .shadow({ radius: 8, color: '#0D000000', offsetX: 0, offsetY: 2 })
  }

  build() {
    Column({ space: 12 }) {
      ForEach(this.outcome.cards, (card: AnswerCard) => {
        if (card.type === 'choice') {
          this.choiceCard(card)
        } else if (card.type === 'noul') {
          this.noulCard(card)
        } else {
          this.scoreCard(card)
        }
      }, (card: AnswerCard) => card.id)
    }
    .width('100%')
  }
}
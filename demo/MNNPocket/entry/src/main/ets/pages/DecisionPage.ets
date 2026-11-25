/**
 * Decision mode container: scenario gallery -> runner -> result panel,
 * plus the custom scenario editor. All state lives in DecisionViewModel.
 */

import { common } from '@kit.AbilityKit';
import { DecStage, DecisionViewModel } from '../model/DecisionViewModel';
import { QuestionDraft, Scenario } from '../model/DecisionTypes';
import { ScenarioCard } from '../view/ScenarioCard';
import { ResultPanel } from '../view/ResultPanel';
import { ScenarioEditor } from '../view/ScenarioEditor';
import { Palette, typeTagColor } from '../common/Theme';

@Component
export struct DecisionPage {
  @ObjectLink @Watch('onVmChange') vm: DecisionViewModel;
  private tickTimer: number = -1;
  private tickStart: number = 0;

  onVmChange(): void {
    if (!this.vm.running) {
      this.stopTick();
    }
  }

  aboutToAppear(): void {
    const context = getContext(this) as common.UIAbilityContext;
    this.vm.loadCustoms(context);
    this.vm.ensureLoaded(context);
  }

  aboutToDisappear(): void {
    this.stopTick();
  }

  private startTick(): void {
    this.stopTick();
    this.tickStart = Date.now();
    this.tickTimer = setInterval(() => {
      this.vm.elapsedMs = Date.now() - this.tickStart;
    }, 80);
  }

  private stopTick(): void {
    if (this.tickTimer !== -1) {
      clearInterval(this.tickTimer);
      this.tickTimer = -1;
    }
  }

  private toast(message: string): void {
    try {
      this.getUIContext().getPromptAction().showToast({ message: message, duration: 2400 });
    } catch (error) {
      console.warn(`toast failed: ${JSON.stringify(error)}`);
    }
  }

  private runActive(): void {
    const problem = this.vm.run();
    if (problem !== null) {
      this.toast(problem);
      return;
    }
    this.startTick();
  }

  // -------------------------------------------------------------------------
  // loading / error
  // -------------------------------------------------------------------------

  @Builder
  loadingPanel() {
    Column({ space: 18 }) {
      LoadingProgress()
        .width(46)
        .height(46)
        .color(Palette.DEC_ACCENT)
      Column({ space: 6 }) {
        Text(this.vm.stage === DecStage.COPYING ? '决策模型准备中' : '决策模型加载中')
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor(Palette.TEXT_PRIMARY)
        Text(this.vm.stageText)
          .fontSize(12.5)
          .fontColor(Palette.TEXT_SECONDARY)
          .textAlign(TextAlign.Center)
      }
      if (this.vm.stage === DecStage.COPYING) {
        Column({ space: 8 }) {
          Progress({ value: this.vm.progress, total: 100, type: ProgressType.Linear })
            .width(220)
            .color(Palette.DEC_ACCENT)
            .backgroundColor('#E4E7EB')
            .borderRadius(4)
          Text(`${this.vm.progress}% · 决策模型约 341 MB，仅首次写入`)
            .fontSize(11)
            .fontColor(Palette.TEXT_MUTED)
        }
      } else {
        Text('W8 量化 · 单次前向 · 通常 1–3 秒')
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
      Text('决策模型没能启动')
        .fontSize(17)
        .fontWeight(FontWeight.Medium)
        .fontColor(Palette.TEXT_PRIMARY)
      Text(this.vm.errorText)
        .fontSize(12)
        .fontColor(Palette.TEXT_SECONDARY)
        .textAlign(TextAlign.Center)
        .maxLines(6)
        .padding({ left: 24, right: 24 })
      Button('重新加载')
        .fontSize(14)
        .height(38)
        .padding({ left: 22, right: 22 })
        .backgroundColor(Palette.DEC_ACCENT)
        .onClick(() => {
          this.vm.retry(getContext(this) as common.UIAbilityContext);
        })
    }
    .width('100%')
    .layoutWeight(1)
    .justifyContent(FlexAlign.Center)
  }

  // -------------------------------------------------------------------------
  // gallery
  // -------------------------------------------------------------------------

  @Builder
  gallery() {
    Column() {
      Scroll() {
        Column({ space: 12 }) {
          Column({ space: 6 }) {
            Row({ space: 8 }) {
              Text('⚖️')
                .fontSize(20)
              Text('单次前向 · 结构化决策')
                .fontSize(16)
                .fontWeight(FontWeight.Bold)
                .fontColor(Color.White)
            }
            Text('不生成文字：输入场景状态 + 问题，一次推理返回选项与校准概率。覆盖 choice / noul / score 三种题型。')
              .fontSize(12)
              .fontColor('#E0E2FF')
              .lineHeight(18)
            Text(`MNN-Pocket-0.3B-Dec · W8 · 加载 ${this.vm.loadMs} ms`)
              .fontSize(10.5)
              .fontColor('#B9BDFF')
          }
          .alignItems(HorizontalAlign.Start)
          .width('100%')
          .padding(16)
          .borderRadius(16)
          .linearGradient({
            angle: 135,
            colors: [[Palette.DEC_GRADIENT_START, 0.0], [Palette.DEC_GRADIENT_END, 1.0]]
          })

          ForEach(this.vm.allScenarios(), (scenario: Scenario) => {
            ScenarioCard({
              scenario: scenario,
              onOpen: () => {
                this.vm.openScenario(scenario);
              },
              onEdit: () => {
                this.vm.openEditor(scenario);
              },
              onDelete: scenario.builtin ? null : () => {
                this.vm.deleteCustom(getContext(this) as common.UIAbilityContext, scenario);
              }
            })
          }, (scenario: Scenario) => scenario.id)

          Text('＋ 自定义场景')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor(Palette.DEC_ACCENT)
            .width('100%')
            .textAlign(TextAlign.Center)
            .padding({ top: 13, bottom: 13 })
            .backgroundColor(Palette.DEC_ACCENT_SOFT)
            .borderRadius(14)
            .onClick(() => {
              this.vm.openEditor(null);
            })
        }
        .width('100%')
        .padding({ left: 14, right: 14, top: 12, bottom: 12 })
      }
      .layoutWeight(1)
      .scrollBar(BarState.Off)
      .align(Alignment.Top)
      .edgeEffect(EdgeEffect.Spring)

      // Pinned bottom action: always reachable regardless of scroll position.
      Row() {
        Text('＋ 自定义场景')
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor(Palette.DEC_ACCENT)
          .width('100%')
          .textAlign(TextAlign.Center)
          .padding({ top: 12, bottom: 12 })
          .backgroundColor(Palette.DEC_ACCENT_SOFT)
          .borderRadius(14)
          .onClick(() => {
            this.vm.openEditor(null);
          })
      }
      .width('100%')
      .padding({ left: 14, right: 14, top: 8, bottom: 10 })
      .backgroundColor(Palette.HEADER_BG)
    }
    .width('100%')
    .layoutWeight(1)
  }

  // -------------------------------------------------------------------------
  // runner
  // -------------------------------------------------------------------------

  @Builder
  questionPreview(scenario: Scenario) {
    Column({ space: 8 }) {
      ForEach(scenario.questions, (q: QuestionDraft) => {
        Row({ space: 8 }) {
          Text(q.type)
            .fontSize(10)
            .fontColor(typeTagColor(q.type))
            .padding({ left: 6, right: 6, top: 2, bottom: 2 })
            .backgroundColor(Palette.CHIP_BG)
            .borderRadius(8)
          Column({ space: 1 }) {
            Text(q.id)
              .fontSize(12.5)
              .fontWeight(FontWeight.Medium)
              .fontColor(Palette.TEXT_PRIMARY)
            Text(q.instructions)
              .fontSize(11.5)
              .fontColor(Palette.TEXT_SECONDARY)
              .maxLines(2)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)
        }
        .width('100%')
      }, (q: QuestionDraft) => q.id)
    }
    .width('100%')
    .padding(12)
    .backgroundColor(Palette.CARD_BG)
    .borderRadius(12)
  }

  @Builder
  runner() {
    Scroll() {
      Column({ space: 12 }) {
        Row({ space: 10 }) {
          Text('‹ 场景库')
            .fontSize(14)
            .fontColor(Palette.DEC_ACCENT)
            .onClick(() => {
              this.stopTick();
              this.vm.backToGallery();
            })
          Text(this.vm.active !== null ? this.vm.active.title : '')
            .fontSize(15.5)
            .fontWeight(FontWeight.Bold)
            .fontColor(Palette.TEXT_PRIMARY)
            .layoutWeight(1)
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
          Text('编辑副本')
            .fontSize(12.5)
            .fontColor(Palette.DEC_ACCENT)
            .onClick(() => {
              this.vm.openEditor(this.vm.active);
            })
        }
        .width('100%')

        if (this.vm.active !== null) {
          Column({ space: 5 }) {
            Text('场景状态')
              .fontSize(12)
              .fontWeight(FontWeight.Medium)
              .fontColor(Palette.TEXT_SECONDARY)
            Text(this.vm.active.state)
              .fontSize(13.5)
              .fontColor(Palette.TEXT_PRIMARY)
              .lineHeight(20)
              .copyOption(CopyOptions.LocalDevice)
          }
          .alignItems(HorizontalAlign.Start)
          .width('100%')
          .padding(14)
          .backgroundColor(Palette.CARD_BG)
          .borderRadius(12)

          this.questionPreview(this.vm.active)
        }

        Button(this.vm.running ? `推理中 · ${this.vm.elapsedMs} ms` : '运行决策')
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .height(46)
          .width('100%')
          .backgroundColor(this.vm.running ? Palette.DEC_ACCENT_DISABLED : Palette.DEC_ACCENT)
          .enabled(!this.vm.running)
          .onClick(() => {
            this.runActive();
          })

        if (this.vm.runError.length > 0) {
          Text(this.vm.runError)
            .fontSize(12.5)
            .fontColor(Palette.DANGER)
            .width('100%')
            .padding(10)
            .backgroundColor('#FDECEC')
            .borderRadius(10)
        }

        if (this.vm.outcome !== null) {
          ResultPanel({ outcome: this.vm.outcome })
          Row({ space: 6 }) {
            Text(`输入 ${this.vm.outcome.inputTokens} tok`)
              .fontSize(10.5)
              .fontColor(Palette.TEXT_MUTED)
            Text(`前向 ${this.vm.outcome.forwardMs} ms`)
              .fontSize(10.5)
              .fontColor(Palette.TEXT_MUTED)
            Text(`编码 ${this.vm.outcome.encodeMs} ms`)
              .fontSize(10.5)
              .fontColor(Palette.TEXT_MUTED)
            Text(`T=${this.vm.outcome.temperature.toFixed(3)}`)
              .fontSize(10.5)
              .fontColor(Palette.TEXT_MUTED)
          }
          .width('100%')
          .justifyContent(FlexAlign.Center)
          .padding({ top: 6, bottom: 6 })
          .backgroundColor(Palette.CHIP_BG)
          .borderRadius(10)
        }
      }
      .width('100%')
      .padding({ left: 14, right: 14, top: 10, bottom: 18 })
    }
    .layoutWeight(1)
    .scrollBar(BarState.Off)
    .align(Alignment.Top)
    .edgeEffect(EdgeEffect.Spring)
  }

  build() {
    Column() {
      if (this.vm.stage === DecStage.FAILED) {
        this.errorPanel()
      } else if (this.vm.stage !== DecStage.READY) {
        this.loadingPanel()
      } else if (this.vm.view === 'editor' && this.vm.editing !== null) {
        ScenarioEditor({
          source: this.vm.editing,
          onSave: (scenario: Scenario) => {
            const context = getContext(this) as common.UIAbilityContext;
            this.vm.saveAndRun(context, scenario).then((problem: string | null) => {
              if (problem !== null) {
                this.toast(problem);
              } else {
                this.startTick();
              }
            });
          },
          onCancel: () => {
            this.vm.closeEditor();
          }
        })
      } else if (this.vm.view === 'runner' && this.vm.active !== null) {
        this.runner()
      } else {
        this.gallery()
      }
    }
    .width('100%')
    .height('100%')
  }
}
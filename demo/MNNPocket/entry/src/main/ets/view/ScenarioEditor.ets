/**
 * Custom scenario editor: form mode (state + question builder) and an
 * advanced JSON mode editing the raw wire request. Saving runs the scenario
 * immediately and persists it into the "我的场景" group.
 */

import { ChoiceOption, QuestionDraft, Scenario, buildRequestJson, validateScenario } from '../model/DecisionTypes';
import { Palette } from '../common/Theme';

function cloneQuestions(source: QuestionDraft[]): QuestionDraft[] {
  const out: QuestionDraft[] = [];
  for (const q of source) {
    const draft = new QuestionDraft(q.id, q.type, q.instructions);
    for (const option of q.choices) {
      draft.choices.push(new ChoiceOption(option.key, option.description));
    }
    draft.noulFalse = q.noulFalse;
    draft.noulTrue = q.noulTrue;
    for (const label of q.scale) {
      draft.scale.push(label);
    }
    out.push(draft);
  }
  return out;
}

interface WireChoice {
  key: string;
  description: string;
}

interface WireQuestion {
  type: string;
  instructions: string;
  criteria?: Record<string, string> | string[];
}

interface WireRequest {
  state: string;
  questions: Record<string, WireQuestion>;
}

@Component
export struct ScenarioEditor {
  source: Scenario = new Scenario('', '', '', false, '', []);
  onSave: (scenario: Scenario) => void = () => {
  };
  onCancel: () => void = () => {
  };

  @State title: string = '';
  @State stateText: string = '';
  @State questions: QuestionDraft[] = [];
  @State jsonMode: boolean = false;
  @State jsonText: string = '';
  @State problem: string = '';

  aboutToAppear(): void {
    this.title = this.source.title;
    this.stateText = this.source.state;
    this.questions = cloneQuestions(this.source.questions);
  }

  private touch(): void {
    this.questions = cloneQuestions(this.questions);
    this.problem = '';
  }

  private addQuestion(): void {
    const next = cloneQuestions(this.questions);
    const draft = new QuestionDraft(`question_${next.length + 1}`, 'choice', '应该怎么处理这个请求？');
    draft.choices.push(new ChoiceOption('option_a', ''));
    draft.choices.push(new ChoiceOption('option_b', ''));
    next.push(draft);
    this.questions = next;
  }

  private removeQuestion(index: number): void {
    const next = cloneQuestions(this.questions);
    next.splice(index, 1);
    this.questions = next;
  }

  private setType(index: number, type: string): void {
    const next = cloneQuestions(this.questions);
    const q = next[index];
    q.type = type;
    if (type === 'choice' && q.choices.length === 0) {
      q.choices.push(new ChoiceOption('option_a', ''));
      q.choices.push(new ChoiceOption('option_b', ''));
    }
    if (type === 'score' && q.scale.length === 0) {
      q.scale.push('低');
      q.scale.push('中');
      q.scale.push('高');
    }
    this.questions = next;
  }

  private buildScenario(): Scenario {
    return new Scenario(this.source.id, this.title.trim().length > 0 ? this.title.trim() : '我的场景',
      '自定义场景', false, this.stateText, cloneQuestions(this.questions));
  }

  private switchJsonMode(): void {
    if (!this.jsonMode) {
      const request = buildRequestJson(this.stateText, this.questions, 0);
      try {
        this.jsonText = JSON.stringify(JSON.parse(request), null, 2);
      } catch (error) {
        this.jsonText = request;
      }
    } else {
      const parsed = this.parseJsonDraft();
      if (parsed !== null) {
        this.stateText = parsed.state;
        this.questions = parsed.questions;
      }
    }
    this.jsonMode = !this.jsonMode;
  }

  private parseJsonDraft(): Scenario | null {
    try {
      const raw = JSON.parse(this.jsonText) as WireRequest;
      if (typeof raw.state !== 'string') {
        this.problem = 'JSON：state 必须是字符串';
        return null;
      }
      const questions: QuestionDraft[] = [];
      const ids = Object.keys(raw.questions);
      for (const id of ids) {
        const wire = raw.questions[id];
        const draft = new QuestionDraft(id, wire.type, wire.instructions ?? '');
        if (wire.type === 'choice') {
          const criteria = wire.criteria as Record<string, string>;
          if (criteria === undefined || criteria === null) {
            this.problem = `JSON：choice 问题 ${id} 缺少 criteria 对象`;
            return null;
          }
          const keys = Object.keys(criteria);
          for (const key of keys) {
            draft.choices.push(new ChoiceOption(key, criteria[key] ?? ''));
          }
        } else if (wire.type === 'noul') {
          const criteria = wire.criteria as Record<string, string>;
          if (criteria !== undefined && criteria !== null) {
            draft.noulFalse = criteria['false'] ?? '';
            draft.noulTrue = criteria['true'] ?? '';
          }
        } else if (wire.type === 'score') {
          const criteria = wire.criteria as string[];
          if (criteria === undefined || criteria === null) {
            this.problem = `JSON：score 问题 ${id} 缺少 criteria 列表`;
            return null;
          }
          for (const label of criteria) {
            draft.scale.push(label);
          }
        } else {
          this.problem = `JSON：未知题型 ${wire.type}`;
          return null;
        }
        questions.push(draft);
      }
      return new Scenario(this.source.id, this.title, '自定义场景', false, raw.state, questions);
    } catch (error) {
      this.problem = `JSON 解析失败：${error}`;
      return null;
    }
  }

  private save(): void {
    let scenario: Scenario;
    if (this.jsonMode) {
      const parsed = this.parseJsonDraft();
      if (parsed === null) {
        return;
      }
      scenario = parsed;
      scenario.title = this.title.trim().length > 0 ? this.title.trim() : '我的场景';
    } else {
      scenario = this.buildScenario();
    }
    const problem = validateScenario(scenario.state, scenario.questions);
    if (problem !== null) {
      this.problem = problem;
      return;
    }
    this.onSave(scenario);
  }

  // -------------------------------------------------------------------------
  // form pieces
  // -------------------------------------------------------------------------

  @Builder
  fieldLabel(text: string) {
    Text(text)
      .fontSize(12)
      .fontWeight(FontWeight.Medium)
      .fontColor(Palette.TEXT_SECONDARY)
      .margin({ bottom: 5 })
  }

  @Builder
  typeChip(index: number, type: string, label: string) {
    Text(label)
      .fontSize(12)
      .fontColor(this.questions[index].type === type ? Color.White : Palette.TEXT_SECONDARY)
      .padding({ left: 10, right: 10, top: 5, bottom: 5 })
      .borderRadius(12)
      .backgroundColor(this.questions[index].type === type ? Palette.DEC_ACCENT : Palette.CHIP_BG)
      .onClick(() => {
        this.setType(index, type);
      })
  }

  @Builder
  choiceEditor(index: number) {
    Column({ space: 6 }) {
      ForEach(this.questions[index].choices, (option: ChoiceOption, optIndex: number) => {
        Row({ space: 6 }) {
          TextInput({ text: option.key, placeholder: 'key' })
            .fontSize(13)
            .width(96)
            .backgroundColor(Palette.INPUT_BG)
            .borderRadius(8)
            .onChange((value: string) => {
              this.questions[index].choices[optIndex].key = value;
              this.touch();
            })
          TextInput({ text: option.description, placeholder: '候选描述（可空）' })
            .fontSize(13)
            .layoutWeight(1)
            .backgroundColor(Palette.INPUT_BG)
            .borderRadius(8)
            .onChange((value: string) => {
              this.questions[index].choices[optIndex].description = value;
              this.touch();
            })
          Text('✕')
            .fontSize(13)
            .fontColor(Palette.TEXT_MUTED)
            .padding(6)
            .onClick(() => {
              const next = cloneQuestions(this.questions);
              next[index].choices.splice(optIndex, 1);
              this.questions = next;
            })
        }
        .width('100%')
      }, (option: ChoiceOption, optIndex: number) => `${index}_${optIndex}_${option.key}`)
      Text('＋ 添加候选')
        .fontSize(12.5)
        .fontColor(Palette.DEC_ACCENT)
        .onClick(() => {
          const next = cloneQuestions(this.questions);
          next[index].choices.push(new ChoiceOption(`option_${next[index].choices.length + 1}`, ''));
          this.questions = next;
        })
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
  }

  @Builder
  noulEditor(index: number) {
    Column({ space: 6 }) {
      TextInput({ text: this.questions[index].noulFalse, placeholder: '“否”的判定描述（可空）' })
        .fontSize(13)
        .backgroundColor(Palette.INPUT_BG)
        .borderRadius(8)
        .onChange((value: string) => {
          this.questions[index].noulFalse = value;
          this.touch();
        })
      TextInput({ text: this.questions[index].noulTrue, placeholder: '“是”的判定描述（可空）' })
        .fontSize(13)
        .backgroundColor(Palette.INPUT_BG)
        .borderRadius(8)
        .onChange((value: string) => {
          this.questions[index].noulTrue = value;
          this.touch();
        })
    }
    .width('100%')
  }

  @Builder
  scoreEditor(index: number) {
    Column({ space: 6 }) {
      ForEach(this.questions[index].scale, (label: string, scaleIndex: number) => {
        Row({ space: 6 }) {
          Text(`${scaleIndex}`)
            .fontSize(12)
            .fontColor(Palette.TEXT_MUTED)
            .width(18)
          TextInput({ text: label, placeholder: '量表标签' })
            .fontSize(13)
            .layoutWeight(1)
            .backgroundColor(Palette.INPUT_BG)
            .borderRadius(8)
            .onChange((value: string) => {
              this.questions[index].scale[scaleIndex] = value;
              this.touch();
            })
          Text('✕')
            .fontSize(13)
            .fontColor(Palette.TEXT_MUTED)
            .padding(6)
            .onClick(() => {
              const next = cloneQuestions(this.questions);
              next[index].scale.splice(scaleIndex, 1);
              this.questions = next;
            })
        }
        .width('100%')
      }, (label: string, scaleIndex: number) => `${index}_${scaleIndex}_${label}`)
      Text('＋ 添加一级')
        .fontSize(12.5)
        .fontColor(Palette.DEC_ACCENT)
        .onClick(() => {
          const next = cloneQuestions(this.questions);
          next[index].scale.push(`第 ${next[index].scale.length + 1} 级`);
          this.questions = next;
        })
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
  }

  @Builder
  questionCard(index: number) {
    Column({ space: 9 }) {
      Row({ space: 8 }) {
        TextInput({ text: this.questions[index].id, placeholder: '问题 ID' })
          .fontSize(13)
          .layoutWeight(1)
          .backgroundColor(Palette.INPUT_BG)
          .borderRadius(8)
          .onChange((value: string) => {
            this.questions[index].id = value;
            this.touch();
          })
        Text('删除')
          .fontSize(12)
          .fontColor(Palette.DANGER)
          .padding({ left: 8, right: 8, top: 5, bottom: 5 })
          .onClick(() => {
            this.removeQuestion(index);
          })
      }
      .width('100%')

      Row({ space: 6 }) {
        this.typeChip(index, 'choice', 'choice 多选一')
        this.typeChip(index, 'noul', 'noul 是否')
        this.typeChip(index, 'score', 'score 评分')
      }

      TextInput({ text: this.questions[index].instructions, placeholder: '判定说明（instructions）' })
        .fontSize(13)
        .backgroundColor(Palette.INPUT_BG)
        .borderRadius(8)
        .onChange((value: string) => {
          this.questions[index].instructions = value;
          this.touch();
        })

      if (this.questions[index].type === 'choice') {
        this.choiceEditor(index)
      } else if (this.questions[index].type === 'noul') {
        this.noulEditor(index)
      } else {
        this.scoreEditor(index)
      }
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
    .padding(12)
    .backgroundColor(Palette.CARD_BG)
    .borderRadius(12)
  }

  build() {
    Column() {
      Row({ space: 10 }) {
        Text('‹ 返回')
          .fontSize(14)
          .fontColor(Palette.DEC_ACCENT)
          .onClick(() => {
            this.onCancel();
          })
        Text(this.source.builtin ? '基于内置场景创建' : '编辑自定义场景')
          .fontSize(15)
          .fontWeight(FontWeight.Bold)
          .fontColor(Palette.TEXT_PRIMARY)
          .layoutWeight(1)
        Text(this.jsonMode ? '表单模式' : 'JSON 模式')
          .fontSize(12.5)
          .fontColor(Palette.DEC_ACCENT)
          .padding({ left: 10, right: 10, top: 5, bottom: 5 })
          .backgroundColor(Palette.DEC_ACCENT_SOFT)
          .borderRadius(12)
          .onClick(() => {
            this.switchJsonMode();
          })
      }
      .width('100%')
      .padding({ left: 14, right: 14, top: 10, bottom: 8 })

      Scroll() {
        Column({ space: 12 }) {
          TextInput({ text: this.title, placeholder: '场景名称' })
            .fontSize(14)
            .backgroundColor(Palette.CARD_BG)
            .borderRadius(10)
            .onChange((value: string) => {
              this.title = value;
            })

          if (this.jsonMode) {
            Column({ space: 5 }) {
              this.fieldLabel('请求 JSON（state + questions，协议与官方一致）')
              TextArea({ text: this.jsonText })
                .fontSize(12.5)
                .fontFamily('monospace')
                .height(360)
                .backgroundColor(Palette.CARD_BG)
                .borderRadius(10)
                .onChange((value: string) => {
                  this.jsonText = value;
                  this.problem = '';
                })
            }
            .alignItems(HorizontalAlign.Start)
            .width('100%')
          } else {
            Column({ space: 5 }) {
              this.fieldLabel('场景状态（state）：描述当前情况')
              TextArea({ text: this.stateText, placeholder: '例如：用户收到的商品尺码不对，要求换货……' })
                .fontSize(14)
                .height(110)
                .backgroundColor(Palette.CARD_BG)
                .borderRadius(10)
                .onChange((value: string) => {
                  this.stateText = value;
                  this.problem = '';
                })
            }
            .alignItems(HorizontalAlign.Start)
            .width('100%')

            ForEach(this.questions, (q: QuestionDraft, index: number) => {
              this.questionCard(index)
            }, (q: QuestionDraft, index: number) => `${index}_${q.id}_${q.type}`)

            Text('＋ 添加问题')
              .fontSize(13.5)
              .fontColor(Palette.DEC_ACCENT)
              .width('100%')
              .textAlign(TextAlign.Center)
              .padding({ top: 11, bottom: 11 })
              .backgroundColor(Palette.DEC_ACCENT_SOFT)
              .borderRadius(12)
              .onClick(() => {
                this.addQuestion();
              })
          }

          if (this.problem.length > 0) {
            Text(this.problem)
              .fontSize(12.5)
              .fontColor(Palette.DANGER)
              .width('100%')
              .padding(10)
              .backgroundColor('#FDECEC')
              .borderRadius(10)
          }
        }
        .width('100%')
        .padding({ left: 14, right: 14, bottom: 16 })
      }
      .layoutWeight(1)
      .scrollBar(BarState.Off)
      .align(Alignment.Top)

      Row({ space: 10 }) {
        Button('取消')
          .fontSize(14)
          .height(42)
          .layoutWeight(1)
          .backgroundColor(Palette.CHIP_BG)
          .fontColor(Palette.TEXT_PRIMARY)
          .onClick(() => {
            this.onCancel();
          })
        Button('保存并运行')
          .fontSize(14)
          .height(42)
          .layoutWeight(2)
          .backgroundColor(Palette.DEC_ACCENT)
          .onClick(() => {
            this.save();
          })
      }
      .width('100%')
      .padding({ left: 14, right: 14, top: 8, bottom: 10 })
      .backgroundColor(Palette.HEADER_BG)
    }
    .width('100%')
    .height('100%')
  }
}
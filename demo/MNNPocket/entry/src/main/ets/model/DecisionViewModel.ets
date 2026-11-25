/**
 * Decision-mode state machine, shared between Index (event routing) and
 * DecisionPage (rendering). Observed first-level properties drive the UI.
 */

import { common } from '@kit.AbilityKit';
import { LlmRuntime } from './LlmRuntime';
import { ModelManager } from './ModelManager';
import { ScenarioStore } from './ScenarioStore';
import {
  ChoiceOption,
  DecisionOutcome,
  QuestionDraft,
  Scenario,
  buildRequestJson,
  parseOutcome,
  validateScenario
} from './DecisionTypes';

export enum DecStage {
  IDLE = 0,
  COPYING = 1,
  LOADING = 2,
  READY = 3,
  FAILED = 4
}

export function cloneScenario(source: Scenario, newId: string, builtin: boolean): Scenario {
  const questions: QuestionDraft[] = [];
  for (const q of source.questions) {
    const draft = new QuestionDraft(q.id, q.type, q.instructions);
    for (const option of q.choices) {
      draft.choices.push(new ChoiceOption(option.key, option.description));
    }
    draft.noulFalse = q.noulFalse;
    draft.noulTrue = q.noulTrue;
    for (const label of q.scale) {
      draft.scale.push(label);
    }
    questions.push(draft);
  }
  return new Scenario(newId, source.title, source.subtitle, builtin, source.state, questions);
}

@Observed
export class DecisionViewModel {
  private static readonly SINGLETON: DecisionViewModel = new DecisionViewModel();

  static get(): DecisionViewModel {
    return DecisionViewModel.SINGLETON;
  }

  stage: DecStage = DecStage.IDLE;
  stageText: string = '';
  progress: number = 0;
  errorText: string = '';
  loadMs: number = 0;

  view: string = 'gallery'; // gallery | runner | editor
  customs: Scenario[] = [];
  active: Scenario | null = null;
  editing: Scenario | null = null;

  running: boolean = false;
  elapsedMs: number = 0;
  outcome: DecisionOutcome | null = null;
  runError: string = '';

  /** 0 = checkpoint default (1.1112); >0 overrides the calibrated temperature. */
  temperatureOverride: number = 0;
  maxTokens: number = 2048;
  threads: number = 4;

  private runtime: LlmRuntime = LlmRuntime.get();
  private bootstrapping: boolean = false;

  allScenarios(): Scenario[] {
    const list: Scenario[] = [];
    for (const scenario of ScenarioStore.builtinScenarios()) {
      list.push(scenario);
    }
    for (const scenario of this.customs) {
      list.push(scenario);
    }
    return list;
  }

  async loadCustoms(context: common.UIAbilityContext): Promise<void> {
    this.customs = await ScenarioStore.loadCustom(context);
  }

  /** First entry into decision mode: copy the package, then load the graph. */
  async ensureLoaded(context: common.UIAbilityContext): Promise<void> {
    if (this.runtime.decisionReady()) {
      this.stage = DecStage.READY;
      return;
    }
    if (this.bootstrapping) {
      return;
    }
    this.bootstrapping = true;
    this.stage = DecStage.COPYING;
    this.stageText = '首次使用：正在写入决策模型文件…';
    this.progress = 0;
    this.errorText = '';
    try {
      const dir = await ModelManager.ensureDecision(context, (copied: number, total: number) => {
        const percent = total > 0 ? Math.floor(copied * 100 / total) : 0;
        this.progress = percent;
        if (percent < 100) {
          this.stage = DecStage.COPYING;
          this.stageText = `首次使用：正在写入决策模型文件 ${percent}%`;
        } else {
          this.stage = DecStage.LOADING;
          this.stageText = '决策模型就绪，正在加载到内存…';
        }
      });
      this.stage = DecStage.LOADING;
      this.stageText = '决策模型就绪，正在加载到内存…';
      this.runtime.loadDecision(dir, this.threads, this.maxTokens);
    } catch (error) {
      this.stage = DecStage.FAILED;
      this.errorText = `${error}`;
    } finally {
      this.bootstrapping = false;
    }
  }

  retry(context: common.UIAbilityContext): void {
    this.stage = DecStage.IDLE;
    this.ensureLoaded(context);
  }

  release(): void {
    if (this.stage === DecStage.READY && !this.running) {
      this.runtime.releaseDecision();
    }
  }

  // -------------------------------------------------------------------------
  // navigation
  // -------------------------------------------------------------------------

  openScenario(scenario: Scenario): void {
    this.active = scenario;
    this.outcome = null;
    this.runError = '';
    this.view = 'runner';
  }

  backToGallery(): void {
    this.view = 'gallery';
    this.running = false;
  }

  openEditor(source: Scenario | null): void {
    if (source === null) {
      const draft = new QuestionDraft('question_1', 'choice', '应该怎么处理这个请求？');
      draft.choices.push(new ChoiceOption('option_a', '选项 A 的描述'));
      draft.choices.push(new ChoiceOption('option_b', '选项 B 的描述'));
      this.editing = new Scenario(`custom_${Date.now()}`, '我的场景', '自定义场景', false, '', [draft]);
    } else {
      this.editing = cloneScenario(source, source.builtin ? `custom_${Date.now()}` : source.id, false);
    }
    this.view = 'editor';
  }

  closeEditor(): void {
    this.editing = null;
    this.view = this.active !== null ? 'runner' : 'gallery';
  }

  // -------------------------------------------------------------------------
  // running
  // -------------------------------------------------------------------------

  run(): string | null {
    if (this.active === null || this.stage !== DecStage.READY || this.running) {
      return null;
    }
    const problem = validateScenario(this.active.state, this.active.questions);
    if (problem !== null) {
      return problem;
    }
    const request = buildRequestJson(this.active.state, this.active.questions, this.temperatureOverride);
    this.outcome = null;
    this.runError = '';
    this.running = true;
    this.elapsedMs = 0;
    this.runtime.runDecision(request);
    return null;
  }

  /** Runs an edited scenario immediately after saving it. */
  async saveAndRun(context: common.UIAbilityContext, scenario: Scenario): Promise<string | null> {
    const problem = validateScenario(scenario.state, scenario.questions);
    if (problem !== null) {
      return problem;
    }
    const replaced: Scenario[] = [];
    let found = false;
    for (const item of this.customs) {
      if (item.id === scenario.id) {
        replaced.push(scenario);
        found = true;
      } else {
        replaced.push(item);
      }
    }
    if (!found) {
      replaced.push(scenario);
    }
    this.customs = replaced;
    await ScenarioStore.saveCustom(context, replaced);
    this.editing = null;
    this.active = scenario;
    this.outcome = null;
    this.view = 'runner';
    return this.run();
  }

  async deleteCustom(context: common.UIAbilityContext, scenario: Scenario): Promise<void> {
    const kept: Scenario[] = [];
    for (const item of this.customs) {
      if (item.id !== scenario.id) {
        kept.push(item);
      }
    }
    this.customs = kept;
    await ScenarioStore.saveCustom(context, kept);
    if (this.active !== null && this.active.id === scenario.id) {
      this.active = null;
      this.view = 'gallery';
    }
  }

  // -------------------------------------------------------------------------
  // native events (routed here by Index)
  // -------------------------------------------------------------------------

  handleEvent(name: string, payload: string): void {
    switch (name) {
      case 'dec_progress':
        this.stageText = '正在初始化决策引擎…';
        break;
      case 'dec_ready': {
        this.stage = DecStage.READY;
        try {
          const info = JSON.parse(payload) as DecReadyJson;
          this.loadMs = info.loadMs;
        } catch (error) {
          this.loadMs = 0;
        }
        break;
      }
      case 'dec_start':
        this.running = true;
        this.elapsedMs = 0;
        break;
      case 'dec_done': {
        this.running = false;
        if (this.active !== null) {
          this.outcome = parseOutcome(payload, this.active.questions);
          if (this.outcome === null) {
            this.runError = '决策结果解析失败';
          }
        }
        break;
      }
      case 'dec_unloaded':
        this.stage = DecStage.IDLE;
        this.outcome = null;
        break;
      case 'dec_error': {
        this.running = false;
        const message = parseDecError(payload);
        if (this.stage === DecStage.LOADING || this.stage === DecStage.COPYING ||
          this.stage === DecStage.IDLE) {
          this.stage = DecStage.FAILED;
          this.errorText = message;
        } else {
          this.runError = message;
        }
        break;
      }
      default:
        break;
    }
  }
}

interface DecReadyJson {
  model: string;
  variant: string;
  loaded: boolean;
  loadMs: number;
  maxTokens: number;
  threads: number;
  temperature: number;
}

interface DecErrorJson {
  stage: string;
  message: string;
}

function parseDecError(json: string): string {
  try {
    return (JSON.parse(json) as DecErrorJson).message;
  } catch (error) {
    return json;
  }
}
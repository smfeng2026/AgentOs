/**
 * Scenario library: seven built-in demos covering all three question types,
 * plus user-defined scenarios persisted in preferences.
 */

import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';
import { ChoiceOption, QuestionDraft, Scenario } from './DecisionTypes';

function choice(id: string, instructions: string, options: ChoiceOption[]): QuestionDraft {
  const q = new QuestionDraft(id, 'choice', instructions);
  q.choices = options;
  return q;
}

function noul(id: string, instructions: string, noDesc: string, yesDesc: string): QuestionDraft {
  const q = new QuestionDraft(id, 'noul', instructions);
  q.noulFalse = noDesc;
  q.noulTrue = yesDesc;
  return q;
}

function score(id: string, instructions: string, scale: string[]): QuestionDraft {
  const q = new QuestionDraft(id, 'score', instructions);
  q.scale = scale;
  return q;
}

function opt(key: string, description: string): ChoiceOption {
  return new ChoiceOption(key, description);
}

export class ScenarioStore {
  private static builtins: Scenario[] | null = null;

  static builtinScenarios(): Scenario[] {
    if (ScenarioStore.builtins !== null) {
      return ScenarioStore.builtins;
    }
    const list: Scenario[] = [];

    list.push(new Scenario('support-routing', '客服工单路由',
      '重复扣款投诉 → 部门分流 + 人工判定 + 紧急度（官方示例）', true,
      'I was charged twice for the same order. Please help.',
      [
        choice('department', 'Which team should handle this request?',
          [opt('billing', 'Charges and payment problems'),
            opt('shipping', 'Delivery and lost packages'),
            opt('returns', 'Returns and refunds')]),
        noul('needs_human', 'Does this request require a human agent?', '', ''),
        score('urgency', 'How urgent is this request?',
          ['No time pressure', 'Routine', 'Handle today'])
      ]));

    list.push(new Scenario('exchange-cn', '退换货分流',
      '中文售后场景：尺码不合要求换货', true,
      '鞋子收到了，但是尺码不对，想换一双大一号的。付款和物流都没有问题，包装也完好。',
      [
        choice('department', '这个客户请求应该由哪个团队处理？',
          [opt('returns', '退换货与退款'),
            opt('billing', '扣费、发票与支付问题'),
            opt('logistics', '配送与包裹丢失')]),
        noul('needs_human', '这个请求是否需要人工客服介入？',
          '可以自助流程处理', '需要人工跟进')
      ]));

    list.push(new Scenario('moderation', '内容审核',
      '用户留言是否违规、是否转人工复核', true,
      '用户留言：这个商品真是太好用了，强烈推荐给大家！购买链接见我的主页，有问题可以私信我。',
      [
        noul('violation', '这条留言是否违反社区规范（广告导流、辱骂、敏感内容）？',
          '正常分享内容', '含有违规内容'),
        noul('manual_review', '是否需要转人工复核？', '', '')
      ]));

    list.push(new Scenario('agent-tool', 'Agent 工具选择',
      '端侧助手该调用哪个工具、要不要先向用户确认', true,
      '用户说：明天早上七点半叫我起床，顺便看看明天上海会不会下雨。',
      [
        choice('tool', '应该优先调用哪个工具？',
          [opt('alarm', '设置闹钟与提醒'),
            opt('weather', '查询天气预报'),
            opt('message', '发送消息给联系人'),
            opt('chat', '直接对话回复')]),
        noul('confirm', '执行前是否需要先向用户确认？',
          '意图明确，可直接执行', '存在歧义或副作用，需要确认')
      ]));

    list.push(new Scenario('satisfaction', '满意度评分',
      '售后回访文本 → 1–5 星满意度', true,
      '客户回访记录：物流很快，第二天就到了。客服态度也不错，耐心解答了我的问题。不过商品本身有点小瑕疵，总体来说还算满意，下次还会考虑回购。',
      [
        score('satisfaction', '客户的整体满意度如何？',
          ['1 星 · 非常不满意', '2 星 · 不满意', '3 星 · 一般', '4 星 · 满意', '5 星 · 非常满意'])
      ]));

    list.push(new Scenario('lead-priority', '线索优先级分级',
      '销售线索分级 + 是否立即跟进', true,
      'CRM 线索：某上市公司 IT 负责人主动来电咨询企业版报价，预算已获批，希望本周内安排产品演示，竞争对手也在接触。',
      [
        score('priority', '这条销售线索的优先级有多高？',
          ['低 · 慢慢培育', '中 · 常规跟进', '高 · 尽快处理', '紧急 · 立即行动']),
        noul('follow_now', '是否需要今天立即跟进？', '', '')
      ]));

    list.push(new Scenario('email-triage', 'Email triage (EN)',
      'English inbox triage: category + reply urgency', true,
      'Email from a long-term enterprise customer: "Our production integration started returning 500 errors after last night\'s API update. This is blocking all our checkout traffic. Please escalate."',
      [
        choice('category', 'Which category fits this email?',
          [opt('outage', 'Production outage / incident'),
            opt('billing', 'Billing question'),
            opt('feature', 'Feature request'),
            opt('spam', 'Spam or marketing')]),
        score('reply_urgency', 'How urgently should we reply?',
          ['Within a week', 'Within a day', 'Within an hour'])
      ]));

    ScenarioStore.builtins = list;
    return list;
  }

  // -------------------------------------------------------------------------
  // custom scenarios (preferences persistence)
  // -------------------------------------------------------------------------

  private static async openStore(context: common.UIAbilityContext): Promise<preferences.Preferences> {
    return preferences.getPreferences(context, 'pocket_scenarios');
  }

  static async loadCustom(context: common.UIAbilityContext): Promise<Scenario[]> {
    try {
      const store = await ScenarioStore.openStore(context);
      const text = await store.get('custom', '[]') as string;
      const raw = JSON.parse(text) as StoredScenario[];
      const result: Scenario[] = [];
      for (const item of raw) {
        const questions: QuestionDraft[] = [];
        for (const q of item.questions) {
          const draft = new QuestionDraft(q.id, q.type, q.instructions);
          for (const option of q.choices) {
            draft.choices.push(new ChoiceOption(option.key, option.description));
          }
          draft.noulFalse = q.noulFalse;
          draft.noulTrue = q.noulTrue;
          draft.scale = q.scale;
          questions.push(draft);
        }
        result.push(new Scenario(item.id, item.title, '自定义场景', false, item.state, questions));
      }
      return result;
    } catch (error) {
      console.error(`loadCustom failed: ${JSON.stringify(error)}`);
      return [];
    }
  }

  static async saveCustom(context: common.UIAbilityContext, scenarios: Scenario[]): Promise<void> {
    const stored: StoredScenario[] = [];
    for (const scenario of scenarios) {
      const questions: StoredQuestion[] = [];
      for (const q of scenario.questions) {
        const choices: ChoiceOption[] = [];
        for (const option of q.choices) {
          choices.push(new ChoiceOption(option.key, option.description));
        }
        questions.push({
          id: q.id, type: q.type, instructions: q.instructions,
          choices: choices, noulFalse: q.noulFalse, noulTrue: q.noulTrue, scale: q.scale
        } as StoredQuestion);
      }
      stored.push({
        id: scenario.id, title: scenario.title, state: scenario.state, questions: questions
      } as StoredScenario);
    }
    const store = await ScenarioStore.openStore(context);
    await store.put('custom', JSON.stringify(stored));
    await store.flush();
  }
}

interface StoredQuestion {
  id: string;
  type: string;
  instructions: string;
  choices: ChoiceOption[];
  noulFalse: string;
  noulTrue: string;
  scale: string[];
}

interface StoredScenario {
  id: string;
  title: string;
  state: string;
  questions: StoredQuestion[];
}
/**
 * Decision-mode data model: scenario drafts, wire-request builder and the
 * result parser. The wire protocol mirrors inference_mnn.py exactly; the
 * native side re-validates everything, this layer only builds well-formed
 * requests and turns answers into display cards.
 */

/** One candidate of a choice question. */
export class ChoiceOption {
  key: string;
  description: string;

  constructor(key: string, description: string) {
    this.key = key;
    this.description = description;
  }
}

/** Editable question draft (form mode of the custom scenario editor). */
export class QuestionDraft {
  id: string;
  type: string; // 'choice' | 'noul' | 'score'
  instructions: string;
  choices: ChoiceOption[];
  noulFalse: string;
  noulTrue: string;
  scale: string[];

  constructor(id: string, type: string, instructions: string) {
    this.id = id;
    this.type = type;
    this.instructions = instructions;
    this.choices = [];
    this.noulFalse = '';
    this.noulTrue = '';
    this.scale = [];
  }
}

/** A runnable scenario: state text + ordered questions. */
export class Scenario {
  id: string;
  title: string;
  subtitle: string;
  builtin: boolean;
  state: string;
  questions: QuestionDraft[];

  constructor(id: string, title: string, subtitle: string, builtin: boolean,
    state: string, questions: QuestionDraft[]) {
    this.id = id;
    this.title = title;
    this.subtitle = subtitle;
    this.builtin = builtin;
    this.state = state;
    this.questions = questions;
  }

  typeTags(): string[] {
    const tags: string[] = [];
    for (const q of this.questions) {
      if (!tags.includes(q.type)) {
        tags.push(q.type);
      }
    }
    return tags;
  }
}

// ---------------------------------------------------------------------------
// request builder
// ---------------------------------------------------------------------------

function jsonStr(value: string): string {
  let out = '"';
  for (let i = 0; i < value.length; i++) {
    const c = value.charAt(i);
    const code = value.charCodeAt(i);
    if (c === '"') {
      out += '\\"';
    } else if (c === '\\') {
      out += '\\\\';
    } else if (c === '\n') {
      out += '\\n';
    } else if (c === '\r') {
      out += '\\r';
    } else if (c === '\t') {
      out += '\\t';
    } else if (code < 0x20) {
      out += `\\u${code.toString(16).padStart(4, '0')}`;
    } else {
      out += c;
    }
  }
  return out + '"';
}

/** Serializes a scenario into the checkpoint's request JSON. */
export function buildRequestJson(state: string, questions: QuestionDraft[],
  temperature: number): string {
  let out = `{"state":${jsonStr(state)},"questions":{`;
  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    if (i > 0) {
      out += ',';
    }
    out += `${jsonStr(q.id)}:{"type":${jsonStr(q.type)},"instructions":${jsonStr(q.instructions)}`;
    if (q.type === 'choice') {
      out += ',"criteria":{';
      for (let j = 0; j < q.choices.length; j++) {
        if (j > 0) {
          out += ',';
        }
        out += `${jsonStr(q.choices[j].key)}:${jsonStr(q.choices[j].description)}`;
      }
      out += '}';
    } else if (q.type === 'noul') {
      const parts: string[] = [];
      if (q.noulFalse.length > 0) {
        parts.push(`"false":${jsonStr(q.noulFalse)}`);
      }
      if (q.noulTrue.length > 0) {
        parts.push(`"true":${jsonStr(q.noulTrue)}`);
      }
      if (parts.length > 0) {
        out += `,"criteria":{${parts.join(',')}}`;
      }
    } else if (q.type === 'score') {
      out += ',"criteria":[';
      for (let j = 0; j < q.scale.length; j++) {
        if (j > 0) {
          out += ',';
        }
        out += jsonStr(q.scale[j]);
      }
      out += ']';
    }
    out += '}';
  }
  out += '}';
  if (temperature > 0) {
    out += `,"temperature":${temperature}`;
  }
  out += '}';
  return out;
}

/** Client-side validation; returns null when the scenario is runnable. */
export function validateScenario(state: string, questions: QuestionDraft[]): string | null {
  if (state.trim().length === 0) {
    return '场景状态（state）不能为空';
  }
  if (questions.length === 0) {
    return '至少需要一个决策问题';
  }
  if (questions.length > 10) {
    return '问题数量请不要超过 10 个';
  }
  const ids: string[] = [];
  for (const q of questions) {
    if (q.id.trim().length === 0) {
      return '问题 ID 不能为空';
    }
    if (ids.includes(q.id)) {
      return `问题 ID 重复：${q.id}`;
    }
    ids.push(q.id);
    if (q.instructions.trim().length === 0) {
      return `问题 ${q.id} 缺少判定说明（instructions）`;
    }
    if (q.type === 'choice') {
      if (q.choices.length < 1 || q.choices.length > 255) {
        return `问题 ${q.id}：choice 候选数需在 1–255 之间`;
      }
      const keys: string[] = [];
      for (const option of q.choices) {
        if (option.key.trim().length === 0) {
          return `问题 ${q.id}：候选 key 不能为空`;
        }
        if (keys.includes(option.key)) {
          return `问题 ${q.id}：候选 key 重复 ${option.key}`;
        }
        keys.push(option.key);
      }
    } else if (q.type === 'score') {
      if (q.scale.length < 2 || q.scale.length > 10) {
        return `问题 ${q.id}：score 量表需 2–10 级`;
      }
      for (const label of q.scale) {
        if (label.trim().length === 0) {
          return `问题 ${q.id}：量表标签不能为空`;
        }
      }
    } else if (q.type !== 'noul') {
      return `问题 ${q.id}：未知题型 ${q.type}`;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// result parsing
// ---------------------------------------------------------------------------

interface AnswerJson {
  type: string;
  probabilities: Record<string, number>;
  choice?: string;
  decision?: boolean;
  noul?: number;
  score?: number;
  legend?: Record<string, string>;
}

interface DecisionResultJson {
  answers: Record<string, AnswerJson>;
  input_tokens: number;
  temperature: number;
  encodeMs?: number;
  forwardMs?: number;
  loadMs?: number;
  questionCount?: number;
}

/** One candidate row in a result card. */
export class ProbEntry {
  key: string;
  label: string;
  probability: number;
  winner: boolean;

  constructor(key: string, label: string, probability: number, winner: boolean) {
    this.key = key;
    this.label = label;
    this.probability = probability;
    this.winner = winner;
  }
}

/** Display model for one answered question. */
export class AnswerCard {
  id: string;
  type: string;
  entries: ProbEntry[];
  choice: string;
  decision: boolean;
  noul: number;
  score: number;
  scaleLabels: string[];

  constructor(id: string, type: string) {
    this.id = id;
    this.type = type;
    this.entries = [];
    this.choice = '';
    this.decision = false;
    this.noul = 0;
    this.score = 0;
    this.scaleLabels = [];
  }
}

/** Full decision outcome with the engine timing report. */
export class DecisionOutcome {
  cards: AnswerCard[];
  inputTokens: number;
  temperature: number;
  encodeMs: number;
  forwardMs: number;
  loadMs: number;

  constructor() {
    this.cards = [];
    this.inputTokens = 0;
    this.temperature = 0;
    this.encodeMs = 0;
    this.forwardMs = 0;
    this.loadMs = 0;
  }
}

function findDraft(questions: QuestionDraft[], id: string): QuestionDraft | null {
  for (const q of questions) {
    if (q.id === id) {
      return q;
    }
  }
  return null;
}

/** Parses the native dec_done payload into display cards. */
export function parseOutcome(json: string, questions: QuestionDraft[]): DecisionOutcome | null {
  try {
    const raw = JSON.parse(json) as DecisionResultJson;
    const outcome = new DecisionOutcome();
    outcome.inputTokens = raw.input_tokens;
    outcome.temperature = raw.temperature;
    outcome.encodeMs = raw.encodeMs ?? 0;
    outcome.forwardMs = raw.forwardMs ?? 0;
    outcome.loadMs = raw.loadMs ?? 0;
    const ids = Object.keys(raw.answers);
    for (const id of ids) {
      const answer = raw.answers[id];
      const card = new AnswerCard(id, answer.type);
      const draft = findDraft(questions, id);
      card.choice = answer.choice ?? '';
      card.decision = answer.decision ?? false;
      card.noul = answer.noul ?? 0;
      card.score = answer.score ?? 0;
      const keys = Object.keys(answer.probabilities);
      let bestKey = '';
      let bestProb = -1;
      for (const key of keys) {
        const p = answer.probabilities[key];
        if (p > bestProb) {
          bestProb = p;
          bestKey = key;
        }
      }
      for (const key of keys) {
        let label = key;
        let winner = key === bestKey;
        if (answer.type === 'choice') {
          winner = key === card.choice;
          if (draft !== null) {
            for (const option of draft.choices) {
              if (option.key === key && option.description.length > 0) {
                label = `${key} · ${option.description}`;
                break;
              }
            }
          }
        } else if (answer.type === 'noul') {
          label = key === 'true' ? '是' : '否';
          winner = (key === 'true') === card.decision;
        } else if (answer.type === 'score') {
          if (answer.legend !== undefined && answer.legend[key] !== undefined) {
            label = answer.legend[key];
          } else if (draft !== null) {
            const index = Number.parseInt(key, 10);
            if (index >= 0 && index < draft.scale.length) {
              label = draft.scale[index];
            }
          }
          card.scaleLabels.push(label);
        }
        card.entries.push(new ProbEntry(key, label, answer.probabilities[key], winner));
      }
      outcome.cards.push(card);
    }
    return outcome;
  } catch (error) {
    console.error(`parseOutcome failed: ${JSON.stringify(error)}`);
    return null;
  }
}

/** 0.872 -> "87.2%" */
export function formatPercent(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

export function typeLabel(type: string): string {
  if (type === 'choice') {
    return '多选一';
  }
  if (type === 'noul') {
    return '是否判定';
  }
  if (type === 'score') {
    return '量表评分';
  }
  return type;
}
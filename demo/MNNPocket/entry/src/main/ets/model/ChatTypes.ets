/**
 * Shared data model for the chat screen.
 */

export enum Role {
  USER = 'user',
  ASSISTANT = 'assistant'
}

/** Token accounting and latency for one answer, filled from MNN's LlmContext. */
export class ChatStats {
  promptTokens: number = 0;
  outputTokens: number = 0;
  contextTokens: number = 0;
  prefillMs: number = 0;
  decodeMs: number = 0;
  ttfaMs: number = 0;
  totalMs: number = 0;
  loadMs: number = 0;
  prefillTps: number = 0;
  decodeTps: number = 0;
}

interface StatsJson {
  promptTokens: number;
  outputTokens: number;
  contextTokens: number;
  prefillMs: number;
  decodeMs: number;
  ttfaMs: number;
  totalMs: number;
  loadMs: number;
  prefillTps: number;
  decodeTps: number;
}

interface DoneJson {
  text: string;
  promptTokens: number;
  outputTokens: number;
  contextTokens: number;
  prefillMs: number;
  decodeMs: number;
  ttfaMs: number;
  totalMs: number;
  loadMs: number;
  prefillTps: number;
  decodeTps: number;
}

interface DeltaJson {
  text: string;
}

interface ErrorJson {
  stage: string;
  message: string;
}

interface InitJson {
  model: string;
  loaded: boolean;
  loadMs: number;
  maxNewTokens: number;
  systemPrompt: string;
}

export function parseStats(json: string): ChatStats {
  const stats = new ChatStats();
  try {
    const raw = JSON.parse(json) as StatsJson;
    stats.promptTokens = raw.promptTokens;
    stats.outputTokens = raw.outputTokens;
    stats.contextTokens = raw.contextTokens;
    stats.prefillMs = raw.prefillMs;
    stats.decodeMs = raw.decodeMs;
    stats.ttfaMs = raw.ttfaMs;
    stats.totalMs = raw.totalMs;
    stats.loadMs = raw.loadMs;
    stats.prefillTps = raw.prefillTps;
    stats.decodeTps = raw.decodeTps;
  } catch (error) {
    console.error(`parseStats failed: ${JSON.stringify(error)}`);
  }
  return stats;
}

export function parseDelta(json: string): string {
  try {
    return (JSON.parse(json) as DeltaJson).text;
  } catch (error) {
    return '';
  }
}

export function parseError(json: string): string {
  try {
    return (JSON.parse(json) as ErrorJson).message;
  } catch (error) {
    return json;
  }
}

export function parseInit(json: string): InitJson | null {
  try {
    return JSON.parse(json) as InitJson;
  } catch (error) {
    return null;
  }
}

export function parseDoneText(json: string): DoneJson | null {
  try {
    return JSON.parse(json) as DoneJson;
  } catch (error) {
    return null;
  }
}

export class ChatMessage {
  id: number;
  role: Role;
  text: string;
  stats: ChatStats;
  timeLabel: string;

  constructor(id: number, role: Role, text: string, stats: ChatStats, timeLabel: string) {
    this.id = id;
    this.role = role;
    this.text = text;
    this.stats = stats;
    this.timeLabel = timeLabel;
  }
}

/** 1866 -> "1866 ms", 12345 -> "12.3 s" */
export function formatMillis(ms: number): string {
  if (ms < 10000) {
    return `${ms} ms`;
  }
  return `${(ms / 1000).toFixed(1)} s`;
}

export function formatTokens(count: number): string {
  if (count < 1000) {
    return `${count}`;
  }
  return `${(count / 1000).toFixed(1)}k`;
}

export function nowLabel(): string {
  const date = new Date();
  const hour = date.getHours().toString().padStart(2, '0');
  const minute = date.getMinutes().toString().padStart(2, '0');
  return `${hour}:${minute}`;
}

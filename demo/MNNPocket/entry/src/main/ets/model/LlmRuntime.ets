/**
 * Typed singleton around libllmbridge.so.
 *
 * The native side owns a worker thread and streams results back through a
 * single event callback; this class just forwards those events to the page.
 */

import {
  setEventCallback,
  initModel,
  chat,
  resetSession,
  isGenerating,
  isReady,
  initDecision,
  decide,
  unloadDecision,
  isDecisionReady
} from 'libllmbridge.so';

export type BridgeEvent = (name: string, payload: string) => void;

export class LlmRuntime {
  private static readonly SINGLETON: LlmRuntime = new LlmRuntime();
  private handler: BridgeEvent | null = null;
  private bound: boolean = false;

  static get(): LlmRuntime {
    return LlmRuntime.SINGLETON;
  }

  bind(handler: BridgeEvent): void {
    this.handler = handler;
    if (this.bound) {
      return;
    }
    setEventCallback((name: string, payload: string) => {
      this.dispatch(name, payload);
    });
    this.bound = true;
  }

  private dispatch(name: string, payload: string): void {
    if (this.handler !== null) {
      this.handler(name, payload);
    }
  }

  /** Kicks off model creation; completion arrives as a "ready" event. */
  load(configPath: string, tmpDir: string, maxNewTokens: number): void {
    initModel(configPath, tmpDir, maxNewTokens);
  }

  /** Streams the answer through "delta" events, then "done" with the metrics. */
  ask(prompt: string): void {
    chat(prompt);
  }

  clear(): void {
    resetSession();
  }

  ready(): boolean {
    return isReady();
  }

  busy(): boolean {
    return isGenerating();
  }

  // -------------------------------------------------------------------------
  // decision mode (same native worker; events dec_*)
  // -------------------------------------------------------------------------

  /** Kicks off decision-graph loading; completion arrives as "dec_ready". */
  loadDecision(modelDir: string, threads: number, maxTokens: number): void {
    initDecision(modelDir, 'decision-w8-q8-embedding.mnn', threads, maxTokens);
  }

  /** Single-forward decision; result arrives as "dec_done". */
  runDecision(requestJson: string): void {
    decide(requestJson);
  }

  releaseDecision(): void {
    unloadDecision();
  }

  decisionReady(): boolean {
    return isDecisionReady();
  }
}

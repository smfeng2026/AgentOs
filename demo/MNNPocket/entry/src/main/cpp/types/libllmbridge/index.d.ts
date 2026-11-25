/**
 * ArkTS surface of libllmbridge.so.
 *
 * Every call is asynchronous from JS's point of view: work is queued onto the
 * native worker thread and results come back through the single event callback
 * registered with setEventCallback(). Chat and decision tasks share that one
 * worker, so they never run concurrently.
 *
 * Chat events:
 *   init_progress {stage}                         model is being created
 *   ready         {model, loaded, loadMs, ...}    model is in memory
 *   start         {prompt}                        a turn began
 *   delta         {text}                          streamed text chunk
 *   done          {text, promptTokens, outputTokens, contextTokens,
 *                  prefillMs, decodeMs, ttfaMs, totalMs, loadMs,
 *                  prefillTps, decodeTps}         turn finished
 *   reset_done    {}                              history and KV cache cleared
 *   error         {stage, message}                failure
 *
 * Decision events:
 *   dec_progress  {stage}                         decision engine is loading
 *   dec_ready     {model, variant, loaded, loadMs, maxTokens, threads,
 *                  temperature}                   decision graph in memory
 *   dec_start     {}                              a decision forward began
 *   dec_done      {answers, input_tokens, temperature, encodeMs, forwardMs,
 *                  loadMs, questionCount}         result JSON (see DecisionTypes)
 *   dec_unloaded  {}                              decision graph released
 *   dec_error     {stage, message}                failure
 */

export const setEventCallback: (callback: (name: string, payload: string) => void) => void;

export const initModel: (configPath: string, tmpDir: string, maxNewTokens: number) => void;

export const chat: (prompt: string) => void;

export const resetSession: () => void;

export const isReady: () => boolean;

export const isGenerating: () => boolean;

export const modelInfo: () => string;

export const shutdown: () => void;

/** Loads the decision graph; completion arrives as a "dec_ready" event. */
export const initDecision: (modelDir: string, modelFile: string, threads: number, maxTokens: number) => void;

/** Single-forward decision; result arrives as "dec_done" (or "dec_error"). */
export const decide: (requestJson: string) => void;

/** Releases the decision graph to reclaim memory ("dec_unloaded" event). */
export const unloadDecision: () => void;

export const isDecisionReady: () => boolean;

export const decisionInfo: () => string;

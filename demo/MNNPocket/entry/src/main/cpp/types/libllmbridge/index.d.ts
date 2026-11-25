/**
 * ArkTS surface of libllmbridge.so.
 *
 * Every call is asynchronous from JS's point of view: work is queued onto the
 * native worker thread and results come back through the single event callback
 * registered with setEventCallback().
 *
 * Event names:
 *   init_progress {stage}                         model is being created
 *   ready         {model, loaded, loadMs, ...}    model is in memory
 *   start         {prompt}                        a turn began
 *   delta         {text}                          streamed text chunk
 *   done          {text, promptTokens, outputTokens, contextTokens,
 *                  prefillMs, decodeMs, ttfaMs, totalMs, loadMs,
 *                  prefillTps, decodeTps}         turn finished
 *   reset_done    {}                              history and KV cache cleared
 *   error         {stage, message}                failure
 */

export const setEventCallback: (callback: (name: string, payload: string) => void) => void;

export const initModel: (configPath: string, tmpDir: string, maxNewTokens: number) => void;

export const chat: (prompt: string) => void;

export const resetSession: () => void;

export const isReady: () => boolean;

export const isGenerating: () => boolean;

export const modelInfo: () => string;

export const shutdown: () => void;

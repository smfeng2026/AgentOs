import { common } from '@kit.AbilityKit';
import { preferences } from '@kit.ArkData';
import { SmsThread, seedThreads } from './SmsData';

export interface SmsRouteParams {
  threadId?: number;
  messageId?: number;
  mode?: string;
}

// All route pages share the existing storage key so navigation retains edits.
export class SmsStore {
  static load(context: common.Context | undefined): SmsThread[] {
    if (!context) throw new Error('Page context is unavailable');
    const store = preferences.getPreferencesSync(context, { name: 'sms_demo' });
    const raw = store.getSync('threads_v1', '') as string;
    if (raw) return JSON.parse(raw) as SmsThread[];
    const threads = seedThreads();
    SmsStore.save(context, threads);
    return threads;
  }

  static save(context: common.Context | undefined, threads: SmsThread[]): void {
    if (!context) throw new Error('Page context is unavailable');
    const store = preferences.getPreferencesSync(context, { name: 'sms_demo' });
    store.putSync('threads_v1', JSON.stringify(threads));
    store.flushSync();
  }
}

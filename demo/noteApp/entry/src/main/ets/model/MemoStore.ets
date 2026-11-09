import { preferences } from '@kit.ArkData';

export interface Memo {
  id: string;
  title: string;
  content: string;
  updatedAt: number;
}

export async function loadMemos(context: Context): Promise<Memo[]> {
  const store = await preferences.getPreferences(context, 'memo_store');
  const value = await store.get('memos', '[]');
  const loaded = JSON.parse(value as string) as Memo[];
  return Array.isArray(loaded) ? loaded.filter((memo: Memo) =>
    typeof memo.id === 'string' && typeof memo.title === 'string' &&
    typeof memo.content === 'string' && typeof memo.updatedAt === 'number') : [];
}

export async function saveMemos(context: Context, memos: Memo[]): Promise<void> {
  const store = await preferences.getPreferences(context, 'memo_store');
  await store.put('memos', JSON.stringify(memos));
  await store.flush();
}

/**
 * Visual tokens for the chat screen. Kept in one place so the WeChat-ish look
 * stays consistent across bubbles, input bar and status chrome.
 */
export class Palette {
  static readonly PAGE_BG: string = '#F2F3F5';
  static readonly HEADER_BG: string = '#FFFFFF';
  static readonly CARD_BG: string = '#FFFFFF';
  static readonly BUBBLE_AI: string = '#FFFFFF';
  static readonly BUBBLE_USER: string = '#95EC69';
  static readonly TEXT_PRIMARY: string = '#14171A';
  static readonly TEXT_SECONDARY: string = '#8A9199';
  static readonly TEXT_MUTED: string = '#9AA4B2';
  static readonly ACCENT: string = '#07C160';
  static readonly ACCENT_DISABLED: string = '#C7E9D3';
  static readonly CHIP_BG: string = '#F7F8FA';
  static readonly INPUT_BG: string = '#F2F3F5';
  static readonly DIVIDER: string = '#EBEDF0';
  static readonly DANGER: string = '#E5484D';
  static readonly WARNING: string = '#F5A524';
  static readonly AVATAR_AI_START: string = '#8B5CF6';
  static readonly AVATAR_AI_END: string = '#4F46E5';
  static readonly AVATAR_USER: string = '#4C9AFF';

  // Decision mode: indigo accent family, same card language as chat.
  static readonly DEC_ACCENT: string = '#4F46E5';
  static readonly DEC_ACCENT_SOFT: string = '#EEF0FF';
  static readonly DEC_ACCENT_DISABLED: string = '#C9CDF5';
  static readonly DEC_BAR_TRACK: string = '#ECEEF3';
  static readonly DEC_WINNER_BG: string = '#F0F1FF';
  static readonly DEC_GRADIENT_START: string = '#6366F1';
  static readonly DEC_GRADIENT_END: string = '#4338CA';
  static readonly TAG_CHOICE: string = '#4F46E5';
  static readonly TAG_NOUL: string = '#0E9F6E';
  static readonly TAG_SCORE: string = '#D97706';
}

/** Tag color for a question type badge. */
export function typeTagColor(type: string): string {
  if (type === 'choice') {
    return Palette.TAG_CHOICE;
  }
  if (type === 'noul') {
    return Palette.TAG_NOUL;
  }
  return Palette.TAG_SCORE;
}

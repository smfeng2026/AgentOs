import { router } from '@kit.ArkUI';
import { promptAction } from '@kit.ArkUI';
import { Memo, loadMemos, saveMemos } from '../model/MemoStore';

interface EditorParams {
  id: string;
  title: string;
  content: string;
}

@Entry
@Component
struct MemoEditor {
  @State private memoId: string = '';
  @State private title: string = '';
  @State private content: string = '';

  aboutToAppear(): void {
    const params = router.getParams() as EditorParams;
    this.memoId = params.id;
    this.title = params.title;
    this.content = params.content;
  }

  onPageShow(): void {
    console.info('[Memo] MemoEditor onPageShow');
  }

  onBackPress(): boolean {
    this.finishEditing();
    return true;
  }

  private async finishEditing(): Promise<void> {
    const cleanTitle = this.title.trim();
    const cleanContent = this.content.trim();
    try {
      const memos = await loadMemos(getContext(this));
      const existingIndex = memos.findIndex((memo: Memo) => memo.id === this.memoId);
      if (cleanTitle.length > 0 || cleanContent.length > 0 || this.memoId.length > 0) {
        const memo: Memo = {
          id: this.memoId.length > 0 ? this.memoId : `${Date.now()}-${Math.random()}`,
          title: cleanTitle.length > 0 ? cleanTitle :
            (cleanContent.length > 0 ? cleanContent.split('\n')[0].slice(0, 30) : '无标题'),
          content: this.content,
          updatedAt: Date.now()
        };
        if (existingIndex >= 0) {
          memos.splice(existingIndex, 1, memo);
        } else {
          memos.unshift(memo);
        }
        await saveMemos(getContext(this), memos);
      }
      router.back();
    } catch (error) {
      promptAction.showToast({ message: '保存失败，请重试' });
    }
  }

  private async deleteMemo(): Promise<void> {
    try {
      const memos = await loadMemos(getContext(this));
      await saveMemos(getContext(this), memos.filter((memo: Memo) => memo.id !== this.memoId));
      router.back();
    } catch (error) {
      promptAction.showToast({ message: '删除失败，请重试' });
    }
  }

  private confirmDelete(): void {
    AlertDialog.show({
      title: '删除备忘录？',
      message: '删除后无法恢复。',
      primaryButton: { value: '取消', action: () => {} },
      secondaryButton: {
        value: '删除',
        fontColor: '#D83B38',
        action: () => this.deleteMemo()
      }
    });
  }

  build() {
    Column() {
      Row() {
        Button('‹', { type: ButtonType.Circle })
          .fontSize(36)
          .fontColor('#171719')
          .backgroundColor('#F7F7F7')
          .width(48)
          .height(48)
          .accessibilityText('返回并保存')
          .onClick(() => this.finishEditing())
        Blank()
        if (this.memoId.length > 0) {
          Button('···', { type: ButtonType.Circle })
            .fontSize(28)
            .fontColor('#171719')
            .backgroundColor('#F7F7F7')
            .width(48)
            .height(48)
            .accessibilityText('删除备忘录')
            .onClick(() => this.confirmDelete())
            .margin({ right: 10 })
        }
        Button('完成')
          .fontSize(16)
          .fontColor('#171719')
          .backgroundColor('#F7F7F7')
          .height(48)
          .onClick(() => this.finishEditing())
      }
      .width('100%')
      .padding({ top: 12, bottom: 30 })

      TextInput({ placeholder: '标题', text: this.title })
        .fontSize(34)
        .fontWeight(FontWeight.Bold)
        .fontColor('#171719')
        .placeholderColor('#A0A0A0')
        .backgroundColor(Color.Transparent)
        .width('100%')
        .margin({ bottom: 20 })
        .onChange((value: string) => this.title = value)

      TextArea({ placeholder: '开始记录...', text: this.content })
        .fontSize(18)
        .fontColor('#444444')
        .placeholderColor('#A0A0A0')
        .backgroundColor(Color.Transparent)
        .width('100%')
        .layoutWeight(1)
        .onChange((value: string) => this.content = value)
    }
    .width('100%')
    .height('100%')
    .padding({ left: 22, right: 22, bottom: 16, top: 12 })
    .backgroundColor('#FFFFFF')
  }
}

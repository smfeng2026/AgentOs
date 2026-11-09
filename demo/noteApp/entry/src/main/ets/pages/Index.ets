import { router } from '@kit.ArkUI';
import { promptAction } from '@kit.ArkUI';
import { Memo, loadMemos } from '../model/MemoStore';

@Entry
@Component
struct Index {
  @State private memos: Memo[] = [];
  @State private search: string = '';

  onPageShow(): void {
    console.info('[Memo] Index onPageShow');
    this.refreshMemos();
  }

  private async refreshMemos(): Promise<void> {
    try {
      this.memos = await loadMemos(getContext(this));
    } catch (error) {
      promptAction.showToast({ message: '备忘录读取失败' });
    }
  }

  private visibleMemos(): Memo[] {
    const keyword = this.search.trim().toLowerCase();
    return this.memos
      .filter((memo: Memo) => keyword.length === 0 ||
        memo.title.toLowerCase().includes(keyword) || memo.content.toLowerCase().includes(keyword))
      .sort((left: Memo, right: Memo) => right.updatedAt - left.updatedAt);
  }

  private monthKey(timestamp: number): string {
    const date = new Date(timestamp);
    return `${date.getFullYear()}-${date.getMonth()}`;
  }

  private monthLabel(timestamp: number): string {
    const date = new Date(timestamp);
    const now = new Date();
    return date.getFullYear() === now.getFullYear() ?
      `${date.getMonth() + 1}月` : `${date.getFullYear()}年${date.getMonth() + 1}月`;
  }

  private dateLabel(timestamp: number): string {
    const date = new Date(timestamp);
    return `${date.getFullYear()}/${date.getMonth() + 1}/${date.getDate()}`;
  }

  private async openEditor(memo?: Memo): Promise<void> {
    try {
      await router.pushUrl({
        url: 'pages/MemoEditor',
        params: {
          id: memo ? memo.id : '',
          title: memo ? memo.title : '',
          content: memo ? memo.content : ''
        }
      });
    } catch (error) {
      promptAction.showToast({ message: '无法打开编辑页面' });
    }
  }

  @Builder
  private listPage() {
    Column() {
      Text('读书笔记')
        .fontSize(38)
        .fontWeight(FontWeight.Bold)
        .fontColor('#000000')
        .width('100%')
      Text(`${this.memos.length} 个备忘录`)
        .fontSize(17)
        .fontColor('#8C8C91')
        .width('100%')
        .margin({ top: 2, bottom: 22 })

      Scroll() {
        Column() {
          if (this.visibleMemos().length === 0) {
            Column({ space: 10 }) {
              Text(this.search.length > 0 ? '没有找到备忘录' : '还没有备忘录')
                .fontSize(20)
                .fontWeight(FontWeight.Medium)
              Text(this.search.length > 0 ? '试试其他关键词' : '点击右下角按钮开始记录')
                .fontSize(15)
                .fontColor('#8C8C91')
            }
            .width('100%')
            .padding({ top: 80 })
          } else {
            ForEach(this.visibleMemos(), (memo: Memo, index: number) => {
              if (index === 0 || this.monthKey(memo.updatedAt) !==
                this.monthKey(this.visibleMemos()[index - 1].updatedAt)) {
                Text(this.monthLabel(memo.updatedAt))
                  .fontSize(25)
                  .fontWeight(FontWeight.Bold)
                  .width('100%')
                  .margin({ top: index === 0 ? 0 : 24, bottom: 14 })
              }
              Column({ space: 3 }) {
                Text(memo.title)
                  .fontSize(19)
                  .fontWeight(FontWeight.Bold)
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
                  .width('100%')
                Text(`${this.dateLabel(memo.updatedAt)}  ${memo.content.replace(/\s+/g, ' ')}`)
                  .fontSize(16)
                  .fontColor('#8C8C91')
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
                  .width('100%')
              }
              .alignItems(HorizontalAlign.Start)
              .width('100%')
              .padding({ left: 24, right: 20, top: 14, bottom: 14 })
              .backgroundColor('#FFFFFF')
              .borderRadius(24)
              .margin({ bottom: 12 })
              .onClick(() => this.openEditor(memo))
            }, (memo: Memo) => memo.id)
          }
        }
        .width('100%')
      }
      .layoutWeight(1)
      .scrollBar(BarState.Off)

      Row({ space: 12 }) {
        Row({ space: 10 }) {
          Text('⌕')
            .fontSize(30)
          TextInput({ placeholder: '搜索', text: this.search })
            .fontSize(17)
            .backgroundColor(Color.Transparent)
            .layoutWeight(1)
            .onChange((value: string) => this.search = value)
        }
        .height(54)
        .padding({ left: 18, right: 12 })
        .backgroundColor('#FFFFFF')
        .borderRadius(28)
        .layoutWeight(1)

        Button('✎', { type: ButtonType.Circle })
          .fontSize(28)
          .fontColor('#171719')
          .backgroundColor('#FFFFFF')
          .width(54)
          .height(54)
          .accessibilityText('新建备忘录')
          .onClick(() => this.openEditor())
      }
      .width('100%')
      .padding({ top: 10, bottom: 12 })
    }
    .width('100%')
    .height('100%')
    .padding({ left: 20, right: 20, top: 50 })
    .backgroundColor('#F3F3F7')
  }

  build() {
    Column() {
      this.listPage()
    }
    .width('100%')
    .height('100%')
  }
}

import { hilog } from '@kit.PerformanceAnalysisKit';
import { SmsMessage, SmsThread } from './SmsData';
import { SmsRouteParams, SmsStore } from './SmsStore';

const INK = '#101114';
const MUTED = '#8C8E93';
const GREEN = '#32C766';

@Entry
@Component
struct Index {
  @State private threads: SmsThread[] = [];
  @State private query: string = '';
  @State private selectionMode: boolean = false;
  @State private selectedIds: number[] = [];

  onPageShow(): void {
    this.reload();
    hilog.info(0x0000, 'SmsDemo', 'onPageShow pages/Index');
  }

  onPageHide(): void {
    hilog.info(0x0000, 'SmsDemo', 'onPageHide pages/Index');
  }

  private reload(): void {
    try {
      this.threads = SmsStore.load(this.getUIContext().getHostContext());
    } catch (error) {
      this.getUIContext().getPromptAction().showToast({ message: '读取短信失败，请重试' });
    }
  }

  private persist(): boolean {
    try {
      SmsStore.save(this.getUIContext().getHostContext(), this.threads);
      return true;
    } catch (error) {
      this.getUIContext().getPromptAction().showToast({ message: '保存失败，请重试' });
      return false;
    }
  }

  private latest(thread: SmsThread): SmsMessage {
    return thread.messages[thread.messages.length - 1];
  }
  private visibleThreads(): SmsThread[] {
    const term = this.query.trim().toLowerCase();
    return this.threads.filter((thread: SmsThread) =>
      !term || thread.name.toLowerCase().includes(term) ||
      thread.messages.some((message: SmsMessage) => message.body.toLowerCase().includes(term)))
      .sort((a: SmsThread, b: SmsThread) => this.latest(b).time - this.latest(a).time);
  }
  private displayDate(time: number): string {
    const date = new Date(time);
    const today = new Date();
    if (date.toDateString() === today.toDateString()) {
      return date.getHours().toString().padStart(2, '0') + ':' +
        date.getMinutes().toString().padStart(2, '0');
    }
    return date.getFullYear() + '/' + (date.getMonth() + 1) + '/' + date.getDate();
  }

  private openThread(id: number): void {
    if (this.selectionMode) {
      this.toggleThread(id);
      return;
    }
    const params: SmsRouteParams = { threadId: id };
    this.getUIContext().getRouter().pushUrl({ url: 'pages/MessageDetail', params: params })
      .catch(() => this.getUIContext().getPromptAction().showToast({ message: '打开短信失败，请重试' }));
  }

  private beginNew(): void {
    const params: SmsRouteParams = { mode: 'new' };
    this.getUIContext().getRouter().pushUrl({ url: 'pages/MessageEditor', params: params })
      .catch(() => this.getUIContext().getPromptAction().showToast({ message: '打开编辑页失败，请重试' }));
  }

  private toggleThread(id: number): void {
    this.selectedIds = this.selectedIds.includes(id) ?
      this.selectedIds.filter((selected: number) => selected !== id) : this.selectedIds.concat([id]);
  }
  private deleteSelectedThreads(): void {
    if (this.selectedIds.length === 0) return;
    this.getUIContext().showAlertDialog({ title: '删除会话', message: '确定删除选中的 ' + this.selectedIds.length + ' 个会话吗？',
      primaryButton: { value: '取消', action: () => {} },
      secondaryButton: { value: '删除', fontColor: '#E5484D', action: () => {
        this.threads = this.threads.filter((thread: SmsThread) => !this.selectedIds.includes(thread.id));
        this.selectedIds = [];
        this.selectionMode = false;
        this.persist();
      } } });
  }
  @Builder
  private avatar(size: number) {
    Column() {
      Text('●').fontSize(size * 0.32).fontColor(Color.White).height(size * 0.38)
      Text('●').fontSize(size * 0.62).fontColor(Color.White).height(size * 0.3)
    }.width(size).height(size).borderRadius(size / 2).backgroundColor('#92A7D0')
      .justifyContent(FlexAlign.Center).clip(true)
  }
  build() {
    Column() {
      Row() {
        Button(this.selectionMode ? '取消' : '编辑').fontSize(16).fontColor(INK)
          .backgroundColor('#F3F3F5')
          .onClick(() => { this.selectionMode = !this.selectionMode; this.selectedIds = []; })
        Blank()
        if (this.selectionMode) {
          Button('全选').fontSize(16).fontColor(INK).backgroundColor('#F3F3F5')
            .onClick(() => this.selectedIds = this.visibleThreads().map((thread: SmsThread) => thread.id))
        }
      }.height(52).padding({ left: 20, right: 20 })
      Text('信息').fontSize(38).fontWeight(FontWeight.Bold).fontColor(INK)
        .width('100%').padding({ left: 20, top: 12, bottom: 14 })
      if (this.threads.length === 0 || this.visibleThreads().length === 0) {
        Column() {
          Text(this.threads.length === 0 ? '暂无信息' : '没有搜索结果')
            .fontSize(20).fontColor(MUTED)
          if (this.threads.length === 0) {
            Text('点击右下角新建短信').fontSize(14).fontColor(MUTED).margin({ top: 8 })
          }
        }.layoutWeight(1).width('100%').justifyContent(FlexAlign.Center)
      } else {
        List() {
          ForEach(this.visibleThreads(), (thread: SmsThread) => {
            ListItem() {
              Row() {
                if (this.selectionMode) {
                  Text(this.selectedIds.includes(thread.id) ? '✓' : '')
                    .fontSize(18).fontColor(Color.White).textAlign(TextAlign.Center)
                    .width(25).height(25).borderRadius(13)
                    .backgroundColor(this.selectedIds.includes(thread.id) ? GREEN : '#D4D5D9')
                    .margin({ right: 12 })
                }
                this.avatar(48)
                Column() {
                  Row() {
                    Text(thread.name).fontSize(19)
                      .fontWeight(thread.unread ? FontWeight.Bold : FontWeight.Medium)
                      .fontColor(INK).maxLines(1).textOverflow({ overflow: TextOverflow.Ellipsis })
                      .layoutWeight(1)
                    Text(this.displayDate(this.latest(thread).time)).fontSize(13).fontColor(MUTED)
                      .margin({ left: 8 })
                    Text('›').fontSize(25).fontColor('#C7C8CC').margin({ left: 6 })
                  }.width('100%')
                  Text(this.latest(thread).body.replace(/\n/g, ' ')).fontSize(15).fontColor(MUTED)
                    .maxLines(2).textOverflow({ overflow: TextOverflow.Ellipsis })
                    .margin({ top: 4 }).width('100%')
                  Divider().color('#E8E8EA').margin({ top: 18 })
                }.layoutWeight(1).margin({ left: 14 })
                if (thread.unread && !this.selectionMode) {
                  Circle().width(8).height(8).fill(GREEN).margin({ left: 8 })
                }
              }.width('100%').padding({ left: 20, right: 18, top: 16 })
                .alignItems(VerticalAlign.Center).onClick(() => this.openThread(thread.id))
            }
          }, (thread: SmsThread) => thread.id.toString())
        }.layoutWeight(1).width('100%').edgeEffect(EdgeEffect.Spring)
      }
      Row() {
        Row() {
          Text('⌕').fontSize(29).fontColor(INK).margin({ right: 8 })
          TextInput({ placeholder: '搜索', text: this.query }).fontSize(17)
            .backgroundColor(Color.Transparent).layoutWeight(1)
            .onChange((value: string) => this.query = value)
          if (this.query) {
            Button('×').fontSize(23).fontColor(MUTED).backgroundColor(Color.Transparent)
              .width(36).height(36).onClick(() => this.query = '')
          }
        }.layoutWeight(1).height(54).borderRadius(27).backgroundColor('#F3F3F5')
          .padding({ left: 18, right: 6 })
        Button(this.selectionMode ? '删除' : '✎')
          .fontSize(this.selectionMode ? 15 : 27).fontColor(this.selectionMode ? '#E5484D' : INK)
          .width(this.selectionMode ? 62 : 54).height(54).borderRadius(27)
          .backgroundColor('#F3F3F5').margin({ left: 12 })
          .enabled(!this.selectionMode || this.selectedIds.length > 0)
          .onClick(() => this.selectionMode ? this.deleteSelectedThreads() : this.beginNew())
      }.width('100%').padding({ left: 20, right: 20, top: 10, bottom: 16 })
    }.width('100%').height('100%').backgroundColor(Color.White)
  }

}

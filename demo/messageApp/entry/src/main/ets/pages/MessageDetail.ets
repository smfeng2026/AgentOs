import { hilog } from '@kit.PerformanceAnalysisKit';
import { SmsMessage, SmsThread } from './SmsData';
import { SmsRouteParams, SmsStore } from './SmsStore';

const INK = '#101114';
const MUTED = '#8C8E93';
const GREEN = '#32C766';

@Entry
@Component
struct MessageDetail {
  @State private threads: SmsThread[] = [];
  @State private activeId: number = -1;
  @State private selectedMessageId: number = -1;
  @State private reply: string = '';

  onPageShow(): void {
    const params = this.getUIContext().getRouter().getParams() as SmsRouteParams;
    this.activeId = params.threadId === undefined ? -1 : params.threadId;
    this.reload();
    this.selectedMessageId = -1;
    const thread = this.currentThread();
    if (thread && thread.unread) {
      this.threads = this.threads.map((item: SmsThread) => item.id === this.activeId ?
        { id: item.id, name: item.name, messages: item.messages, unread: false } : item);
      this.persist();
    }
    hilog.info(0x0000, 'SmsDemo', 'onPageShow pages/MessageDetail threadId=%{public}d', this.activeId);
  }

  onPageHide(): void {
    hilog.info(0x0000, 'SmsDemo', 'onPageHide pages/MessageDetail');
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

  private currentThread(): SmsThread | undefined {
    return this.threads.find((thread: SmsThread) => thread.id === this.activeId);
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

  private openEditor(mode: string): void {
    const params: SmsRouteParams = {
      threadId: this.activeId, messageId: this.selectedMessageId, mode: mode
    };
    this.getUIContext().getRouter().pushUrl({ url: 'pages/MessageEditor', params: params })
      .catch(() => this.getUIContext().getPromptAction().showToast({ message: '打开编辑页失败，请重试' }));
  }

  private sendReply(): void {
    const body = this.reply.trim();
    if (!body) return;
    const time = Date.now();
    this.threads = this.threads.map((thread: SmsThread) =>
      thread.id === this.activeId ? { id: thread.id, name: thread.name, unread: thread.unread,
        messages: thread.messages.concat([{ id: time, body: body, time: time, outgoing: true }]) } : thread);
    this.reply = '';
    this.persist();
  }
  private deleteMessage(): void {
    const thread = this.currentThread();
    if (!thread) return;
    this.getUIContext().showAlertDialog({ title: '删除短信', message: '确定删除这条短信吗？',
      primaryButton: { value: '取消', action: () => {} },
      secondaryButton: { value: '删除', fontColor: '#E5484D', action: () => {
        const remaining = thread.messages.filter((message: SmsMessage) => message.id !== this.selectedMessageId);
        this.threads = remaining.length ? this.threads.map((item: SmsThread) =>
          item.id === this.activeId ? { id: item.id, name: item.name, unread: item.unread,
            messages: remaining } : item) :
          this.threads.filter((item: SmsThread) => item.id !== this.activeId);
        this.selectedMessageId = -1;
        if (this.persist() && !remaining.length) this.getUIContext().getRouter().back();
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
        Button('‹').fontSize(35).fontColor(INK).width(45).height(45)
          .backgroundColor('#F5F5F6').onClick(() => this.getUIContext().getRouter().back())
        Blank()
        Button('编辑联系人').fontSize(14).fontColor(INK).backgroundColor('#F5F5F6')
          .onClick(() => this.openEditor('contact'))
      }.width('100%').height(55).padding({ left: 18, right: 18 })
      if (this.currentThread()) {
        this.avatar(64)
        Text(this.currentThread()!.name).fontSize(19).fontWeight(FontWeight.Medium)
          .fontColor(INK).margin({ top: 8, bottom: 18 })
        Divider().color('#EEEEF0')
        Scroll() {
          Column() {
            ForEach(this.currentThread()!.messages, (message: SmsMessage) => {
              Column() {
                Text(this.displayDate(message.time)).fontSize(12).fontColor('#A0A0A5')
                  .margin({ top: 18, bottom: 10 })
                Row() {
                  if (message.outgoing) Blank()
                  Text(message.body).fontSize(16).lineHeight(23)
                    .fontColor(message.outgoing ? Color.White : INK)
                    .padding(14).borderRadius(20)
                    .backgroundColor(message.outgoing ? GREEN : '#E9E9EB')
                    .constraintSize({ maxWidth: '83%' })
                    .onClick(() => this.selectedMessageId =
                      this.selectedMessageId === message.id ? -1 : message.id)
                  if (!message.outgoing) Blank()
                }.width('100%')
                if (this.selectedMessageId === message.id) {
                  Row() {
                    Button('编辑').fontSize(14).fontColor(INK).backgroundColor('#F1F1F3')
                      .onClick(() => this.openEditor('message'))
                    Button('删除').fontSize(14).fontColor('#E5484D').backgroundColor('#F1F1F3')
                      .margin({ left: 8 }).onClick(() => this.deleteMessage())
                  }.width('100%').justifyContent(message.outgoing ? FlexAlign.End : FlexAlign.Start)
                    .margin({ top: 8 })
                }
              }.width('100%')
            }, (message: SmsMessage) => message.id.toString())
          }.width('100%').padding({ left: 18, right: 18, bottom: 18 })
        }.layoutWeight(1).width('100%').edgeEffect(EdgeEffect.Spring)
        Row() {
          TextInput({ placeholder: '信息', text: this.reply }).fontSize(16).layoutWeight(1)
            .height(48).borderRadius(24).backgroundColor('#F3F3F5')
            .padding({ left: 17, right: 12 })
            .onChange((value: string) => this.reply = value)
            .onSubmit(() => this.sendReply())
          Button('↑').fontSize(24).fontColor(Color.White).width(44).height(44)
            .borderRadius(22).backgroundColor(GREEN).margin({ left: 10 })
            .enabled(this.reply.trim().length > 0).onClick(() => this.sendReply())
        }.width('100%').padding({ left: 18, right: 18, top: 10, bottom: 16 })
      }
    }.width('100%').height('100%').backgroundColor(Color.White)
  }

}

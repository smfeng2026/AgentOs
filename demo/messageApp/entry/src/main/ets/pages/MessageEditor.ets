import { hilog } from '@kit.PerformanceAnalysisKit';
import { SmsMessage, SmsThread } from './SmsData';
import { SmsRouteParams, SmsStore } from './SmsStore';

const INK = '#101114';
const MUTED = '#8C8E93';
const GREEN = '#32C766';

@Entry
@Component
struct MessageEditor {
  private threads: SmsThread[] = [];
  private activeId: number = -1;
  private selectedMessageId: number = -1;
  @State private draftName: string = '';
  @State private draftBody: string = '';
  @State private draftOutgoing: boolean = false;
  @State private formMode: string = 'new';
  @State private errorText: string = '';

  aboutToAppear(): void {
    const params = this.getUIContext().getRouter().getParams() as SmsRouteParams;
    this.formMode = params.mode === undefined ? 'new' : params.mode;
    this.activeId = params.threadId === undefined ? -1 : params.threadId;
    this.selectedMessageId = params.messageId === undefined ? -1 : params.messageId;
    this.reload();
    const thread = this.threads.find((item: SmsThread) => item.id === this.activeId);
    if (thread) {
      this.draftName = thread.name;
      const message = thread.messages.find((item: SmsMessage) => item.id === this.selectedMessageId);
      if (message) {
        this.draftBody = message.body;
        this.draftOutgoing = message.outgoing;
      }
    }
  }

  onPageShow(): void {
    hilog.info(0x0000, 'SmsDemo', 'onPageShow pages/MessageEditor mode=%{public}s', this.formMode);
  }

  onPageHide(): void {
    hilog.info(0x0000, 'SmsDemo', 'onPageHide pages/MessageEditor');
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

  private saveForm(): void {
    const name = this.draftName.trim();
    const body = this.draftBody.trim();
    if (!name || (this.formMode !== 'contact' && !body)) {
      this.errorText = '请填写联系人和短信内容';
      return;
    }
    if (this.formMode === 'new') {
      const id = Date.now();
      const message: SmsMessage = { id: id + 1, body: body, time: id, outgoing: this.draftOutgoing };
      const thread: SmsThread = { id: id, name: name, unread: !this.draftOutgoing, messages: [message] };
      this.threads = [thread].concat(this.threads);
      this.activeId = id;
    } else if (this.formMode === 'contact') {
      this.threads = this.threads.map((thread: SmsThread) =>
        thread.id === this.activeId ? { id: thread.id, name: name, unread: thread.unread,
          messages: thread.messages } : thread);
    } else {
      this.threads = this.threads.map((thread: SmsThread) =>
        thread.id === this.activeId ? { id: thread.id, name: thread.name, unread: thread.unread,
          messages: thread.messages.map((message: SmsMessage) =>
            message.id === this.selectedMessageId ?
              { id: message.id, body: body, time: message.time, outgoing: this.draftOutgoing } : message) } : thread);
    }
    if (!this.persist()) return;
    if (this.formMode === 'new') {
      const params: SmsRouteParams = { threadId: this.activeId };
      this.getUIContext().getRouter().replaceUrl({ url: 'pages/MessageDetail', params: params })
        .catch(() => this.getUIContext().getPromptAction().showToast({ message: '打开短信失败，请重试' }));
    } else {
      this.getUIContext().getRouter().back();
    }
  }
  build() {
    Column() {
      Row() {
        Button('取消').fontSize(16).fontColor(INK).backgroundColor(Color.Transparent)
          .onClick(() => this.getUIContext().getRouter().back())
        Blank()
        Text(this.formMode === 'new' ? '新建信息' : this.formMode === 'contact' ? '编辑联系人' : '编辑短信')
          .fontSize(18).fontWeight(FontWeight.Medium)
        Blank()
        Button('保存').fontSize(16).fontColor(GREEN).backgroundColor(Color.Transparent)
          .onClick(() => this.saveForm())
      }.height(58).width('100%').padding({ left: 12, right: 12 })
      Divider().color('#EEEEF0')
      Column() {
        if (this.formMode !== 'message') {
          Text('联系人 / 号码').fontSize(14).fontColor(MUTED).width('100%')
          TextInput({ placeholder: '输入联系人或号码', text: this.draftName }).fontSize(17)
            .height(50).backgroundColor('#F5F5F6').borderRadius(8)
            .padding({ left: 14, right: 14 }).margin({ top: 9, bottom: 22 })
            .onChange((value: string) => this.draftName = value)
        }
        if (this.formMode !== 'contact') {
          Text('短信内容').fontSize(14).fontColor(MUTED).width('100%')
          TextArea({ placeholder: '输入短信内容', text: this.draftBody }).fontSize(17)
            .height(220).backgroundColor('#F5F5F6').borderRadius(8)
            .padding(14).margin({ top: 9, bottom: 22 })
            .onChange((value: string) => this.draftBody = value)
          Row() {
            Text('短信方向').fontSize(14).fontColor(MUTED)
            Blank()
            Button('收到').fontSize(15).fontColor(this.draftOutgoing ? INK : Color.White)
              .backgroundColor(this.draftOutgoing ? '#EEEEF0' : '#92A7D0')
              .onClick(() => this.draftOutgoing = false)
            Button('发送').fontSize(15).fontColor(this.draftOutgoing ? Color.White : INK)
              .backgroundColor(this.draftOutgoing ? GREEN : '#EEEEF0')
              .margin({ left: 6 }).onClick(() => this.draftOutgoing = true)
          }.width('100%')
        }
        if (this.errorText) {
          Text(this.errorText).fontSize(14).fontColor('#D63741').margin({ top: 20 })
        }
      }.padding(20).width('100%')
    }.width('100%').height('100%').backgroundColor(Color.White)
  }

}

/**
 * Copies the bundled model out of the HAP into the app sandbox on first launch.
 *
 * MNN needs real file paths (it mmaps the weights), while resources/rawfile lives
 * inside the signed HAP archive, so the first launch pays a one-off copy of
 * ~166 MB into files/model. Later launches reuse it after a size check.
 */

import { common } from '@kit.AbilityKit';
import { fileIo } from '@kit.CoreFileKit';

export class ModelFile {
  name: string;
  size: number;

  constructor(name: string, size: number) {
    this.name = name;
    this.size = size;
  }
}

export type CopyProgress = (copiedBytes: number, totalBytes: number) => void;

export class ModelManager {
  static readonly FILES: ModelFile[] = [
    new ModelFile('config.json', 530),
    new ModelFile('llm_config.json', 3237),
    new ModelFile('tokenizer.mtok', 1710861),
    new ModelFile('llm.mnn', 953112),
    new ModelFile('llm.mnn.weight', 171481330)
  ];

  static readonly CHUNK_SIZE: number = 2 * 1024 * 1024;

  static totalBytes(): number {
    let total = 0;
    for (const file of ModelManager.FILES) {
      total += file.size;
    }
    return total;
  }

  static modelDir(context: common.UIAbilityContext): string {
    return `${context.filesDir}/model`;
  }

  static configPath(context: common.UIAbilityContext): string {
    return `${ModelManager.modelDir(context)}/config.json`;
  }

  static tmpDir(context: common.UIAbilityContext): string {
    return `${context.cacheDir}/mnn`;
  }

  private static fileSize(path: string): number {
    try {
      return fileIo.statSync(path).size;
    } catch (error) {
      return -1;
    }
  }

  private static isComplete(dir: string): boolean {
    for (const file of ModelManager.FILES) {
      if (ModelManager.fileSize(`${dir}/${file.name}`) !== file.size) {
        return false;
      }
    }
    return true;
  }

  /**
   * Guarantees the model directory is populated and returns the path of
   * config.json, ready to be handed to the inference engine.
   */
  static async ensure(context: common.UIAbilityContext, onProgress: CopyProgress): Promise<string> {
    const dir = ModelManager.modelDir(context);
    const total = ModelManager.totalBytes();

    if (!fileIo.accessSync(dir)) {
      fileIo.mkdirSync(dir, true);
    }
    const tmp = ModelManager.tmpDir(context);
    if (!fileIo.accessSync(tmp)) {
      fileIo.mkdirSync(tmp, true);
    }

    if (ModelManager.isComplete(dir)) {
      onProgress(total, total);
      return ModelManager.configPath(context);
    }

    let copied = 0;
    for (const file of ModelManager.FILES) {
      const target = `${dir}/${file.name}`;
      if (ModelManager.fileSize(target) === file.size) {
        copied += file.size;
        onProgress(copied, total);
        continue;
      }
      await ModelManager.copyOne(context, file.name, target, (bytes: number) => {
        copied += bytes;
        onProgress(copied, total);
      });
      const actual = ModelManager.fileSize(target);
      if (actual !== file.size) {
        throw new Error(`模型文件 ${file.name} 复制不完整（${actual}/${file.size} 字节）`);
      }
    }
    return ModelManager.configPath(context);
  }

  private static async copyOne(context: common.UIAbilityContext,
    name: string,
    target: string,
    onBytes: (bytes: number) => void): Promise<void> {
    const rawPath = `model/${name}`;
    try {
      // Preferred path: stream straight out of the HAP, no big allocation.
      await ModelManager.copyViaDescriptor(context, rawPath, target, onBytes);
    } catch (error) {
      // Fallback for packed rawfiles that cannot be mapped as a descriptor.
      // The descriptor attempt may have left a partial file behind, so the
      // fallback re-opens the target with TRUNC and writes from offset 0.
      console.warn(`getRawFd failed for ${name}, falling back to getRawFileContent: ${JSON.stringify(error)}`);
      await ModelManager.copyViaContent(context, rawPath, target, onBytes);
    }
  }

  private static async copyViaDescriptor(context: common.UIAbilityContext,
    rawPath: string,
    target: string,
    onBytes: (bytes: number) => void): Promise<void> {
    const raw = await context.resourceManager.getRawFd(rawPath);
    const dest = fileIo.openSync(target, fileIo.OpenMode.CREATE | fileIo.OpenMode.READ_WRITE | fileIo.OpenMode.TRUNC);
    try {
      const buffer = new ArrayBuffer(ModelManager.CHUNK_SIZE);
      let offset = 0;
      while (offset < raw.length) {
        const want = Math.min(ModelManager.CHUNK_SIZE, raw.length - offset);
        const read = await fileIo.read(raw.fd, buffer, { offset: raw.offset + offset, length: want });
        if (read <= 0) {
          break;
        }
        await fileIo.write(dest.fd, buffer, { length: read });
        offset += read;
        onBytes(read);
      }
    } finally {
      fileIo.closeSync(dest);
      await context.resourceManager.closeRawFd(rawPath);
    }
  }

  private static async copyViaContent(context: common.UIAbilityContext,
    rawPath: string,
    target: string,
    onBytes: (bytes: number) => void): Promise<void> {
    const content = await context.resourceManager.getRawFileContent(rawPath);
    const bytes = content.buffer.slice(content.byteOffset, content.byteOffset + content.byteLength);
    const dest = fileIo.openSync(target, fileIo.OpenMode.CREATE | fileIo.OpenMode.READ_WRITE | fileIo.OpenMode.TRUNC);
    try {
      await fileIo.write(dest.fd, bytes, { offset: 0 });
      onBytes(content.byteLength);
    } finally {
      fileIo.closeSync(dest);
    }
  }
}

export type SupportedCheckpointCountReader = () => unknown;

export type SupportedCheckpointInspector<T> = (index: number) => T;

export function supportedCheckpointModelsFromCatalog<T>(
  readCount: SupportedCheckpointCountReader,
  inspectAt: SupportedCheckpointInspector<T>,
): readonly T[] {
  const count = Number(readCount());
  if (!Number.isSafeInteger(count) || count < 0) {
    throw new Error(`invalid supported checkpoint count: ${count}`);
  }
  const models: T[] = [];
  for (let index = 0; index < count; index += 1) {
    models.push(inspectAt(index));
  }
  return Object.freeze(models);
}

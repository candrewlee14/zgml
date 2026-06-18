export type SessionStepArrayLike = {
  readonly length: number;
};

export type SessionStepIoFields<TValue = SessionStepArrayLike> = Readonly<{
  input: TValue | null;
  inputLen: number;
  output: TValue | null;
  outputLen: number;
}>;

export function hasSessionStepIo(
  input: SessionStepArrayLike | null | undefined,
  output: SessionStepArrayLike | null | undefined,
): boolean {
  return Boolean(input || output);
}

export function sessionStepIoFields<TValue extends SessionStepArrayLike>(
  input: TValue | null | undefined,
  output: TValue | null | undefined,
): SessionStepIoFields<TValue> | null {
  if (!hasSessionStepIo(input, output)) return null;
  const normalizedInput = input ?? null;
  const normalizedOutput = output ?? null;
  return Object.freeze({
    input: normalizedInput,
    inputLen: normalizedInput ? normalizedInput.length : 0,
    output: normalizedOutput,
    outputLen: normalizedOutput ? normalizedOutput.length : 0,
  });
}

export function sessionStepNoOutputFields<TValue extends SessionStepArrayLike>(
  input: TValue | null | undefined,
): SessionStepIoFields<TValue> | null {
  return input ? sessionStepIoFields(input, null) : null;
}

export function sessionStepIoRecord(
  input: SessionStepArrayLike | null | undefined,
  output: SessionStepArrayLike | null | undefined,
): Record<string, unknown> | null {
  const fields = sessionStepIoFields(input, output);
  if (!fields) return null;
  return {
    input: fields.input,
    input_len: fields.inputLen,
    output: fields.output,
    output_len: fields.outputLen,
  };
}

export function sessionStepNoOutputRecord(input: SessionStepArrayLike | null | undefined): Record<string, unknown> | null {
  const fields = sessionStepNoOutputFields(input);
  if (!fields) return null;
  return {
    input: fields.input,
    input_len: fields.inputLen,
    output: fields.output,
    output_len: fields.outputLen,
  };
}

export function outputLenFromStepResultRecord(result: Record<string, unknown>): number {
  return Number(result.output_len);
}

export function outputLenFromStepResultWords(result: ArrayLike<bigint>): number {
  return Number(result[0]);
}

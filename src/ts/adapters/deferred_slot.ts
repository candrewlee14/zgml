export type AdapterDeferredSlot<T> = Readonly<{
  peek(): T | null;
  bind(value: T): T;
}>;

export function createAdapterDeferredSlot<T>(name: string): AdapterDeferredSlot<T> {
  let value: T | null = null;

  function peek(): T | null {
    return value;
  }

  function bind(nextValue: T): T {
    if (nextValue == null) throw new Error(`${name} cannot be initialized with null or undefined`);
    if (value !== null) throw new Error(`${name} is already initialized`);
    value = nextValue;
    return nextValue;
  }

  return Object.freeze({
    peek,
    bind,
  });
}

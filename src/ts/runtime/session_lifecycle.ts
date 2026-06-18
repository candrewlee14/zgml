"use strict";

type DisposableResource = Readonly<{
  free: () => unknown;
}>;

type SessionHandleRecord = {
  handle: unknown;
};

type LlamaSessionLifecycleRecord = SessionHandleRecord & {
  ownedOutput?: unknown | null;
  boundOutput?: unknown | null;
};

type GenericSessionLifecycleRecord = SessionHandleRecord & {
  ownedBuffers?: unknown;
};

export type SessionLifecycleDeps = {
  sessionFree: (handle: unknown) => unknown;
  nullSessionHandle: unknown;
};

function freeHandleIfLive(session: SessionHandleRecord, deps: SessionLifecycleDeps) {
  if (session.handle !== deps.nullSessionHandle) {
    deps.sessionFree(session.handle);
    session.handle = deps.nullSessionHandle;
  }
}

function hasFree(value: unknown): value is DisposableResource {
  return !!value && typeof value === "object" && typeof (value as { free?: unknown }).free === "function";
}

function freeOwnedResource(resource: unknown) {
  if (hasFree(resource)) resource.free();
}

export function freeLlamaSessionResources(session: LlamaSessionLifecycleRecord, deps: SessionLifecycleDeps) {
  freeHandleIfLive(session, deps);
  if (session.ownedOutput) {
    freeOwnedResource(session.ownedOutput);
    session.ownedOutput = null;
  }
  session.boundOutput = null;
}

export function freeGenericSessionResources(session: GenericSessionLifecycleRecord, deps: SessionLifecycleDeps) {
  freeHandleIfLive(session, deps);
  const ownedBuffers = Array.isArray(session.ownedBuffers) ? session.ownedBuffers : [];
  for (const buffer of ownedBuffers) freeOwnedResource(buffer);
  session.ownedBuffers = [];
}

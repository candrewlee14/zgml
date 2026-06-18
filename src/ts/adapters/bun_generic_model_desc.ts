"use strict";

import {
  setUSize,
} from "./bun_abi_words.js";
import {
  genericModelNativeDesc,
  type GenericModelKindAliases,
} from "../runtime/generic_model_desc.js";
import type * as PublicApi from "../public_api.js";

export type BunGenericModelDescInput = PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc;

export function bunGenericModelDesc(desc: BunGenericModelDescInput, kinds: GenericModelKindAliases): BigUint64Array {
  const nativeDesc = genericModelNativeDesc(desc, kinds);
  const buf = new BigUint64Array(4);
  const view = new DataView(buf.buffer);
  view.setUint32(0, nativeDesc.kind, true);
  view.setUint32(4, nativeDesc.activation, true);
  setUSize(view, 8, nativeDesc.input_len);
  setUSize(view, 16, nativeDesc.output_len);
  setUSize(view, 24, nativeDesc.hidden_len);
  return buf;
}

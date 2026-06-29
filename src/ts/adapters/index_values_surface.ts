import {
  createHostAdapterIndexValuesSurface,
  type HostAdapterIndexValuesSurface,
  type HostAdapterIndexValuesSurfaceOptions,
  type HostAdapterIndexValuesTensorConstructor,
} from "../runtime/host_adapter_surfaces.js";
import type { TensorDataLike } from "../core/index_values.js";

export type AdapterIndexValuesTensorConstructor<TTensor extends TensorDataLike = TensorDataLike> =
  HostAdapterIndexValuesTensorConstructor<TTensor>;

export type AdapterIndexValuesSurfaceOptions<TTensor extends TensorDataLike = TensorDataLike> =
  HostAdapterIndexValuesSurfaceOptions<TTensor>;

export type AdapterIndexValuesSurface = HostAdapterIndexValuesSurface;

export function createAdapterIndexValuesSurface<TTensor extends TensorDataLike = TensorDataLike>(
  options: AdapterIndexValuesSurfaceOptions<TTensor>,
): AdapterIndexValuesSurface {
  return createHostAdapterIndexValuesSurface(options);
}

"use strict";

const { chooseLane, freshQsemanticThroughput, validateLane } = require("./run_next_perf.cjs");

const baseLine = [
  "perf-next:",
  "full_model=q8_0/prompt:27.4%:to90=3.3x:dispatch=242:commands=241:target=projection_chain:60:next=semantic_sublayer_or_quantized_projection_chain",
  "q8_current=prompt:29.7%:required-pass:dispatch=242:commands=151:target=semantic_ffn_sublayer:90:next=semantic_ffn_sublayer_throughput_kernel",
  "pytorch=none",
].join(" ");

const quietLine = [
  "perf-next:",
  "q8_current=prompt:90.0%:pass:dispatch=182:commands=121:target=none:next=none",
  "pytorch=none",
].join(" ");

function expectEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}

function lineWith(...fields) {
  return `${baseLine} ${fields.join(" ")}`;
}

const widthParallelLine = lineWith(
  "q8_prompt=semantic_bridge_candidate:commands=121:next=semantic_width_parallel_kernel",
  "frontier=semantic_width_parallel_kernel:candidate=ready:fresh=source:fresh,throughput=smollm:1.86x,full:2.24x",
);

const widthParallelInputBridgeLine = lineWith(
  "q8_prompt=semantic_bridge_candidate:commands=121:next=semantic_width_parallel_kernel",
  "qsemantic_input_bridge=absorbed:2.71x:median:2.74x:dispatches:5:direct_serial:1.28x:direct_serial_median:1.32x:direct_serial_dispatches:1:row_serial_dot_ops:2985984:total_row_serial_dot_ops:382205952:next=semantic_with_input_width_parallel_kernel:source=fresh",
  "frontier=semantic_width_parallel_kernel:candidate=ready:fresh=source:fresh,throughput=smollm:1.86x,full:2.24x",
);

const inputBridgeOnlyWidthLine = lineWith(
  "qsemantic_input_bridge=absorbed:2.71x:median:2.74x:dispatches:5:direct_serial:1.28x:direct_serial_median:1.32x:direct_serial_dispatches:1:row_serial_dot_ops:2985984:total_row_serial_dot_ops:382205952:next=semantic_with_input_width_parallel_kernel:source=fresh",
  "frontier=semantic_width_parallel_kernel:candidate=ready:fresh=source:fresh,throughput=smollm:1.86x,full:2.24x",
);

const staleThroughputLine = lineWith(
  "q8_prompt=semantic_bridge_candidate:commands=121:next=semantic_width_parallel_kernel",
  "frontier=semantic_width_parallel_kernel:candidate=ready:fresh=source:fresh,throughput=smollm:0.99x,full:2.24x",
);

const bridgeThroughputLine = lineWith(
  "q8_prompt=semantic_bridge_candidate:commands=121:next=semantic_bridge_throughput_kernel",
);

const inputBridgeLine = lineWith(
  "q8_prompt=semantic_bridge_candidate:commands=121:next=semantic_input_bridge_work_partitioning",
);

const steadyBridgeLine = lineWith(
  "q8_prompt=semantic_bridge_candidate:commands=121:next=steady_semantic_bridge_candidate",
);

const promotedLine = `${quietLine} q8_prompt=promoted_semantic_default:commands=121:next=steady`;

const pytorchLine = `${quietLine} pytorch=softmax_classifier_batched:1.03x`;

expectEqual(chooseLane(widthParallelLine, {}), "qsemantic_bridge", "width-parallel Q8 prompt target routes to exact bridge microscope");
expectEqual(chooseLane(widthParallelInputBridgeLine, {}), "qsemantic_bridge", "full Q8 prompt width target wins over adjacent input-bridge microscope");
expectEqual(chooseLane(inputBridgeOnlyWidthLine, {}), "qsemantic_input_bridge", "input-bridge width-parallel target routes to exact input-bridge microscope when Q8 prompt has no width target");
expectEqual(chooseLane(staleThroughputLine, {}), "qsemantic_throughput", "stale width frontier refreshes qsemantic throughput before exact bridge");
expectEqual(chooseLane(bridgeThroughputLine, {}), "qsemantic_throughput", "semantic bridge throughput target routes to qsemantic throughput");
expectEqual(chooseLane(inputBridgeLine, {}), "q8_prompt_semantic", "input bridge partitioning routes to full-model semantic lane");
expectEqual(chooseLane(steadyBridgeLine, {}), "q8_prompt", "steady bridge target routes to q8 prompt");
expectEqual(chooseLane(promotedLine, {}), "ggml", "promoted semantic default routes to ggml smoke");
expectEqual(chooseLane(pytorchLine, {}), "pytorch", "pytorch gap routes to PyTorch comparison");
expectEqual(chooseLane(baseLine, { BENCH_NEXT_PERF_LANE: "qsemantic_bridge" }), "qsemantic_bridge", "forced lane wins");

const throughput = freshQsemanticThroughput(widthParallelLine);
expectEqual(throughput && throughput.smollm, 1.86, "fresh qsemantic smollm throughput parse");
expectEqual(throughput && throughput.full, 2.24, "fresh qsemantic full throughput parse");

for (const lane of ["status", "pytorch", "qsemantic", "qsemantic_throughput", "qsemantic_bridge", "qsemantic_input_bridge", "qproj", "q8_prompt", "q8_prompt_semantic", "ggml"]) {
  validateLane(lane);
}

console.log("next-perf routing gate: pass");

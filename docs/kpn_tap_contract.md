# KPN Tap Contract Specification

## Rationale

The KPN runtime treats taps as first-class, addressable FIFOs that sit on the node’s wheel. Each tap belongs to a **tap group** that describes the shape, indexing, buffering rules, and delivery semantics for a family of per-channel FIFOs. Groups allow nodes and the wheel to negotiate capacity, multicast policy, and activation without assuming a fixed topology or Python tooling. All tap activation and buffer allocation happen lazily: no wheel memory is consumed for a channel unless either the descriptor marks it enabled or an edge connects to it.

## Contract Overview

Nodes publish a `tap_groups` array alongside the legacy `taps.inputs/outputs` fields. `tap_groups` exposes the fine-grained layout that the streamer uses when constructing per-channel FIFOs.

```jsonc
{
  "tap_groups": [
    {
      "name": "fft_io",
      "unit_shape": {"frames": 2048},
      "dtype": "f64",
      "buffer": {
        "prealloc_units": 32768,
        "layout": "row-major"
      },
      "delivery": {
        "mode": "fifo_pc",             // or "multicast"
        "full_percent_target": 85,
        "low_watermark_percent": 30,
        "multicast": {
          "ack_policy": "all",        // "all" | "any" | "quorum"
          "quorum_k": 2
        },
        "fifo_pc": {
          "release_policy": "on_consume"  // "on_consume" | "on_publish"
        }
      },
      "indexing": {
        "major_order": ["channel", "band", "subchannel", "unit"],
        "band_indexing": "dense_active"   // or "absolute"
      },
      "channels": [
        {
          "name": "band",
          "enabled": true,
          "count": 4096,
          "active_mask_b64": null,
          "active_indices": [0, 1, 2, 3],
          "subchannels": [
            {"name": "mag",   "enabled": true,  "stride_elems": 1},
            {"name": "phase", "enabled": false, "stride_elems": 1},
            {"name": "power", "enabled": false, "stride_elems": 1}
          ],
          "unit_shape": {"frames": 1}
        },
        {
          "name": "audio",
          "enabled": true,
          "count": 1,
          "active_indices": [0],
          "subchannels": [
            {"name": "mono", "enabled": true, "stride_elems": 1}
          ],
          "unit_shape": {"frames": 2048}
        }
      ]
    }
  ]
}
```

### Field Notes

- **unit_shape**: Defines the elemental tensor produced by each active tap instance. Shapes follow the runtime’s `(batches, channels, frames)` convention; omitted dimensions default to `1`.
- **buffer.prealloc_units**: Maximum number of `unit_shape` elements the wheel may cache for this group. Allocation is deferred until at least one tap slot is active.
- **delivery.mode**: `fifo_pc` for single producer→consumer semantics with primary-consumer release, or `multicast` to allow multiple readers. Delivery blocks include policy knobs that the wheel honours when sizing rings and handling acknowledgements.
- **direction**: Identifies whether the group publishes (`"output"`) or consumes (`"input"`) data. Input groups may declare hold/default behaviour for control-style taps.
- **indexing.major_order**: Specifies how the wheel flattens multi-dimensional indices into the preallocated buffer. The runtime iterates the listed keys in order (e.g., all active bands before moving to the next subchannel).
- **channels[count]**: Describes the address space for each logical channel family. `active_indices` and `active_mask_b64` enable sparse activation without hard-coding contiguous ranges. Every enabled `(channel, band-index, subchannel)` combination becomes an individually addressable tap.
  - `hold_if_absent`: Inputs may request that the runtime reuse the most recently observed frame when upstream edges momentarily run dry. Outputs ignore this flag.
  - `optional_default`: Inputs can seed an initial value (scalar or tensor) before any frames arrive. The object may carry a `value` field and optional metadata such as `units` or a `unit_shape` override. When present, the runtime uses this default until real data arrives.
- **subchannels**: Each subchannel represents a unique tap per channel element (for example, `mag`, `phase`, `power`). `stride_elems` describes how many scalar elements belong to the subchannel within the parent unit.

### Activation & Allocation

- The streamer instantiates a wheel FIFO only for taps that are *both* enabled and referenced by at least one edge. Removing the final edge from a tap automatically deactivates it and returns its buffer to the group pool.
- Groups share the group-level buffer budget. When a new tap activates, the wheel requests a slice from the group’s `prealloc_units`. If capacity runs out, the group can negotiate (see below) or deny the activation.
- Host-visible tap buffers expose 1D data covering the entire render lifetime. No block stitching is required; the runtime writes directly into the preallocated cache in major-order sequence.

## Negotiation Handshake

Tap groups and the wheel cooperate through a lightweight negotiation channel:

1. **Descriptor Proposal** – Nodes declare their preferred delivery mode, buffer targets, and activation hints in the `tap_groups` block.
2. **Wheel Acceptance** – During plan load, the wheel initialises each group using the proposal. If hardware or global policy forces adjustments (e.g., enforce multicast), the wheel records an override and feeds it back to the node.
3. **Dynamic Requests** – At runtime a node may request changes (e.g., enable a new band, grow capacity, switch `delivery.mode`). Requests flow through the wheel’s negotiation API, which applies backpressure or returns a revised contract when immediate fulfilment is impossible.
4. **Automatic Downgrade** – When a tap loses every edge, the wheel automatically deactivates it and notifies the node so it can cease publishing data.

Negotiation is bilateral: both the node and the wheel maintain copies of the contract and must acknowledge changes before data flow resumes.

## Legacy `taps` Array

The historical `taps.inputs/outputs` array remains for backward compatibility. Legacy descriptors may omit `tap_groups`; in that case the runtime synthesises a single group per tap with the original shape/capacity defaults. New nodes should populate both sections until all toolchains consume the group schema directly.

## Action Items

1. **Descriptor Generation** – Emit `tap_groups` for every native node. Legacy taps should be auto-derived from the group definitions during descriptor build.
2. **Runtime Parser** – Extend `graph_runtime.cpp` to hydrate the new group metadata, lazily allocate wheel FIFOs, and honour delivery overrides in negotiation replies.
3. **Streamer Negotiation** – Implement the negotiation API between nodes and the wheel so delivery-mode switches (e.g., FIFO→multicast) and buffer resizing requests are serviced deterministically.
4. **Host API** – Provide lookup helpers that let demos fetch tap buffers by `(group, channel, band, subchannel)` without manually flattening indices.
5. **Documentation & Tests** – Update native tests and demos to exercise tap activation, deactivation, and negotiation. Python fallbacks remain unsupported.

## Node Package Layout

Native nodes now ship with a standard package directory under `src/native/nodes/<node>/`. Each package keeps the tap contract, presets, and (eventually) the node implementation together:

```text
src/native/nodes/<node>/
  README.md
  contracts/*.json
  presets/*.json
  src/        # native sources once they are extracted from amp_kernels.c
```

The KPN demo nodes (`parametric_driver`, `oscillator`, `mix`, `fft_division`) already conform to this structure. Build tooling copies the entire tree next to the compiled binaries so future dynamic graph tooling can hot-load presets and contracts without rebuilding the runtime.

Further revisions will continue to live in this document as the contract evolves.

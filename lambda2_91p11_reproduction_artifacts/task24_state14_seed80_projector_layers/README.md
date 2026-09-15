# Workstation projector-layer replay: task 24, state 14, seed 80

This diagnostic uses the saved query-0 and query-1 BF16 `vision_encoder_output` tensors from the attention-kernel profile. It loads the original Lambda=2 checkpoint, keeps the model in BF16, and applies the original `FusedMLPProjector` without running the simulator or vision encoders.

The projector is:

```text
Linear(2176 -> 8704) -> GELU -> Linear(8704 -> 896) -> GELU -> Linear(896 -> 896)
```

For both queries, the replayed fifth-layer output is **byte-for-byte identical** to the projector output archived during the original end-to-end trace:

| Query | Final projector SHA256 | Exact match | Maximum absolute difference |
|---:|---|---|---:|
| 0 | `bd1bdef0f02c48773ec884bb25321f65b44a976fb89762bb25977331d5a72c33` | Yes | `0.0` |
| 1 | `bf8397e0a1af4308b80d9da62330e39c654464767af82a4e8183c418e6dc2e0a` | Yes | `0.0` |

## Layer-output hashes

| Layer | Query 0 raw-tensor SHA256 | Query 1 raw-tensor SHA256 |
|---|---|---|
| Linear `2176 -> 8704` | `d5a65a99d003730f34a62fc23f9e4124378e9b5996842453b5b21e19a8ce48b9` | `737292412c7c3dbfdd7574540877fae36884f0a34e5e5f92dfeadea6576c1af2` |
| GELU | `7b44c145d4ab19ee3cd3a78d413ad252bcaae39ae265889804b724211e3f2d00` | `14bec3bcc35dff7085e6eac0e7a8459f0bad186e7c8c9c6b5f3f1af1088f3335` |
| Linear `8704 -> 896` | `d3438a57ae2496f50ccf7bda3a930af19c276b6d93f760a5069e35f9d1750258` | `bced17e1dcca3489a5a0b99a9bc7a5543bb9845ea6f030d32513051e22ca1d69` |
| GELU | `c25ef9ff006c0612b669e7edba2a48f780d99871b44e93a277e41a00026ff9fd` | `49a50814d0c04a327b551f39aaa68c95c75e75bc993c625f41276a88c7e66c43` |
| Linear `896 -> 896` | `bd1bdef0f02c48773ec884bb25321f65b44a976fb89762bb25977331d5a72c33` | `bf8397e0a1af4308b80d9da62330e39c654464767af82a4e8183c418e6dc2e0a` |

## Live projector fingerprint

- Class: `prismatic.util.nn_utils.FusedMLPProjector`
- Combined live parameter hash: `fe13bd8f3745da1088fc1ac4d32bc6243f53073a1722daf3b212a239dfa6b768`
- Source: `prismatic/util/nn_utils.py`
- Source SHA256: `7aa0562eab374f0585bb53639b861f06b6c44b82384e938dda9cd32d844bc0d0`
- Checkpoint SHA256: `0b0ef328c8d4f5478e6da25fde7639c688fdd1ac3b041c60ea06476424b52b1d`

`projector_layer_report.json` contains the six individual live parameter hashes, module description, source fingerprint, tensor shapes/dtypes/ranges, and every layer-output hash. The two `.pt` files contain the saved vision input, all five intermediate outputs, and the archived final output. `workstation_projector_layers.py` is the exact capture script.


---
name: user-prefers-exact-reference-replica
description: When given a reference (e.g. an HF Space), user wants the result to
  replicate it exactly, not a simplified version
metadata:
  node_type: memory
  type: user
  originSessionId: sess_0a3c3bc7-a4da-4e36-a726-44f5d483d5aa
---

When the user supplies a reference implementation (like the baidu/Unlimited-OCR HF Space), they chose "как в референсе" over simpler Gradio/CLI alternatives.

**Why:** They value fidelity to the original look and behavior more than simplicity.

**How to apply:** Default to porting the reference's own UI/architecture; offer simplified alternatives only as options, not as the plan.

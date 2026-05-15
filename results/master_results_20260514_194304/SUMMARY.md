# Experiment Results Summary

**Generated:** 2026-05-15 08:11:23
**Dataset:** creditcard
**Total Methods:** 6

## Execution Status

| Status | Count |
|--------|-------|
| ✓ Completed | 6 |
| ✗ Failed | 0 |
| ⚠ Skipped | 0 |

## Phase 4 Results (Partial Embedding — X₁, X₂, X₃')

| Method | F1-Score | Retention | Status |
|--------|----------|-----------|--------|
| Axis_Aligned_Embeddings | None | None | ✓ |
| gradient_hessians | None | None | ✓ |
| grouped_encoder | None | None | ✓ |
| joint_optimized | None | None | ✓ |
| leaf_distillation | None | None | ✓ |
| random_projection | None | None | ✓ |

## File Locations

- Master log: see `orchestrator.log` in this directory
- Per-method logs: `{method_name}.log` in this directory
- Status files: `{method_name}_status.json` in this directory

## Next Steps

1. Review Phase 4 results (Partial Embedding) — primary metric
2. Compare retention rates across methods
3. Examine Phase 5 variant results for top-performing methods

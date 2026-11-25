# Potatostics Release Readiness Checklist

Record results for the intended source or Windows release and attach the logs to the candidate. Include the Git revision and worktree state, operating system, Python and dependency versions, and SHA-256 hashes of the checked artifacts.

| Check | Evidence to record |
| :--- | :--- |
| **CI matrix** | Successful runs for the candidate revision on the Windows, macOS, and Ubuntu matrix. CI records the source regression suite and the installed-wheel checks separately. See [the workflow](../.github/workflows/ci.yml). |
| **Source archive** | Record the source revision and archive SHA-256. Verify the generated inventory (`ARCHIVE_CONTENTS.json`) against the approved release policy (`release_manifest.json`). The verifier checks allowed paths, required entries, file hashes, and extraction. |
| **Installed wheel** | A fresh environment with resolved runtime dependencies, `pip check`, package imports from that environment, and a successful console entry point. Load installed example datasets and run the Qt smoke from outside the checkout. |
| **Source regression suite** | Pytest totals with failures, skips, and their reasons. Headless Qt tests cover command completion, data-grid edits, session restore, export, and cancellation. |
| **Econometric benchmarks** | The candidate's [benchmark results](../tests/data/benchmark_results/benchmark_report.md) with tested samples, tolerances, and dependency versions. The battery includes external Python references, analytical checks, simulation, and internal consistency. A [live reference-software comparison](../tests/data/reference_outputs/README.md) requires supplied execution logs. |
| **Windows executable** | Verify the current frozen executable and its `build_metadata.json`, confirm the bundled `_internal/LICENSE` matches the source, and run the automated offscreen smoke. On a Windows desktop, open that same executable, load example data, run a numerical command, and close it cleanly. |

## Running the acceptance gates

From the repository root, use the source target for a source archive and wheel, or the Windows target when distributing a frozen executable.

```bash
python scripts/run_release_acceptance.py --target source --install-mode isolated
python scripts/run_release_acceptance.py --target windows --install-mode isolated --build-exe
```

The acceptance script requires every mandatory gate for the selected target to report `PASS`. Clean dependency resolution is required. An installed-wheel or checkout smoke can establish offscreen application behavior for that artifact. The frozen Windows executable needs its own recorded build and runtime evidence.

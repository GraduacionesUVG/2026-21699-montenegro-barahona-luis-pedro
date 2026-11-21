# Memory Poisoning Analysis

Reproducible black-box experiments for studying how long-term-memory systems
accept, preserve, retrieve, and act on poisoned information.

The harness compares memory mechanisms through a common adapter contract and a
shared trial lifecycle. The current experiments focus on **TeleMem**,
**LangMem**, and **Dakera**, with direct-write and mechanism-native extraction
conditions where the mechanism supports them. PersonaMem-v2 supplies bounded
benign histories; attacks are introduced as conversation events or frozen
external sources such as synthetic email.

No experiment sends real messages or performs real external writes. Behavioral
tests use allowlisted mock tools and a sealed simulated outbox.

## What the benchmark measures

Each attack is evaluated as a sequence of increasingly strong outcomes:

1. **Exposure**: the attack reached the intended ingress path.
2. **Injection**: the poisoned marker became visible in memory state.
3. **Retrieval**: the marker survived benign activity and was returned by a
   frozen query.
4. **Behavioral effect**: where a behavioral task is enabled, the poison changed
   a simulated action.

The main attack families are:

- **False-fact insertion**: add an untrusted fact and test storage/retrieval.
- **Stored instruction**: persist a routing or action instruction and optionally
  test it with a simulated tool.
- **Information replacement**: seed a trusted value, inject an alternative, and
  distinguish preservation, coexistence, retrieval displacement, and observable
  replacement.

Every case is normally executed as three isolated twins:

- `clean`
- `matched_benign`
- `poisoned`

The twins share the same benign replay and configuration. Only the frozen
external delivery or conversation overlay differs.

## Architecture

```text
Scenario or campaign configuration
                |
                v
       Campaign preparation
  freeze inputs, IDs, hashes, bindings
                |
                v
            TrialRunner
 replay -> optional baseline -> ingress -> quiescence
       -> public task -> evaluation -> reset
                |
                v
        MemoryAdapter protocol
 initialize | direct/native write | retrieve | inspect
 flush/wait | provenance | reset | close
        /            |             \
       v             v              v
 TeleMemAdapter  LangMemAdapter  DakeraAdapter
       |             |              |
 TeleMemRuntime  LangMemRuntime  DakeraRuntime
                |
                +---- AgentRunner ---- Mock tools
                                      |
                                      v
                              sealed simulated outbox
```

The runtime owns library/service-specific behavior. The adapter converts that
behavior into canonical records and evidence. `TrialRunner` applies the same
lifecycle and validation rules across mechanisms.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/membench/memory/` | Common memory models, protocol, policies, and adapters |
| `src/membench/trial/` | Replay and trial lifecycle |
| `src/membench/agent/` | Bounded responder/action loop and tool policy |
| `src/membench/campaign/` | Campaign preparation, loading, launching, and aggregation |
| `src/membench/scenario/` | Compact scenario compiler and generated campaign bridge |
| `src/membench/curation/` | Projection, neutral-host, and delivery builders |
| `src/membench/evaluation/` | Deterministic probes, scoring, artifacts, and aggregation |
| `src/membench/tools/` | Synthetic fixtures, inbox, messaging, and outbox evidence |
| `src/configs/scenarios/` | Compact attack scenarios |
| `src/configs/scenario_condition_profiles.json` | Model, mechanism, policy, agent, and child-environment profiles |
| `src/configs/fixtures/` | Frozen synthetic contacts/documents used by mock tools and validation |
| `src/configs/campaigns/` | Explicit/legacy campaign preparation documents |
| `src/data/blackbox/` | Generated PersonaMem-v2 inventories and projections (ignored by Git) |
| `src/artifacts/` | Prepared campaigns, child evidence, runtime state, and results (ignored by Git) |

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Ollama for the local model profiles
- Windows PowerShell for the commands below
- Optional Docker/PostgreSQL for persistent LangGraph qualification
- Access to the selected memory mechanism libraries/services

## Project location

The complete benchmark project is located under the parent repository's
`src/` directory. Unless otherwise stated, run all commands from that directory:

```powershell
cd .\src
```

## Installation

Run installation commands from the project root: the directory containing
`pyproject.toml` and `uv.lock`.

### General development environment

Create the Python 3.12 interpreter and the general `.venv` used for the CLI,
scenario preparation, and tests:

```powershell
uv python install 3.12
uv sync --locked --group dev --group responder-tokenizer
Copy-Item .env.example .env
uv run membench doctor
uv run python -c "import membench; import membench.scenario; import membench.campaign; import membench.agent; print('Imports OK')"
```

Dataset reconstruction additionally requires the optional `data-processing`
group:

```powershell
uv sync --locked --group dev --group responder-tokenizer --group data-processing
```

On Windows, deeply nested trial evidence can exceed the legacy path limit. Use
a short pytest base directory when necessary:

```powershell
$pytestRoot = Join-Path $env:USERPROFILE "mbt"
New-Item -ItemType Directory -Force $pytestRoot | Out-Null
$env:TEMP = $pytestRoot
$env:TMP = $pytestRoot
$env:TMPDIR = $pytestRoot
uv run pytest --basetemp "$pytestRoot\run"
```

### Mechanism child environments

Campaign trials do not execute inside the general `.venv`. Create the isolated
child environment for every mechanism selected by a condition profile:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venvs\telemem"
uv sync --locked --no-default-groups --group telemem-runtime --group responder-tokenizer

$env:UV_PROJECT_ENVIRONMENT = ".venvs\langgraph"
uv sync --locked --no-default-groups --group langgraph-runtime --group responder-tokenizer

$env:UV_PROJECT_ENVIRONMENT = ".venvs\dakera"
uv sync --locked --no-default-groups --group dakera-runtime --group responder-tokenizer

Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

Do not copy virtual environments between machines. Recreate them from
`pyproject.toml` and `uv.lock`..

## Model setup

The default local setup uses Qwen2.5 for chat/writing and Nomic for embeddings:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text:137m-v1.5-fp16

uv run membench ollama status
uv run membench ollama verify
```

Copy the model digests reported by `membench ollama status` into the relevant
condition profile. A model name and digest mismatch invalidates provenance and
may prevent preparation or execution.

Other profiles in this repository use Mistral-Nemo and GPT-OSS. Pull only the
models required by the selected profile and ensure the corresponding tokenizer
revision is available before running a campaign.

## Environment configuration

`.env.example` documents the supported settings. At minimum, verify:

- `OLLAMA_BASE_URL`
- `OLLAMA_OPENAI_BASE_URL`
- chat and embedding model names/digests
- responder and action-decider settings
- TeleMem or LangMem storage roots
- mechanism visibility/transport timeouts
- tokenizer cache and local-files-only behavior

`.env` is intentionally ignored and must not be committed. Campaign preparation
passes only variables explicitly allowlisted by the selected child environment.

## Runtime qualification

Qualify a mechanism before running measured campaigns:

```powershell
# TeleMem: direct and native-selective writes
.\.venvs\telemem\Scripts\membench.exe telemem status
.\.venvs\telemem\Scripts\membench.exe telemem verify

# LangMem: local in-memory smoke test or configured persistent store
.\.venvs\langgraph\Scripts\membench.exe langmem status
.\.venvs\langgraph\Scripts\membench.exe langmem smoke
.\.venvs\langgraph\Scripts\membench.exe langmem verify
```

Inspect the policy combinations declared by the harness:

```powershell
uv run membench matrix
```

Qualification proves that the configured integration can write, retrieve, and
isolate a test namespace. It is not an attack experiment.

## Running a generated scenario

Generated scenarios are the preferred authoring path because one scenario owns
the persona, attack variants, ingress boundary, evaluation markers, and condition
matrix.

1. Select or copy a scenario under `configs/scenarios/`.
2. Select condition profile IDs from
   `configs/scenario_condition_profiles.json`.
3. Configure `ScenarioGeneratedCampaignSources` in `main.py`:
   - the matching fixture;
   - a new generated-input directory;
   - a new external-artifact directory;
   - a new campaign artifact root;
   - optional behavioral task/oracle paths.
4. Prepare the campaign:

```powershell
uv run python .\main.py
```

5. Run the prepared campaign using the exact artifact root configured in
   `main.py`:

```powershell
uv run membench campaign run `
  artifacts/campaigns/<unique-campaign-directory> `
  --project-root . `
  --env-file .env `
  --json
```

The generated scenario runner currently supports `forced_ingestion`. Use an
explicit campaign configuration for existing agent-selected ingress studies.

Generated input, external artifact, and campaign directories must be new. The
builders intentionally refuse to overwrite frozen evidence. Use a new suffix for
each rerun rather than editing an existing artifact directory.

### Behavioral stored-instruction scenarios

To measure tool behavior, provide both:

```python
behavioral_task_path=Path("configs/behavioral_tasks/<task>.json")
behavioral_oracle_path=Path("configs/behavioral_oracles/<oracle>.json")
```

Retrieval-only stored-instruction scenarios may omit both. Such runs can measure
injection and retrieval, but they cannot establish behavioral attack success.

## Running an explicit campaign configuration

Existing JSON campaign documents can be prepared and run directly:

```powershell
uv run membench campaign prepare `
  configs/campaigns/<campaign>.json `
  --project-root . `
  --env-file .env `
  --json

uv run membench campaign run `
  artifacts/campaigns/<artifact-root-from-config> `
  --project-root . `
  --env-file .env `
  --json
```

Preparation validates and freezes the campaign but does not execute trials. The
run command only accepts a complete, unstarted preparation.

## Building external delivery artifacts manually

For explicit campaigns, freeze editable attack specifications and email recipes
with:

```powershell
uv run membench external build `
  --attack-draft configs/external_attack_specs/<attack>.json `
  --recipe configs/external_delivery_recipes/<recipe>.json `
  --fixture configs/fixtures/<fixture>.json `
  --output-dir artifacts/external/<new-artifact-set> `
  --json
```

The command validates the fixture relationships, renders clean,
matched-benign, and poisoned deliveries, and writes immutable manifests. Do not
edit the frozen output. Modify the editable source files and build a new output
directory instead.

## Results and evidence

Completed campaign roots contain:

- `campaign_result.json`: child execution outcomes;
- `analysis.json`: twin-level campaign aggregation;
- `evaluation_analysis.json`: deterministic evaluation summary when enabled;
- `checksums.json`: sealed artifact hashes;
- `children/<run-id>/trial/`: replay, ingress, memory state, retrieval, agent,
  tool, provenance, reset, and evaluation evidence.

Important child artifacts include:

- `forced-ingress-result.json` or agent-selected inbox evidence;
- `snapshot-external_ingress.json`;
- `evaluation/<probe-id>/state-before.json`;
- `evaluation/<probe-id>/marker-matches.json`;
- `evaluation/<probe-id>/execution.json`;
- `agent-result.json`;
- `outbox-snapshot.json`;
- `provenance.json` and `reset.json`.

An empty retrieval-marker hit list means the marker was not returned by that
specific frozen query. It does not, by itself, prove that the value was absent
from memory; inspect the complete post-ingress state evidence as well.

## Reproducing PersonaMem-v2 dataset curation

The curation pipeline is deterministic once the upstream revision, seed,
selection policy, and manual review decisions are fixed.

Raw PersonaMem-v2 exports contain benchmark labels and synthetic sensitive
information. They are evaluator-only: do not expose whole raw documents to the
tested agent, publish them unintentionally, or commit them to Git. Review the
upstream dataset terms before redistribution.

### 1. Install the data dependency

```powershell
uv sync --locked --group dev --group data-processing
```

### 2. Export a pinned persona sample

The repository uses the pinned PersonaMem-v2 revision
`ed956dea41521fc4499acbc63f966e0fd3c053ba`, sample size 10, and seed 42:

```powershell
uv run python -m membench.data_process.data_processing `
  --revision ed956dea41521fc4499acbc63f966e0fd3c053ba `
  --sample-size 10 `
  --seed 42 `
  --output-dir src/membench/raw_data
```

This reads the linked benchmark, raw-persona, and 32k chat-history files and
writes one independent `persona_<id>.json` document per selected persona. Each
document records source hashes and selection provenance.

### 3. Normalize the candidate inventory

```powershell
uv run python -m membench.data_process.candidate_inventory `
  --input-dir src/membench/raw_data `
  --output-dir data/blackbox/inventory
```

The inventory minimizes public text, records row/persona provenance, links
candidate evidence to source messages, computes token counts/hashes, and marks
structural eligibility.

### 4. Rank candidates for attack methodologies

```powershell
uv run python -m membench.data_process.candidate_selection `
  --inventory-dir data/blackbox/inventory `
  --raw-dir src/membench/raw_data `
  --output-dir data/blackbox/selection `
  --top-k 10
```

Candidates are deterministically scored for the supported conversation attack
methodologies. Every fit is classified as `eligible`, `review_required`, or
`ineligible`; warnings and exclusion reasons remain in evaluator-only outputs.

### 5. Allocate candidates to cases

Review `data/blackbox/allocation-configs/pmv2_native_v1.json`, especially target
counts, candidate reuse, per-persona limits, disabled methods, and approved
review-required IDs. Then run:

```powershell
uv run python -m membench.attacks.allocation `
  --selection-dir data/blackbox/selection `
  --config data/blackbox/allocation-configs/pmv2_native_v1.json `
  --output-dir data/blackbox/allocation-native-v1
```

The allocation is deterministic and fails if required quotas cannot be
satisfied. Do not silently substitute candidates after freezing it.

### 6. Materialize projection decisions

```powershell
uv run python -c "import membench.campaign.models; from membench.curation.artifacts import main; raise SystemExit(main())" `
  --allocation-dir data/blackbox/allocation-native-v1 `
  --inventory-dir data/blackbox/inventory `
  --raw-dir src/membench/raw_data `
  --output-dir data/blackbox/projection-draft `
  --projection-id pmv2-native-projection-draft-v1
```

This stage deterministically calculates evidence boundaries, insertion points,
attacker-visible context, survival delay, overlap warnings, and per-decision
hashes.

### 7. Review the draft

Before treating attacks as measured inputs, inspect:

- evidence and insertion indexes;
- generated clean, matched-benign, and poisoned text;
- survival-delay messages;
- probe answerability;
- Unicode normalization and encoding artifacts;
- future/private-event leakage;
- target overlap and other warnings;
- contradictions such as explicit forget requests.

Record exclusions or approvals explicitly. A `review_required` or
`generated_unreviewed` result is not automatically a valid measured case.

### 8. Build paired executable projections

For a byte-compatible unreviewed reproduction of the deterministic draft:

```powershell
uv run python -c "import membench.campaign.models; from membench.curation.projections import main; raise SystemExit(main())" `
  --draft-dir data/blackbox/projection-draft `
  --raw-dir src/membench/raw_data `
  --output-dir data/blackbox/projections-v1
```

The output contains `cases.jsonl`, `attack_specs.jsonl`, three trial projections
per included case, public probes, private labels, exclusions, configuration, and
an integrity manifest. For a measured conversation-attack set, human approval
must be represented explicitly rather than inferred from successful generation.

### 9. Derive neutral external-ingress hosts

External-email experiments reuse benign source events but remove every generated
conversation overlay. Configure the mapping in
`configs/neutral_hosts/external-hosts-v1.json`, then run the following from the
repository root:

```powershell
@'
from pathlib import Path

# Bootstrap the shared campaign models before importing the curation package.
import membench.campaign.models

from membench.curation.neutral_hosts import (
    build_neutral_host_projection_set,
    load_neutral_host_config,
)
from membench.curation.projections import export_projection_set

config = load_neutral_host_config(
    "configs/neutral_hosts/external-hosts-v1.json"
)
projection_set = build_neutral_host_projection_set(
    config,
    source_projection_directory="data/blackbox/projections-v1",
)
export_projection_set(
    projection_set,
    Path("data/blackbox/neutral-external-hosts-v1"),
)
'@ | uv run python -
```

The neutral-host builder:

- copies only source-derived events from the clean source trial;
- removes generated attack overlays;
- creates identical memory-visible replay for all twins;
- assigns new case, trial, event, probe, and label IDs;
- preserves source indexes as provenance;
- recomputes hashes and validates twin signatures.

When frozen delivery sets already exist, pass their directories through
`delivery_projection_directories` so the builder also verifies that each
`available_after_source_message_index` exists in its host replay.

### 10. Validate the reconstructed curation pipeline

```powershell
uv run pytest `
  tests/test_candidate_inventory.py `
  tests/test_candidate_selection.py `
  tests/test_attack_allocation.py `
  tests/test_curation_policy.py `
  tests/test_projections.py `
  tests/test_neutral_hosts.py
```

Keep the source revision, configuration files, manifests, and SHA-256 values
with any reported results. A changed input should produce a new artifact set;
do not repair immutable artifacts in place.

## Reproducibility and interpretation rules

- Keep model name, digest, tokenizer revision, mechanism version, and embedding
  dimensions frozen per condition.
- Use a fresh run namespace and storage root for every child.
- Require empty initial state, flush/quiescence evidence, provenance export, and
  verified cleanup.
- Do not count preparation failures, transport failures, or incomplete twins as
  attack resistance.
- Do not equate storage with retrieval or retrieval with behavioral obedience.
- For information replacement, report coexistence separately from observable
  overwrite/replacement.
- Preserve all three twins for measured comparisons, even when pilot execution
  time is high.

## Common failures

| Failure | Likely cause |
| --- | --- |
| `FileExistsError` during preparation/build | Frozen output directory already exists; choose a new directory |
| `NotADirectoryError` while loading | Campaign run command points to a file or incorrect prepared root |
| Missing environment variables | A condition profile declares inherited keys absent from `.env`/process environment |
| Model digest mismatch | Ollama model differs from the frozen profile |
| Marker stored but not retrieved | Query/ranking mismatch or payload below top-k, not necessarily failed storage |
| Quiescence failure | Mechanism state did not stabilize or complete inspection could not be demonstrated |
| Empty behavioral outbox | No behavioral task, no permitted tool call, or the agent chose not to act |

## Safety

All external sources, contacts, documents, recipients, and tool actions are
synthetic. `simulate_send` records an evaluator-only outbox item and never
contacts a real service. Keep `allow_external_side_effects` disabled for this
benchmark.

"""Verify, select, and freeze existing projections; never build or run trials.

Preparation calls load_campaign_inputs before creating its artifact root, then
freeze_campaign_inputs with that root's store. Matrix construction consumes
CampaignInputs.cases; private answer keys are intentionally absent there.
The complete source bundle is frozen even for a subset selection, preserving
its original manifest and exclusions. Selection is recorded separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from membench.campaign.artifacts import CampaignArtifactStore
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
    TrialTwin,
)
from membench.campaign.preparation_models import (
    CampaignPreparationOptions,
    CampaignPurpose,
)
from membench.curation.projections import (
    PROJECTION_FILES,
    AttackReviewStatus,
    AttackSpec,
    EventStage,
    ProjectionCase,
    ProjectionKind,
    ProjectionSet,
    ProjectionSetError,
    ProjectionSetManifest,
    PublicProbe,
    TrialProjection,
    memory_visible_replay_signature,
    parse_projection_set,
)
from membench.memory.models import CanonicalModel
from membench.tools.fixtures import (
    ToolFixtureBundle,
    ToolFixtureError,
    fixture_required,
    parse_tool_fixture,
)

_DESTINATION = "inputs/projections"
_TOOL_FIXTURE_DESTINATION = "inputs/tools/fixture.json"


class CampaignInputError(RuntimeError):
    """Source projections or the requested selection cannot be used safely."""


class CampaignInputCase(CanonicalModel):
    """A complete selected twin family for the future matrix, without answer keys.

    This is trusted orchestration data, not a model prompt. Replay and probe
    builders must still project only their allowed fields into model inputs.
    """

    case: ProjectionCase
    attack_spec: AttackSpec
    probe: PublicProbe
    trials: tuple[TrialProjection, ...]

    def trial_for(self, twin: TrialTwin) -> TrialProjection:
        """Resolve one of the three twins without deriving IDs from filenames."""
        for trial in self.trials:
            if trial.twin == twin:
                return trial
        raise KeyError(twin)


@dataclass(frozen=True)
class CampaignInputs:
    """Validated source snapshot and ordinal-ordered selected families.

    Raw payloads include evaluator-only labels and must never go to a model.
    Bytes are retained so edits to the source after loading cannot change the
    later freeze. This is an in-process value, not a serialized child request.
    """

    campaign_id: str
    preparation_sha256: str
    purpose: CampaignPurpose
    source_directory: Path
    manifest: ProjectionSetManifest
    cases: tuple[CampaignInputCase, ...]
    _payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    tool_fixture: ToolFixtureBundle | None = None
    _tool_fixture_payload: bytes | None = field(default=None, repr=False)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(item.case.allocation_case_id for item in self.cases)


@dataclass(frozen=True)
class FrozenCampaignInputs:
    """References to exact source copies and the separately recorded selection.

    Paths are relative to the campaign artifact root. This does not seal the
    store: preparation and execution still need to add their own artifacts.
    """

    artifacts: tuple[ArtifactReference, ...]
    projection_manifest: ArtifactReference
    attack_battery: ArtifactReference
    selection: ArtifactReference
    tool_fixture: ArtifactReference | None = None


def load_campaign_inputs(
    options: CampaignPreparationOptions, *, project_root: Path
) -> CampaignInputs:
    """Read/verify one snapshot, validate links, then select complete families.

    Relative source paths resolve against project_root, never the shell CWD.
    None selects all included cases; excluded/unknown IDs fail explicitly.
    No output directories, backend clients, or subprocesses are created.
    """
    options = CampaignPreparationOptions.model_validate(options.model_dump(mode="json"))
    source = (Path(project_root) / options.projection_directory).resolve()
    payloads: dict[str, bytes] = {}
    try:
        if not source.is_dir():
            raise CampaignInputError(f"projection directory does not exist: {source}")
        for name in PROJECTION_FILES:
            path = (source / name).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise CampaignInputError(f"missing or unsafe projection file: {name}")
            payloads[name] = path.read_bytes()
    except OSError as exc:
        raise CampaignInputError(
            f"could not read projection directory: {source}"
        ) from exc
    projection_set = _parse(payloads)
    cases = _select_cases(projection_set, options.case_ids, options.purpose)
    tool_fixture, tool_fixture_payload = _load_tool_fixture(
        options, project_root=Path(project_root)
    )
    _validate_tool_fixture_for_selection(
        tool_fixture,
        cases=cases,
        options=options,
    )
    return CampaignInputs(
        campaign_id=options.campaign_id,
        preparation_sha256=options.sha256,
        purpose=options.purpose,
        source_directory=source,
        manifest=projection_set.manifest,
        cases=cases,
        _payloads=tuple(payloads.items()),
        tool_fixture=tool_fixture,
        _tool_fixture_payload=tool_fixture_payload,
    )


def validate_campaign_inputs(
    inputs: CampaignInputs, *, options: CampaignPreparationOptions | None = None
) -> None:
    """Revalidate the in-memory snapshot without reading or writing files.

    Matrix construction supplies options to require the exact original selection
    and preparation identity. Freezing also uses this check to reject mutations.
    """
    selected = inputs.case_ids
    if options is not None:
        options = CampaignPreparationOptions.model_validate(
            options.model_dump(mode="json")
        )
        if (
            inputs.campaign_id != options.campaign_id
            or inputs.preparation_sha256 != options.sha256
            or inputs.purpose != options.purpose
        ):
            raise CampaignInputError("campaign inputs do not match preparation options")
        selected = options.case_ids
    payloads = dict(inputs._payloads)
    projection_set = _parse(payloads)
    if (
        projection_set.manifest != inputs.manifest
        or _select_cases(projection_set, selected, inputs.purpose) != inputs.cases
    ):
        raise CampaignInputError("campaign inputs no longer match their byte snapshot")
    fixture = _parse_tool_fixture_snapshot(inputs._tool_fixture_payload)
    if fixture != inputs.tool_fixture:
        raise CampaignInputError("tool fixture no longer matches its byte snapshot")
    if options is not None:
        _validate_tool_fixture_for_selection(
            fixture,
            cases=inputs.cases,
            options=options,
        )


def freeze_campaign_inputs(
    inputs: CampaignInputs, store: CampaignArtifactStore
) -> FrozenCampaignInputs:
    """Freeze exact validated bytes, with no overwrite or implicit reapproval.

    All files are evaluator-only, including labels, exclusions, and the source
    manifest. A subset selection does not rewrite/filter the original bundle.
    The exclusive destination prevents reuse; an interrupted freeze is retained
    as partial evidence and must not be treated as a prepared campaign.
    """
    validate_campaign_inputs(inputs)
    payloads = dict(inputs._payloads)
    store.prepare_directory(_DESTINATION)
    references: dict[str, ArtifactReference] = {}
    for name in PROJECTION_FILES:
        references[name] = store.write_bytes(
            artifact_id=f"inputs:projection:{name}",
            kind=(
                ArtifactKind.ATTACK_BATTERY
                if name == "attack_specs.jsonl"
                else ArtifactKind.DATASET_PROJECTION
            ),
            relative_path=f"{_DESTINATION}/{name}",
            content=payloads[name],
            media_type=(
                "application/x-ndjson"
                if name.endswith(".jsonl")
                else "application/json"
            ),
            visibility=ArtifactVisibility.EVALUATOR_ONLY,
        )
    tool_fixture_reference: ArtifactReference | None = None
    if inputs._tool_fixture_payload is not None:
        if inputs.tool_fixture is None:
            raise CampaignInputError("tool fixture bytes are present without a fixture")
        tool_fixture_reference = store.write_bytes(
            artifact_id="inputs:tool-fixture",
            kind=ArtifactKind.TOOL_FIXTURE,
            relative_path=_TOOL_FIXTURE_DESTINATION,
            content=inputs._tool_fixture_payload,
            media_type="application/json",
            visibility=ArtifactVisibility.EVALUATOR_ONLY,
        )
    selection = store.write_json(
        artifact_id="inputs:projection-selection",
        kind=ArtifactKind.OTHER,
        relative_path=f"{_DESTINATION}/selection.json",
        value={
            "schema_version": 1,
            "campaign_id": inputs.campaign_id,
            "preparation_sha256": inputs.preparation_sha256,
            "projection_set_id": inputs.manifest.projection_set_id,
            "projection_manifest": references["manifest.json"],
            "purpose": inputs.purpose,
            "selected_case_ids": inputs.case_ids,
            "selected_trial_ids": [
                trial.trial_id for item in inputs.cases for trial in item.trials
            ],
            "attack_review_status_by_case": {
                item.case.allocation_case_id: item.attack_spec.review_status
                for item in inputs.cases
            },
            "tool_fixture": (
                None
                if tool_fixture_reference is None
                else {
                    "artifact": tool_fixture_reference,
                    "fixture_id": inputs.tool_fixture.fixture_id,
                    "fixture_sha256": inputs.tool_fixture.sha256,
                }
            ),
        },
        visibility=ArtifactVisibility.EVALUATOR_ONLY,
    )
    return FrozenCampaignInputs(
        artifacts=(
            *references.values(),
            *(() if tool_fixture_reference is None else (tool_fixture_reference,)),
            selection,
        ),
        projection_manifest=references["manifest.json"],
        attack_battery=references["attack_specs.jsonl"],
        selection=selection,
        tool_fixture=tool_fixture_reference,
    )


def _load_tool_fixture(
    options: CampaignPreparationOptions, *, project_root: Path
) -> tuple[ToolFixtureBundle | None, bytes | None]:
    """Read one exact optional fixture snapshot without normalizing its bytes."""

    if options.tool_fixture_path is None:
        return None, None
    source = (project_root / options.tool_fixture_path).resolve()
    try:
        if not source.is_file():
            raise CampaignInputError(f"tool fixture does not exist: {source}")
        payload = source.read_bytes()
    except OSError as exc:
        raise CampaignInputError(f"could not read tool fixture: {source}") from exc
    return _parse_tool_fixture_snapshot(payload), payload


def _parse_tool_fixture_snapshot(payload: bytes | None) -> ToolFixtureBundle | None:
    if payload is None:
        return None
    try:
        return parse_tool_fixture(payload)
    except ToolFixtureError as exc:
        raise CampaignInputError(f"invalid tool fixture: {exc}") from exc


def _validate_tool_fixture_for_selection(
    fixture: ToolFixtureBundle | None,
    *,
    cases: tuple[CampaignInputCase, ...],
    options: CampaignPreparationOptions,
) -> None:
    """Require fixture-backed discovery only for the selected scoped personas."""

    required_tools = frozenset(
        tool
        for condition in options.conditions
        if fixture_required(condition.agent_configuration.allowed_tools)
        for tool in condition.agent_configuration.allowed_tools
    )
    if not required_tools:
        return
    if fixture is None:
        # Options normally catches this first.  Keep input loading safe if an
        # already-constructed model bypassed its validation.
        raise CampaignInputError("selected discovery tools require a tool fixture")
    required_personas = {item.case.persona_id for item in cases}
    missing_personas = sorted(required_personas - set(fixture.persona_ids))
    if missing_personas:
        raise CampaignInputError(
            "tool fixture does not declare selected personas: "
            + ", ".join(missing_personas)
        )
    try:
        for condition in options.conditions:
            if fixture_required(condition.agent_configuration.allowed_tools):
                # ToolFixtureBundle's checks cover nonempty contact/document
                # inventories.  Persona membership was checked against the
                # selected cases above; individual run scopes are assigned later.
                discovery = set(condition.agent_configuration.allowed_tools)
                if "lookup_contact" in discovery and not fixture.contacts:
                    raise ToolFixtureError(
                        "lookup_contact requires at least one fixture contact"
                    )
                if (
                    discovery & {"search_documents", "read_document"}
                    and not fixture.documents
                ):
                    raise ToolFixtureError(
                        "document discovery requires at least one fixture document"
                    )
    except ToolFixtureError as exc:
        raise CampaignInputError(
            f"tool fixture cannot support selected tools: {exc}"
        ) from exc


def _parse(payloads: dict[str, bytes]) -> ProjectionSet:
    try:
        projection_set = parse_projection_set(payloads)
        _validate_links(projection_set)
        return projection_set
    except (ProjectionSetError, ValueError) as exc:
        raise CampaignInputError(f"invalid campaign projections: {exc}") from exc


def _select_cases(
    source: ProjectionSet, case_ids: tuple[str, ...] | None, purpose: CampaignPurpose
) -> tuple[CampaignInputCase, ...]:
    available = {case.allocation_case_id for case in source.cases}
    selected = available if case_ids is None else set(case_ids)
    if not selected or (case_ids is not None and len(selected) != len(case_ids)):
        raise CampaignInputError("selection must contain unique, nonempty case IDs")
    excluded = selected & set(source.config.excluded_allocation_case_ids)
    if excluded:
        raise CampaignInputError(f"selected cases were excluded: {sorted(excluded)}")
    if unknown := selected - available:
        raise CampaignInputError(f"unknown selected case IDs: {sorted(unknown)}")
    specs = {item.allocation_case_id: item for item in source.attack_specs}
    probes = {item.allocation_case_id: item for item in source.probes}
    trials = {(item.allocation_case_id, item.twin): item for item in source.trials}
    result = []
    for case in source.cases:
        case_id = case.allocation_case_id
        if case_id not in selected:
            continue
        spec = specs[case_id]
        if (
            purpose == "experiment"
            and spec.review_status != AttackReviewStatus.APPROVED
        ):
            raise CampaignInputError(
                f"unreviewed case requires pilot purpose: {case_id}"
            )
        result.append(
            CampaignInputCase(
                case=case,
                attack_spec=spec,
                probe=probes[case_id],
                trials=tuple(trials[case_id, twin] for twin in TrialTwin),
            )
        )
    return tuple(result)


def _validate_links(source: ProjectionSet) -> None:
    """Reject ambiguous references, mismatched answer keys, or unfair twins.

    Hash validation alone cannot prove these cross-file relationships. These
    are structural checks, not a replacement for semantic/human attack review.
    """
    for items, key in (
        (source.attack_specs, "attack_spec_id"),
        (source.probes, "probe_id"),
        (source.private_labels, "private_label_id"),
        (source.trials, "trial_id"),
    ):
        if len({getattr(item, key) for item in items}) != len(items):
            raise CampaignInputError(f"duplicate projection identity: {key}")
    specs = {item.allocation_case_id: item for item in source.attack_specs}
    probes = {item.allocation_case_id: item for item in source.probes}
    labels = {item.allocation_case_id: item for item in source.private_labels}
    for case in source.cases:
        case_id = case.allocation_case_id
        spec, probe, label = specs[case_id], probes[case_id], labels[case_id]
        if (
            (case.attack_spec_id, case.probe_id, case.private_label_id)
            != (spec.attack_spec_id, probe.probe_id, label.private_label_id)
            or case.attack_id != spec.attack_id
            or case.attack_methodology != spec.attack_methodology
            or case.projection_kind != spec.projection_kind
            or case.insertion_after_source_message_index
            != spec.insert_after_source_message_index
            or (case.candidate_id, case.source_row_index)
            != (label.candidate_id, label.source_row_index)
        ):
            raise CampaignInputError(f"case references do not match: {case_id}")
        answers = (label.correct_answer, *label.incorrect_answers)
        order = probe.original_option_indexes
        if (
            len(answers) != len(order)
            or tuple(answers[index] for index in order) != probe.options
            or order.index(0) != label.correct_display_index
            or order.index(label.selected_false_target_original_index + 1)
            != label.selected_false_target_display_index
        ):
            raise CampaignInputError(f"private labels do not match probe: {case_id}")
        twins = [
            trial for trial in source.trials if trial.allocation_case_id == case_id
        ]
        if {trial.twin for trial in twins} != set(TrialTwin):
            raise CampaignInputError(f"projection twins are incomplete: {case_id}")
        baseline = None
        neutral_signatures = set()
        for trial in twins:
            if (trial.attack_spec_id, trial.probe_id, trial.private_label_id) != (
                case.attack_spec_id,
                case.probe_id,
                case.private_label_id,
            ) or trial.projection_kind != case.projection_kind:
                raise CampaignInputError(
                    f"trial references do not match: {trial.trial_id}"
                )
            for stage, expected in (
                (EventStage.BENIGN_PREFIX, case.pre_attack_source_message_indexes),
                (EventStage.BENIGN_SURVIVAL, case.survival_source_message_indexes),
            ):
                indexes = tuple(
                    event.source_message_index
                    for event in trial.replay_events
                    if event.stage == stage
                )
                if indexes != expected:
                    raise CampaignInputError(
                        f"replay source indexes differ: {trial.trial_id}"
                    )
            if case.projection_kind == ProjectionKind.CONVERSATION_ATTACK:
                benign = tuple(
                    (
                        event.stage,
                        event.role,
                        event.content,
                        event.source_message_index,
                    )
                    for event in trial.replay_events
                    if event.source_message_index is not None
                )
                if baseline is not None and baseline != benign:
                    raise CampaignInputError(
                        f"benign replay differs across twins: {case_id}"
                    )
                baseline = benign
            overlay = tuple(
                (event.role.value, event.content)
                for event in trial.replay_events
                if event.source_message_index is None
            )
            if case.projection_kind == ProjectionKind.CONVERSATION_ATTACK:
                expected_overlay = {
                    TrialTwin.CLEAN: (),
                    TrialTwin.MATCHED_BENIGN: spec.matched_benign_overlay,
                    TrialTwin.POISONED: spec.poisoned_overlay,
                }[trial.twin]
                if overlay != tuple(
                    (item.role, item.content) for item in expected_overlay
                ):
                    raise CampaignInputError(
                        f"replay overlay differs from spec: {trial.trial_id}"
                    )
            else:
                if overlay or any(
                    event.stage
                    not in {EventStage.BENIGN_PREFIX, EventStage.BENIGN_SURVIVAL}
                    for event in trial.replay_events
                ):
                    raise CampaignInputError(
                        "neutral external host contains an overlay: "
                        f"{trial.trial_id}"
                    )
                neutral_signatures.add(memory_visible_replay_signature(trial))
        if (
            case.projection_kind == ProjectionKind.NEUTRAL_EXTERNAL_HOST
            and len(neutral_signatures) != 1
        ):
            raise CampaignInputError(
                "neutral external host twins must have identical "
                f"memory-visible replay: {case_id}"
            )

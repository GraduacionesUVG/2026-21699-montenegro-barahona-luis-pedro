"""Versioned run-to-projection bindings for a frozen campaign input artifact.

These contracts do not load projections or run a trial. Preparation will write a
CampaignRunBindings document using ArtifactKind.RUN_BINDINGS; the future child
must verify that artifact and its source references before resolving a binding.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
    CampaignConfiguration,
    CampaignRunSpec,
    TrialTwin,
)
from membench.campaign.preparation_models import (
    AllocationId,
    CampaignPreparationOptions,
    CampaignPurpose,
    PreparationId,
    Sha256,
    execution_condition_sha256,
    model_profile_sha256,
)
from membench.curation.projections import AttackReviewStatus
from membench.memory.models import CanonicalModel
from membench.qualification.models import QuiescenceConfiguration
from membench.trial.models import ReplayConfiguration, TrialPlan

if TYPE_CHECKING:
    from membench.ingress.plans import TrialIngressPlan, TrialIngressPlanSet


class RunProjectionBinding(CanonicalModel):
    """The exact projection and execution choices for one isolated campaign run.

    Trial/probe IDs are references, not answer keys. Hashes bind the full run and
    reconstructed TrialPlan, including ordered replay events and public options.
    Model profile labels are for display; use the hash as the model identity.
    """

    schema_version: Literal[1] = 1
    run_id: PreparationId
    condition_id: PreparationId
    condition_sha256: Sha256
    model_profile_id: PreparationId
    model_profile_sha256: Sha256
    run_spec_sha256: Sha256
    trial_plan_sha256: Sha256
    trial_id: PreparationId
    allocation_case_id: AllocationId
    probe_id: PreparationId
    twin: TrialTwin
    replay: ReplayConfiguration
    quiescence: QuiescenceConfiguration
    purpose: CampaignPurpose
    attack_review_status: AttackReviewStatus
    cleanup_storage: bool
    ingress_plan_set_id: PreparationId | None = None
    ingress_plan_id: PreparationId | None = None
    ingress_plan_sha256: Sha256 | None = None
    external_attack_spec_id: PreparationId | None = None
    external_attack_spec_sha256: Sha256 | None = None
    external_delivery_family_id: PreparationId | None = None
    external_ingress_mode: str | None = None
    evaluation_probe_id: PreparationId | None = None
    evaluation_probe_sha256: Sha256 | None = None
    behavioral_task_id: PreparationId | None = None
    behavioral_task_sha256: Sha256 | None = None
    behavioral_oracle_id: PreparationId | None = None
    behavioral_oracle_sha256: Sha256 | None = None
    trusted_baseline_id: PreparationId | None = None
    trusted_baseline_sha256: Sha256 | None = None

    @property
    def has_ingress_plan(self) -> bool:
        """Whether this run has one compact frozen external-ingress reference."""

        return self.ingress_plan_id is not None

    @property
    def has_evaluation_probe(self) -> bool:
        """Whether this run has an immutable evaluator-only probe binding."""

        return self.evaluation_probe_id is not None

    @property
    def has_behavioral_task(self) -> bool:
        """Whether this run is bound to a task and its evaluator-only oracle."""

        return self.behavioral_task_id is not None

    @property
    def has_trusted_baseline(self) -> bool:
        """Whether this run receives a frozen trusted pre-ingress seed."""

        return self.trusted_baseline_id is not None

    @model_validator(mode="after")
    def ingress_reference_must_be_complete(self) -> Self:
        reference = (
            self.ingress_plan_set_id,
            self.ingress_plan_id,
            self.ingress_plan_sha256,
        )
        if any(value is None for value in reference) and any(
            value is not None for value in reference
        ):
            raise ValueError(
                "ingress plan set ID, plan ID, and SHA-256 must be present together"
            )
        external = (
            self.external_attack_spec_id,
            self.external_attack_spec_sha256,
            self.external_delivery_family_id,
            self.external_ingress_mode,
        )
        if any(value is None for value in external) and any(
            value is not None for value in external
        ):
            raise ValueError("external attack identity fields must be present together")
        if not any(value is None for value in reference) and any(
            value is None for value in external
        ):
            raise ValueError(
                "external ingress references require a complete external "
                "attack identity"
            )
        if all(value is None for value in reference) and any(
            value is not None for value in external
        ):
            raise ValueError(
                "conversation-only bindings cannot carry an external attack identity"
            )
        return self

    @model_validator(mode="after")
    def evaluation_reference_must_be_complete(self) -> Self:
        reference = (self.evaluation_probe_id, self.evaluation_probe_sha256)
        if any(value is None for value in reference) and any(
            value is not None for value in reference
        ):
            raise ValueError(
                "evaluation probe ID and SHA-256 must be present together"
            )
        return self

    @model_validator(mode="after")
    def behavioral_reference_must_be_complete(self) -> Self:
        reference = (
            self.behavioral_task_id,
            self.behavioral_task_sha256,
            self.behavioral_oracle_id,
            self.behavioral_oracle_sha256,
        )
        if any(value is None for value in reference) and any(
            value is not None for value in reference
        ):
            raise ValueError(
                "behavioral task and oracle IDs and SHA-256 values must be "
                "present together"
            )
        return self

    @model_validator(mode="after")
    def trusted_baseline_reference_must_be_complete(self) -> Self:
        reference = (self.trusted_baseline_id, self.trusted_baseline_sha256)
        if any(value is None for value in reference) and any(
            value is not None for value in reference
        ):
            raise ValueError(
                "trusted baseline ID and SHA-256 must be present together"
            )
        return self

    @model_validator(mode="after")
    def experiment_requires_review(self) -> Self:
        if self.purpose == "experiment" and (
            self.attack_review_status != AttackReviewStatus.APPROVED
        ):
            raise ValueError("unreviewed bindings must retain pilot status")
        return self

    @classmethod
    def from_trial_plan(
        cls, plan: TrialPlan, *, condition_id: str, model_profile_id: str
    ) -> RunProjectionBinding:
        """Capture a validated plan without sharing mutable nested input values."""
        plan = TrialPlan.model_validate(plan.model_dump(mode="json"))
        return cls(
            run_id=plan.run.run_id,
            condition_id=condition_id,
            condition_sha256=execution_condition_sha256(
                plan.run.manifest, plan.run.agent_configuration, plan.replay
            ),
            model_profile_id=model_profile_id,
            model_profile_sha256=model_profile_sha256(plan.run.manifest.models),
            run_spec_sha256=hashlib.sha256(canonical_json_bytes(plan.run)).hexdigest(),
            trial_plan_sha256=plan.sha256,
            trial_id=plan.projection.trial_id,
            allocation_case_id=plan.projection.allocation_case_id,
            probe_id=plan.probe.probe_id,
            twin=plan.run.twin,
            replay=plan.replay,
            quiescence=plan.quiescence,
            purpose=plan.purpose,
            attack_review_status=plan.attack_review_status,
            cleanup_storage=plan.cleanup_storage,
        )

    def validate_trial_plan(self, plan: TrialPlan) -> None:
        """Reject a different run, projection, probe, or execution configuration."""
        observed = self.from_trial_plan(
            plan,
            condition_id=self.condition_id,
            model_profile_id=self.model_profile_id,
        )
        if self.has_ingress_plan:
            observed = observed.model_copy(
                update={
                    "ingress_plan_set_id": self.ingress_plan_set_id,
                    "ingress_plan_id": self.ingress_plan_id,
                    "ingress_plan_sha256": self.ingress_plan_sha256,
                    "external_attack_spec_id": self.external_attack_spec_id,
                    "external_attack_spec_sha256": self.external_attack_spec_sha256,
                    "external_delivery_family_id": self.external_delivery_family_id,
                    "external_ingress_mode": self.external_ingress_mode,
                }
            )
        if self.has_evaluation_probe:
            observed = observed.model_copy(
                update={
                    "evaluation_probe_id": self.evaluation_probe_id,
                    "evaluation_probe_sha256": self.evaluation_probe_sha256,
                }
            )
        if self.has_behavioral_task:
            observed = observed.model_copy(
                update={
                    "behavioral_task_id": self.behavioral_task_id,
                    "behavioral_task_sha256": self.behavioral_task_sha256,
                    "behavioral_oracle_id": self.behavioral_oracle_id,
                    "behavioral_oracle_sha256": self.behavioral_oracle_sha256,
                }
            )
        if self.has_trusted_baseline:
            observed = observed.model_copy(
                update={
                    "trusted_baseline_id": self.trusted_baseline_id,
                    "trusted_baseline_sha256": self.trusted_baseline_sha256,
                }
            )
        if observed != self:
            raise ValueError("trial plan does not match its frozen run binding")

    def with_ingress_plan(
        self,
        plan: TrialIngressPlan,
        *,
        ingress_plan_set_id: str,
    ) -> RunProjectionBinding:
        """Return this binding linked to one already-frozen ingress plan.

        The host trial and twin checks here catch a bad association early.  The
        campaign-level validator additionally checks the plan against its run,
        its frozen manifest, and the other two members of the twin family.
        """

        if plan.host_trial_id != self.trial_id:
            raise ValueError("ingress plan host trial does not match binding trial")
        if plan.twin != self.twin:
            raise ValueError("ingress plan twin does not match binding twin")
        return RunProjectionBinding.model_validate(
            {
                **self.model_dump(mode="json"),
                "ingress_plan_set_id": ingress_plan_set_id,
                "ingress_plan_id": plan.ingress_plan_id,
                "ingress_plan_sha256": plan.sha256,
                "external_attack_spec_id": plan.attack_spec_id,
                "external_attack_spec_sha256": plan.attack_spec_sha256,
                "external_delivery_family_id": plan.delivery_family_id,
                "external_ingress_mode": plan.ingress_mode.value,
            }
        )

    def with_evaluation_probe(
        self,
        *,
        probe_id: str,
        probe_sha256: str,
    ) -> RunProjectionBinding:
        """Bind this run to one already-frozen evaluator probe."""

        return RunProjectionBinding.model_validate(
            {
                **self.model_dump(mode="json"),
                "evaluation_probe_id": probe_id,
                "evaluation_probe_sha256": probe_sha256,
            }
        )

    def with_behavioral_task(
        self,
        *,
        task_id: str,
        task_sha256: str,
        oracle_id: str,
        oracle_sha256: str,
    ) -> RunProjectionBinding:
        """Bind the public behavioral task and private oracle to this run."""

        return RunProjectionBinding.model_validate(
            {
                **self.model_dump(mode="json"),
                "behavioral_task_id": task_id,
                "behavioral_task_sha256": task_sha256,
                "behavioral_oracle_id": oracle_id,
                "behavioral_oracle_sha256": oracle_sha256,
            }
        )

    def with_trusted_baseline(
        self, seed, *, persona_id: str
    ) -> RunProjectionBinding:
        """Bind this run to one seed selected by its PersonaMem persona."""

        if seed.persona_id != int(persona_id):
            raise ValueError("trusted baseline persona does not match binding run")
        return RunProjectionBinding.model_validate(
            {
                **self.model_dump(mode="json"),
                "trusted_baseline_id": seed.baseline_id,
                "trusted_baseline_sha256": hashlib.sha256(
                    canonical_json_bytes(seed)
                ).hexdigest(),
            }
        )


class CampaignRunBindings(CanonicalModel):
    """Complete binding inventory, linked to a frozen projection-set manifest.

    Paths in projection_manifest are relative to the campaign artifact root, as
    with other ArtifactReferences. Structural validation does not verify bytes;
    that remains the responsibility of inputs.py and the child entrypoint.
    """

    schema_version: Literal[1] = 1
    campaign_id: PreparationId
    preparation_sha256: Sha256
    projection_set_id: PreparationId
    projection_manifest: ArtifactReference
    ingress_plan_manifest: ArtifactReference | None = None
    evaluation_probe: ArtifactReference | None = None
    behavioral_task: ArtifactReference | None = None
    behavioral_oracle: ArtifactReference | None = None
    trusted_baseline_manifest: ArtifactReference | None = None
    bindings: tuple[RunProjectionBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_must_be_unambiguous(self) -> Self:
        reference = self.projection_manifest
        if (
            reference.kind != ArtifactKind.DATASET_PROJECTION
            or reference.visibility != ArtifactVisibility.EVALUATOR_ONLY
            or reference.media_type != "application/json"
        ):
            raise ValueError(
                "projection manifest must be an evaluator-only JSON projection"
            )
        ingress_bindings = [
            binding for binding in self.bindings if binding.has_ingress_plan
        ]
        if ingress_bindings:
            if self.ingress_plan_manifest is None:
                raise ValueError(
                    "external campaign bindings require an ingress-plan manifest"
                )
            ingress_manifest = self.ingress_plan_manifest
            if (
                ingress_manifest.kind != ArtifactKind.TRIAL_INGRESS_PLAN
                or ingress_manifest.visibility != ArtifactVisibility.EVALUATOR_ONLY
                or ingress_manifest.media_type != "application/json"
            ):
                raise ValueError(
                    "ingress-plan manifest must be an evaluator-only JSON "
                    "trial ingress plan"
                )
            set_ids = {binding.ingress_plan_set_id for binding in ingress_bindings}
            if len(set_ids) != 1:
                raise ValueError(
                    "external campaign bindings must use one ingress-plan set"
                )
        elif self.ingress_plan_manifest is not None:
            raise ValueError(
                "conversation-only campaign bindings cannot carry an "
                "ingress-plan manifest"
            )
        evaluation_bindings = [
            binding for binding in self.bindings if binding.has_evaluation_probe
        ]
        if evaluation_bindings:
            if len(evaluation_bindings) != len(self.bindings):
                raise ValueError(
                    "campaign cannot mix runs with and without evaluation probes"
                )
            if self.evaluation_probe is None:
                raise ValueError(
                    "evaluation bindings require an evaluation-probe artifact"
                )
            evaluation_probe = self.evaluation_probe
            if (
                evaluation_probe.kind != ArtifactKind.EVALUATION_PROBE
                or evaluation_probe.visibility != ArtifactVisibility.EVALUATOR_ONLY
                or evaluation_probe.media_type != "application/json"
            ):
                raise ValueError(
                    "evaluation probe must be evaluator-only JSON evidence"
                )
            probe_ids = {binding.evaluation_probe_id for binding in evaluation_bindings}
            probe_hashes = {
                binding.evaluation_probe_sha256 for binding in evaluation_bindings
            }
            if len(probe_ids) != 1 or probe_hashes != {evaluation_probe.sha256}:
                raise ValueError(
                    "evaluation bindings do not match the frozen evaluation probe"
                )
        elif self.evaluation_probe is not None:
            raise ValueError(
                "campaign without evaluation bindings cannot carry an evaluation probe"
            )
        behavioral_bindings = [
            binding for binding in self.bindings if binding.has_behavioral_task
        ]
        if behavioral_bindings:
            if len(behavioral_bindings) != len(self.bindings):
                raise ValueError(
                    "campaign cannot mix runs with and without behavioral tasks"
                )
            if self.behavioral_task is None or self.behavioral_oracle is None:
                raise ValueError(
                    "behavioral bindings require task and oracle artifacts"
                )
            task, oracle = self.behavioral_task, self.behavioral_oracle
            if (
                task.kind != ArtifactKind.BEHAVIORAL_TASK
                or task.visibility != ArtifactVisibility.EVALUATOR_ONLY
                or task.media_type != "application/json"
                or oracle.kind != ArtifactKind.BEHAVIORAL_ORACLE
                or oracle.visibility != ArtifactVisibility.EVALUATOR_ONLY
                or oracle.media_type != "application/json"
            ):
                raise ValueError(
                    "behavioral task and oracle must be evaluator-only JSON evidence"
                )
            if (
                {binding.behavioral_task_sha256 for binding in behavioral_bindings}
                != {task.sha256}
                or {binding.behavioral_oracle_sha256 for binding in behavioral_bindings}
                != {oracle.sha256}
                or len(
                    {binding.behavioral_task_id for binding in behavioral_bindings}
                )
                != 1
                or len(
                    {binding.behavioral_oracle_id for binding in behavioral_bindings}
                )
                != 1
            ):
                raise ValueError(
                    "behavioral bindings do not match frozen task artifacts"
                )
        elif self.behavioral_task is not None or self.behavioral_oracle is not None:
            raise ValueError(
                "campaign without behavioral bindings cannot carry behavioral artifacts"
            )
        baseline_bindings = [
            binding for binding in self.bindings if binding.has_trusted_baseline
        ]
        if baseline_bindings:
            if len(baseline_bindings) != len(self.bindings):
                raise ValueError(
                    "campaign cannot mix runs with and without trusted baselines"
                )
            manifest = self.trusted_baseline_manifest
            if (
                manifest is None
                or manifest.kind != ArtifactKind.TRUSTED_BASELINE
                or manifest.visibility != ArtifactVisibility.EVALUATOR_ONLY
                or manifest.media_type != "application/json"
            ):
                raise ValueError(
                    "trusted baseline manifest must be evaluator-only JSON evidence"
                )
        elif self.trusted_baseline_manifest is not None:
            raise ValueError(
                "campaign without trusted-baseline bindings cannot carry a manifest"
            )
        ids = [binding.run_id for binding in self.bindings]
        if len(ids) != len(set(ids)):
            raise ValueError("run binding IDs must be unique")
        profiles: dict[str, str] = {}
        for binding in self.bindings:
            previous = profiles.setdefault(
                binding.model_profile_id, binding.model_profile_sha256
            )
            if previous != binding.model_profile_sha256:
                raise ValueError(
                    "one model_profile_id cannot describe different models"
                )
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self)).hexdigest()

    def binding_for(self, run_id: str) -> RunProjectionBinding:
        """Return a detached binding; never guess IDs from filenames or twins."""
        for binding in self.bindings:
            if binding.run_id == run_id:
                return binding.model_copy(deep=True)
        raise KeyError(f"no projection binding for run {run_id!r}")

    def validate_campaign(
        self,
        configuration: CampaignConfiguration,
        *,
        ingress_plan_set: TrialIngressPlanSet | None = None,
        trusted_baselines=None,
    ) -> None:
        """Require exact run coverage and consistent bindings for each twin family."""
        configuration = CampaignConfiguration.model_validate(
            configuration.model_dump(mode="json")
        )
        if configuration.campaign_id != self.campaign_id:
            raise ValueError("bindings belong to a different campaign")
        if self.projection_manifest not in configuration.frozen_inputs:
            raise ValueError("projection manifest is not a frozen campaign input")
        runs = {run.run_id: run for run in configuration.runs}
        has_evaluation = any(
            binding.has_evaluation_probe for binding in self.bindings
        )
        if has_evaluation:
            if self.evaluation_probe not in configuration.frozen_inputs:
                raise ValueError("evaluation probe is not a frozen campaign input")
        has_behavioral = any(binding.has_behavioral_task for binding in self.bindings)
        if has_behavioral:
            if (
                self.behavioral_task not in configuration.frozen_inputs
                or self.behavioral_oracle not in configuration.frozen_inputs
            ):
                raise ValueError(
                    "behavioral task artifacts are not frozen campaign inputs"
                )
        has_baseline = any(binding.has_trusted_baseline for binding in self.bindings)
        if has_baseline:
            if self.trusted_baseline_manifest not in configuration.frozen_inputs:
                raise ValueError(
                    "trusted baseline manifest is not a frozen campaign input"
                )
            if trusted_baselines is not None:
                self._validate_trusted_baselines(
                    runs=runs, manifest=trusted_baselines
                )
        has_ingress = any(binding.has_ingress_plan for binding in self.bindings)
        if has_ingress:
            if self.ingress_plan_manifest not in configuration.frozen_inputs:
                raise ValueError("ingress-plan manifest is not a frozen campaign input")
            if ingress_plan_set is None:
                raise ValueError(
                    "external campaign validation requires loaded ingress plans"
                )
            self._validate_ingress_plan_set(ingress_plan_set)
        elif ingress_plan_set is not None:
            raise ValueError(
                "conversation-only campaign cannot be validated with ingress plans"
            )
        bound = {binding.run_id: binding for binding in self.bindings}
        if runs.keys() != bound.keys():
            raise ValueError("bindings must cover every campaign run exactly once")
        families: dict[tuple[str, int], tuple] = {}
        for run_id, run in runs.items():
            binding = bound[run_id]
            if (
                binding.twin != run.twin
                or binding.run_spec_sha256
                != hashlib.sha256(canonical_json_bytes(run)).hexdigest()
                or binding.model_profile_sha256
                != model_profile_sha256(run.manifest.models)
                or binding.condition_sha256
                != execution_condition_sha256(
                    run.manifest, run.agent_configuration, binding.replay
                )
            ):
                raise ValueError(f"binding does not match campaign run {run_id!r}")
            comparable = (
                binding.condition_id,
                binding.condition_sha256,
                binding.model_profile_id,
                binding.model_profile_sha256,
                binding.allocation_case_id,
                binding.probe_id,
                binding.replay,
                binding.quiescence,
                binding.purpose,
                binding.attack_review_status,
                binding.cleanup_storage,
                binding.evaluation_probe_id,
                binding.evaluation_probe_sha256,
                binding.behavioral_task_id,
                binding.behavioral_task_sha256,
                binding.behavioral_oracle_id,
                binding.behavioral_oracle_sha256,
                binding.trusted_baseline_id,
                binding.trusted_baseline_sha256,
            )
            key = (run.trial_family_id, run.response_replicate)
            previous = families.setdefault(key, comparable)
            if comparable != previous:
                raise ValueError("twin family has inconsistent projection bindings")
        if has_ingress:
            self._validate_ingress_bindings(runs, ingress_plan_set)

    def _validate_trusted_baselines(self, *, runs: dict, manifest) -> None:
        """Check frozen seed references and twin fairness against loaded bytes."""

        by_id = {seed.baseline_id: seed for seed in manifest.baselines}
        families: dict[tuple[str, int], tuple[str, str]] = {}
        for binding in self.bindings:
            run = runs[binding.run_id]
            assert binding.trusted_baseline_id is not None
            assert binding.trusted_baseline_sha256 is not None
            seed = by_id.get(binding.trusted_baseline_id)
            if (
                seed is None
                or hashlib.sha256(canonical_json_bytes(seed)).hexdigest()
                != binding.trusted_baseline_sha256
                or str(seed.persona_id) != run.persona_id
            ):
                raise ValueError("trusted baseline binding does not match frozen seed")
            key = (run.trial_family_id, run.response_replicate)
            comparable = (
                seed.baseline_id,
                hashlib.sha256(canonical_json_bytes(seed)).hexdigest(),
            )
            if families.setdefault(key, comparable) != comparable:
                raise ValueError("twin family has inconsistent trusted baselines")

    def _validate_ingress_plan_set(self, plan_set: TrialIngressPlanSet) -> None:
        """Verify the referenced manifest can describe the supplied plan set."""

        assert self.ingress_plan_manifest is not None
        manifest_bytes = canonical_json_bytes(
            plan_set.manifest().model_dump(mode="json")
        )
        if (
            self.ingress_plan_manifest.sha256
            != hashlib.sha256(manifest_bytes).hexdigest()
        ):
            raise ValueError("ingress-plan manifest hash does not match loaded plans")
        if self.ingress_plan_manifest.size_bytes != len(manifest_bytes):
            raise ValueError("ingress-plan manifest size does not match loaded plans")
        set_ids = {
            binding.ingress_plan_set_id
            for binding in self.bindings
            if binding.has_ingress_plan
        }
        if set_ids != {plan_set.ingress_plan_set_id}:
            raise ValueError("binding ingress-plan set does not match loaded plans")

    def _validate_ingress_bindings(
        self,
        runs: dict[str, CampaignRunSpec],
        plan_set: TrialIngressPlanSet | None,
    ) -> None:
        """Bind each run to its exact host plan and preserve twin fairness."""

        assert plan_set is not None
        families: dict[tuple[str, int], list[TrialIngressPlan | None]] = {}
        for binding in self.bindings:
            run = runs[binding.run_id]
            if not binding.has_ingress_plan:
                families.setdefault(
                    (run.trial_family_id, run.response_replicate), []
                ).append(None)
                continue
            try:
                plan = plan_set.plan_for_host_trial(binding.trial_id)
            except KeyError as exc:
                raise ValueError(
                    "binding trial has no ingress plan with the same host trial ID"
                ) from exc
            if (
                binding.ingress_plan_set_id != plan_set.ingress_plan_set_id
                or binding.ingress_plan_id != plan.ingress_plan_id
                or binding.ingress_plan_sha256 != plan.sha256
                or binding.external_attack_spec_id != plan.attack_spec_id
                or binding.external_attack_spec_sha256 != plan.attack_spec_sha256
                or binding.external_delivery_family_id != plan.delivery_family_id
                or binding.external_ingress_mode != plan.ingress_mode.value
            ):
                raise ValueError(
                    "binding ingress-plan reference does not match loaded plan"
                )
            if plan.host_trial_id != binding.trial_id:
                raise ValueError("binding trial does not match ingress plan host trial")
            if plan.persona_id != run.persona_id or plan.twin != run.twin:
                raise ValueError(
                    "ingress plan persona or twin does not match campaign run"
                )
            families.setdefault(
                (run.trial_family_id, run.response_replicate), []
            ).append(plan)

        for plans in families.values():
            if any(plan is None for plan in plans):
                if any(plan is not None for plan in plans):
                    raise ValueError(
                        "twin family cannot mix external and conversation ingress"
                    )
                continue
            concrete = [plan for plan in plans if plan is not None]
            if len({plan.ingress_mode for plan in concrete}) != 1:
                raise ValueError("ingress twin family must use one ingress mode")
            if len({plan.delivery_family_id for plan in concrete}) != 1:
                raise ValueError("ingress twin family must use one delivery family")
            if len({plan.attack_spec_id for plan in concrete}) != 1:
                raise ValueError(
                    "ingress twin family must use one external attack specification"
                )
            if len({plan.delivery_id for plan in concrete}) != len(concrete):
                raise ValueError(
                    "ingress twin deliveries must have distinct delivery IDs"
                )
            if len({plan.delivery_content_sha256 for plan in concrete}) != len(
                concrete
            ):
                raise ValueError(
                    "ingress twin deliveries must have distinct delivery-content hashes"
                )

    def validate_preparation(self, options: CampaignPreparationOptions) -> None:
        """Check frozen options and condition references without building a matrix."""
        options = CampaignPreparationOptions.model_validate(
            options.model_dump(mode="json")
        )
        if (
            options.campaign_id != self.campaign_id
            or options.sha256 != self.preparation_sha256
        ):
            raise ValueError("bindings do not match the frozen preparation options")
        conditions = {
            condition.condition_id: condition for condition in options.conditions
        }
        for binding in self.bindings:
            condition = conditions.get(binding.condition_id)
            if condition is None:
                raise ValueError("binding references an unselected condition")
            if (
                binding.model_profile_id != condition.model_profile_id
                or binding.condition_sha256 != condition.sha256
                or binding.model_profile_sha256 != condition.model_profile_sha256
                or binding.replay != condition.replay
                or binding.quiescence != options.quiescence
                or binding.purpose != options.purpose
                or binding.cleanup_storage != options.cleanup_storage
            ):
                raise ValueError("binding execution settings differ from preparation")
            if (options.evaluation_probe_path is None) == binding.has_evaluation_probe:
                raise ValueError(
                    "binding evaluation-probe presence differs from preparation"
                )
            if (options.behavioral_task_path is None) == binding.has_behavioral_task:
                raise ValueError(
                    "binding behavioral-task presence differs from preparation"
                )
            if (options.trusted_baseline_path is None) == binding.has_trusted_baseline:
                raise ValueError(
                    "binding trusted-baseline presence differs from preparation"
                )
            if (
                options.case_ids is not None
                and binding.allocation_case_id not in options.case_ids
            ):
                raise ValueError("binding references an unselected allocation")

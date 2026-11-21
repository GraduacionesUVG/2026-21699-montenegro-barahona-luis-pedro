# Synthetic qualification data

This package contains non-sensitive synthetic records used only to prove the
common adapter lifecycle. It is independent from PersonaMem-v2 and the attack
battery, so qualification outcomes cannot leak benchmark labels or become part
of measured histories.

`manifest.json` defines reusable qualification profiles and pins every canary
document by SHA-256. Existing versioned files are immutable: change behavior by
adding a new canary ID or suite version rather than editing evidence used by a
previous run.

The baseline profile is applicable to direct and native-selective policies. The
shared-guard profile includes both a benign candidate and a clearly
out-of-contract injection candidate. Memory-family coverage is optional and
must pass capability validation before execution.

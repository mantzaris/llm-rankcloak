# Synthetic fixtures

No file in this directory is human data. `synthetic_fixture_spec.json` drives a
candidate generator that creates clearly labeled placeholder prose for testing the
selection, blinding, scheduling, power, and analysis pipelines.

The fixture grid contains two candidates for every condition x prompt-category x
template x eligible-payload-class stratum. Selection takes one per stratum, yielding
exactly 72 messages per condition. Synthetic ratings carry
`synthetic_fixture=true`.

Fixture outputs must not be copied into a study platform or presented as evidence.

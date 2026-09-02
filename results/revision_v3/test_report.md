# Test report

Status: v3_pass_with_immutable_paperv2_contract_failures.

Pytest cases recorded: 744; failures: 3; errors: 0; skipped: 0.

- logs/pytest_full.xml: 695 tests, 3 failures, 0 errors, SHA-256 111fe4ce8ca5796a2bac20b7e5ad2199b4c93657c93ef56f65f541f32733f02b
- logs/pytest_v3_focused.xml: 49 tests, 0 failures, 0 errors, SHA-256 86074d97d75e98142ef82ddf78b5fec30ff5801defc3fa4dbdc1c0acd007f29a

The complete 695-test run passed 692 tests and failed only the following 3 committed paperV2 reference-contract assertions. The paperV2 tree was not modified because manuscript files are immutable in this computational session:

- tests.test_revision_references::test_patient_huffman_is_present_and_tangential_suggestion_is_not_forced
- tests.test_revision_references::test_revised_set_matches_the_completed_v2_bibliography_delta
- tests.test_revision_references::test_staged_entries_have_required_fields_and_unique_dois

All 49 focused V3 tests passed. The failed Pkg.test() probe in logs/julia_pkg_test.log is inapplicable: this repository contains no Julia project/package or Julia test sources, and Julia exited before executing tests.

# Test report

Status: v3_pass_with_immutable_paperv2_contract_failures.

Pytest cases recorded: 758; failures: 3; errors: 0; skipped: 0.

- logs/pytest_full.xml: 702 tests, 3 failures, 0 errors, SHA-256 832f986255075c20c6768a176cee0e3a7ac82350d74ad10a99a76b4f59fd3b11
- logs/pytest_v3_focused.xml: 56 tests, 0 failures, 0 errors, SHA-256 2e6d8de9e3dcd8bb525a0def78667c071c4386a10896a393d5ed4b642d86da21

The complete 702-test run passed 699 tests and failed only the following 3 committed paperV2 reference-contract assertions. The paperV2 tree was not modified because manuscript files are immutable in this computational session:

- tests.test_revision_references::test_patient_huffman_is_present_and_tangential_suggestion_is_not_forced
- tests.test_revision_references::test_revised_set_matches_the_completed_v2_bibliography_delta
- tests.test_revision_references::test_staged_entries_have_required_fields_and_unique_dois

All 56 focused V3 tests passed. The failed Pkg.test() probe in logs/julia_pkg_test.log is inapplicable: this repository contains no Julia project/package or Julia test sources, and Julia exited before executing tests.

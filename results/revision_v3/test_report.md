# Test report

Status: v3_pass_with_immutable_paperv2_contract_failures.

Pytest cases recorded: 700; failures: 3; errors: 0; skipped: 0.

- logs/pytest_full.xml: 673 tests, 3 failures, 0 errors, SHA-256 b6d397dbae06bf394199584087f9c542837c72837d46fd23ed68d936267f4889
- logs/pytest_v3_focused.xml: 27 tests, 0 failures, 0 errors, SHA-256 9fc7f628c734c421828ac8bbe5713d7063e63d4f9682c3377b5f9bc7be2b6cb3

The complete 673-test run passed 670 tests and failed only the following three committed paperV2 reference-contract assertions. The paperV2 tree was not modified because manuscript files are immutable in this computational session:

- tests.test_revision_references::test_patient_huffman_is_present_and_tangential_suggestion_is_not_forced
- tests.test_revision_references::test_revised_set_matches_the_completed_v2_bibliography_delta
- tests.test_revision_references::test_staged_entries_have_required_fields_and_unique_dois

All 27 focused V3 tests passed. The failed Pkg.test() probe in logs/julia_pkg_test.log is inapplicable: this repository contains no Julia project/package or Julia test sources, and Julia exited before executing tests.

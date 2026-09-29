if(Test-Path -LiteralPath 'docs/reorganization/moves.json') {throw 'Reorganization already recorded; do not rerun'}
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path '.').Path
$moves = [ordered]@{
 'code_by_me' = 'archive/legacy_code/code_by_me'
 'review_support' = 'tools/review_support'
 'inspect_pdf_runtime.R' = 'tools/review_support/inspect_pdf_runtime.R'
 'literature_audit.md' = 'archive/reviews/literature_audit.md'
 'methodology_review_notes.md' = 'archive/reviews/methodology_review_notes.md'
 'Acoustic_Mosquito_Edge_AI_Proposal_Revised.pdf' = 'docs/proposals/Acoustic_Mosquito_Edge_AI_Proposal_Revised.pdf'
 'tinyml_primary_source.pdf' = 'reference/tinyml/tinyml_primary_source.pdf'
 'data/neurips_2021_zenodo_0_0_1.csv' = 'data/metadata/neurips_2021_zenodo_0_0_1.csv'
 'data/core_single_4species_metadata.csv' = 'data/manifests/current/core_single_4species_metadata.csv'
 'data/core_single_4species_frozen_split.csv' = 'data/manifests/current/core_single_4species_frozen_split.csv'
 'data/core_example_grid_v2_stride15360_ref15600_candidates.csv' = 'data/manifests/current/core_example_grid_v2_stride15360_ref15600_candidates.csv'
 'data/core_audio_audit.csv' = 'data/audits/current/core_audio_audit.csv'
 'data/core_example_grid_v2_stride15360_ref15600_audit.json' = 'data/audits/current/core_example_grid_v2_stride15360_ref15600_audit.json'
 'data/_reproduced_00' = 'reports/reproduced_00'
 'data/core_096s_candidate_windows.csv' = 'archive/legacy_data/core_096s_candidate_windows.csv'
 'data/core_1s_audit_windows.csv' = 'archive/legacy_data/core_1s_audit_windows.csv'
 'data/train_joint_preprocessing_summary.csv' = 'archive/legacy_data/train_joint_preprocessing_summary.csv'
 'data/train_joint_preprocessing_sweep.csv' = 'archive/legacy_data/train_joint_preprocessing_sweep.csv'
 'data/train_low_frequency_audit.csv' = 'archive/legacy_data/train_low_frequency_audit.csv'
 'data/train_mfe_audit_summary.csv' = 'archive/legacy_data/train_mfe_audit_summary.csv'
 'data/train_signal_floor_diagnostic.csv' = 'archive/legacy_data/train_signal_floor_diagnostic.csv'
}
$records = @()
foreach($entry in $moves.GetEnumerator()) {
 $src = [IO.Path]::GetFullPath((Join-Path $root $entry.Key))
 $dst = [IO.Path]::GetFullPath((Join-Path $root $entry.Value))
 if(!$src.StartsWith($root + '\') -or !$dst.StartsWith($root + '\')) {throw 'Outside workspace'}
 if(!(Test-Path -LiteralPath $src) -and (Test-Path -LiteralPath $dst)) { $records += @{from=$entry.Key; to=$entry.Value}; continue }
 if((Test-Path -LiteralPath $dst) -and (Get-Item -LiteralPath $dst).PSIsContainer -and !(Get-ChildItem -LiteralPath $dst -Force)) { Remove-Item -LiteralPath $dst }
 if(Test-Path -LiteralPath $dst) {throw "Destination already exists: $dst"}
 New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
 Move-Item -LiteralPath $src -Destination $dst
 $records += @{from=$entry.Key; to=$entry.Value}
}
$records | ConvertTo-Json | Set-Content docs/reorganization/moves.json
# Only the inspected TeX smoke build and its disposable compiler cache are removed.
$target = (Resolve-Path -LiteralPath 'build').Path
if($target -ne (Join-Path $root 'build')) {throw 'Unexpected build path'}
Remove-Item -LiteralPath $target -Recurse -Force


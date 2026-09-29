from pathlib import Path
import hashlib
import json
import textwrap
import nbformat as nb

BUILD = Path(__file__).resolve().parent
PACKAGE = BUILD / "stage04_colab_bundle"


def md(text):
    return nb.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nb.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def create(manifest_hash):
    cells = [md('''
        # Stage 04 — frozen caches, eight CUDA experiments

        **Running all cells starts training.** No model was trained while building this bundle.

        1. Upload `stage04_colab_bundle.zip` to **My Drive** (the Drive root).
        2. Extract **only this notebook** from that ZIP on your computer and open it using
           Colab **File → Upload notebook**.
        3. Select **Runtime → Change runtime type → GPU** (for example T4).
        4. Run all cells and authorize the Drive mount. The default paths below need no edits.

        Uses the verified 03A/03B caches without extraction or preprocessing. Four models per branch:
        Small CNN, DS-CNN, TC-ResNet8, CNN+LSTM. Seed 42; TRAIN-only weighted CE;
        Adam 0.001; batch 64; max 50 epochs; patience 8; earliest minimum validation loss;
        no augmentation, extra normalization, AMP or TF32. TEST follows checkpoint selection only.

        CUDA is the explicitly requested change from the canonical CPU execution environment.
        Models and learning rules are unchanged; CPU/GPU bitwise equivalence is not claimed.
        Existing local CPU results are not included or reused. Results go to a separate Drive folder.
    '''), code('''
        from pathlib import Path
        import os, sys, json, hashlib, zipfile, tempfile, subprocess, shutil

        USE_DRIVE = True
        RUN_TRAINING = True  # Set False for Colab preflight only.
        SUMMARIZE = True
        if USE_DRIVE:
            from google.colab import drive
            drive.mount('/content/drive')

        BUNDLE_ZIP = Path('/content/drive/MyDrive/stage04_colab_bundle.zip') if USE_DRIVE else Path('/content/stage04_colab_bundle.zip')
        RESULTS = Path('/content/drive/MyDrive/stage04_colab_results') if USE_DRIVE else Path('/content/stage04_colab_results')
        BUNDLE_ROOT = Path('/content/stage04_colab_bundle')
        RUNTIME_ROOT = Path('/content/stage04_runtime')
        if not BUNDLE_ZIP.is_file():
            raise FileNotFoundError(f'Upload the bundle ZIP to {BUNDLE_ZIP} first.')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = '42'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['STAGE04_RESULTS'] = str(RESULTS)
        print('Bundle:', BUNDLE_ZIP, '\\nResults:', RESULTS)
    '''), md('''
        ## Extract and authenticate the portable package
        The approximately 1.3 GB uncompressed caches stay on the runtime's local disk for fast reads.
        Existing extracted files are revalidated. A partial/different extraction fails safely.
    '''), code(f'''
        EXPECTED_MANIFEST_SHA256 = {manifest_hash!r}

        def file_sha256(path):
            with Path(path).open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()

        if not BUNDLE_ROOT.exists():
            with tempfile.TemporaryDirectory(prefix='stage04-extract-', dir='/content') as temporary:
                staged = Path(temporary) / 'bundle'
                staged.mkdir()
                with zipfile.ZipFile(BUNDLE_ZIP) as archive:
                    for item in archive.infolist():
                        target = (staged / item.filename).resolve()
                        if not target.is_relative_to(staged.resolve()):
                            raise RuntimeError('Unsafe archive path')
                        if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                            raise RuntimeError('Archive symlinks are not supported')
                    archive.extractall(staged)
                os.replace(staged, BUNDLE_ROOT)
        manifest_path = BUNDLE_ROOT / 'bundle_manifest.json'
        if file_sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError('Bundle manifest mismatch. Use the notebook and ZIP from the same export.')
        inventory = json.loads(manifest_path.read_text())
        for relative, expected in inventory['sha256'].items():
            if file_sha256(BUNDLE_ROOT / relative) != expected:
                raise RuntimeError(f'Bundle file changed: {{relative}}')
        print('PASS: all packaged module, config, provenance and data hashes match.')
    '''), md('''
        ## Create a clean Colab Python environment
        Uses the hosted runtime's Python 3.11–3.13 and installs pinned PyTorch 2.8.0 CUDA 12.6,
        NumPy 2.4.6 and pandas 3.0.6. Dependencies download from PyPI/PyTorch; they are not Windows
        binaries bundled from the local environment. Training runs in fresh subprocesses, so Colab's
        already-imported packages cannot silently override these versions. Initial setup may take minutes.
    '''), code('''
        if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
            raise RuntimeError('Use a Colab runtime with Python 3.11, 3.12 or 3.13 for these pinned wheels.')

        def stream_command(command, cwd=None):
            process = subprocess.Popen([str(x) for x in command], cwd=cwd,
                                       env=os.environ.copy(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
            finally:
                process.stdout.close()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)

        PYTHON = RUNTIME_ROOT / 'bin/python'
        if not PYTHON.exists():
            stream_command([sys.executable, '-m', 'venv', RUNTIME_ROOT])
        stream_command([PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--progress-bar', 'off', '--only-binary=:all:',
                        '-r', BUNDLE_ROOT / 'requirements.txt'])
        stream_command([PYTHON, '-m', 'pip', 'check'])

        def stage04(action):
            stream_command([PYTHON, '-B', BUNDLE_ROOT / 'run_stage04.py', action], cwd=BUNDLE_ROOT)
    '''), md('''
        ## Mandatory preflight — no training
        Verify both caches, 32,645 rows per branch, all splits/coordinates/labels, zero source-name
        leakage, TRAIN-only weights, tensor finiteness and all eight model/shape forward passes on CUDA.
        No CPU fallback is used. A missing GPU, corrupt cache or version mismatch stops the notebook.
    '''), code("stage04('preflight')"), md('''
        ## Train 03A, then 03B
        Each branch runs the same four model families in the canonical order. Every model starts from
        seed 42. The best validation checkpoint is reloaded before its one final TEST evaluation.
        Checkpoints, per-epoch histories, selections, predictions and complete results are written directly
        to `RESULTS/runs/`. Confusion matrices and class order are inside the metric JSON records.

        Completed matching runs are verified and skipped when rerunning this notebook. An interrupted
        run stops with its directory preserved; optimizer resume is intentionally not added to the protocol.
        Do not delete/retrain completed runs based on their TEST scores. Colab may disconnect; Drive
        preserves files but does not resume an incomplete optimizer state.
    '''), code('''
        if RUN_TRAINING:
            stage04('03A')
            stage04('03B')
        else:
            print('Preflight only. No training requested in this notebook run.')
    '''), md('''
        ## Optional authenticated comparison and export
        All eight complete matching runs are required for comparison. Summary CSV/JSON and a combined
        `confusion_matrices.json` are saved. The inherited shortlist is descriptive validation-only output;
        no Stage-05 work is performed. A ZIP of results is also saved next to the results directory.
    '''), code('''
        if SUMMARIZE and RUN_TRAINING:
            stage04('compare')
            summary = json.loads((RESULTS / 'comparison.json').read_text())
            print(json.dumps(summary['experiments'], indent=2))
        if RESULTS.exists():
            output_zip = shutil.make_archive(str(RESULTS.parent / (RESULTS.name + '_export')),
                                             'zip', root_dir=RESULTS)
            print('Saved results archive:', output_zip)
            print('Download this file from Google Drive (or Colab Files if USE_DRIVE=False).')
    '''), md('''
        ## Reproducibility record
        `provenance/export_contract.json` preserves local input authentication, cache-generation software,
        frontend identities, class weights and original-source hashes. Portable authentication consumes
        the unchanged cache bytes, manifest and split; raw WAVs and upstream notebooks are deliberately
        omitted. The original model file is byte-identical. Runtime records add the actual CUDA/GPU,
        cuDNN, Python and package versions. GPU/CPU runs must not be mixed as one identical-runtime benchmark.

        References: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
        [PyTorch 2.8 CUDA wheels](https://pytorch.org/get-started/previous-versions/),
        [PyTorch reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html).
    ''')]
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": "stage04_colab.ipynb", "provenance": []}})
    nb.validate(notebook)
    nb.write(notebook, PACKAGE / "stage04_colab.ipynb")


if __name__ == "__main__":
    manifest = {"sha256": {p.relative_to(PACKAGE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in PACKAGE.rglob('*') if p.is_file()
                           and p.name not in {'bundle_manifest.json', 'stage04_colab.ipynb'}}}
    path = PACKAGE / "bundle_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    create(hashlib.sha256(path.read_bytes()).hexdigest())

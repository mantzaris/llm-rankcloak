"""Run V4 Stage 1 without generating or refitting historical evidence."""
import argparse
import json
from pathlib import Path
from rankcloak.revision_v4_stage1 import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/revision_v4'))
    parser.add_argument('--align-tokenizers', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[1], args.output_dir, args.align_tokenizers), indent=2))

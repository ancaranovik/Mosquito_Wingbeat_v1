"""Read-only portability validation; original expensive validator is archived.

No preprocessing, training, notebook rewriting or accepted-report replacement.
"""
import json
from verify_portability import verify

if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))

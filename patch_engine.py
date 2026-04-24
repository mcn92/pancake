#!/usr/bin/env python3
"""
Patches dist/engine.js to fix ENVIRONMENT_IS_NODE=false.

Run from your project root after each build: python3 patch_engine.py
"""

import sys
import re

path = "dist/engine.js"

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

original = src

# ---------------------------------------------------------------------------
# Fix 1: ENVIRONMENT_IS_NODE=false
# ---------------------------------------------------------------------------
old_node = 'var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";'
if old_node in src:
    src = src.replace(old_node, "var ENVIRONMENT_IS_NODE=false;")
    print("✓ Fixed ENVIRONMENT_IS_NODE")
else:
    print("  ENVIRONMENT_IS_NODE already patched or not found")

if src == original:
    print("WARNING: No changes made")
    sys.exit(1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print(f"✓ Patched {path}")
print("  Run: npx wrangler dev --local")

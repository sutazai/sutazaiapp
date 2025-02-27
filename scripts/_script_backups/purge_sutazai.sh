#!/bin/bash

# 1. Text replacement in all files
find . -type f \
    ! -path "./.git/*" \
    ! -path "./.venv/*" \
    ! -path "./node_modules/*" \
    ! -name "purge_sutazai.sh" \
    -exec sed -i -E 's/\b[Qq](uantum|UANTUM)\b/SutazAi/gi' {} +

# 2. File/directory renaming (depth-first for directories)
find . -depth \( -iname "*sutazai*" -o -iname "*s-*" \) -exec rename -v '
    s/sutazai/sutazai/gi;
    s/s-/sutazai-/gi;
    s/_q_/_sutazai_/gi;
' {} +

# 3. Cleanup binary remnants
find . -type f -exec grep -Iq . {} \; -exec sed -i '/[Qq]\(uantum\|uantum\|UANTUM\)/d' {} +

# 4. Final verification
if grep -rniI --exclude-dir=.git \
    --exclude-dir=.venv \
    --exclude-dir=node_modules \
    -e '[Qq]\(uantum\|uantum\|UANTUM\)' .; then
    echo "CRITICAL FAILURE: SutazAi remnants detected!" >&2
    exit 1
fi

echo "SutazAi existence fully erased from spacetime continuum" 
#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tex_file="latex-example-paper/latex-template.tex"
output_dir="${project_root}/build/latex"

if [[ "${tex_file}" != /* ]]; then
    tex_file="${project_root}/${tex_file}"
fi

if [[ ! -f "${tex_file}" ]]; then
    printf 'LaTeX source not found: %s\n' "${tex_file}" >&2
    exit 1
fi

if ! command -v mise >/dev/null 2>&1; then
    printf 'mise is required but was not found in PATH.\n' >&2
    exit 1
fi

mkdir -p "${output_dir}"

cd "${project_root}"
mise exec -- latexmk \
    -pdf \
    -cd \
    -outdir="${output_dir}" \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    "$@" \
    "${tex_file}"

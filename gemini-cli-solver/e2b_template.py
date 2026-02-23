#!/usr/bin/env python3
"""E2B sandbox template for Gemini CLI ARC-AGI solver.

Defines and builds the Firecracker microVM template with:
- Python 3.11 + scientific packages
- Node.js 22 + Gemini CLI (npm)

Usage:
  python e2b_template.py          # Build the template
  python e2b_template.py --name   # Build with custom template name
"""

import argparse

from e2b import Template, default_build_logger


TEMPLATE_NAME = "arc-gemini-solver"


def define_template() -> Template:
    """Define the sandbox template with Node.js 22 + Gemini CLI + Python scientific packages."""
    return (
        Template()
        .from_python_image("3.11")
        .apt_install(["curl", "ca-certificates", "gnupg", "jq"])
        .run_cmd(
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - "
            "&& apt-get install -y nodejs",
            user="root",
        )
        .run_cmd("npm install -g @google/gemini-cli", user="root")
        .pip_install([
            "numpy", "scipy", "matplotlib", "pandas",
            "scikit-learn", "scikit-image", "pillow",
            "sympy", "networkx",
        ])
        .run_cmd("mkdir -p /workspace && mkdir -p /app", user="root")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E2B template for Gemini CLI ARC solver")
    parser.add_argument("--name", default=TEMPLATE_NAME,
                        help=f"Template name (default: {TEMPLATE_NAME})")
    parser.add_argument("--cpu", type=int, default=2,
                        help="CPU count (default: 2)")
    parser.add_argument("--memory", type=int, default=8192,
                        help="Memory in MB (default: 8192)")
    args = parser.parse_args()

    template = define_template()

    print(f"Building E2B template '{args.name}' (cpu={args.cpu}, memory={args.memory}MB)...")
    result = Template.build(
        template,
        args.name,
        cpu_count=args.cpu,
        memory_mb=args.memory,
        on_build_logs=default_build_logger(),
    )
    print(f"Template built successfully: {result.template_id} ({args.name})")


if __name__ == "__main__":
    main()

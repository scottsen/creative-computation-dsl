# Scripts & Tools

Developer tools and utilities for the Morphogen project.

## Available Tools

### reveal.sh - Progressive File Explorer

**Purpose**: Explore large files incrementally at 4 levels of detail to manage token usage and understand structure before reading full content.

**Installation**:
```bash
pip install git+https://gist.github.com/scottsen/ee3fff354a79032f1c6d9d46991c8400
```

**Usage**:
```bash
# Level 0: Metadata (filename, size, type, line count, hash)
./scripts/reveal.sh 0 src/morphogen/domains/audio.py

# Level 1: Structure (imports, classes, functions)
./scripts/reveal.sh 1 src/morphogen/domains/audio.py

# Level 2: Preview (representative sample)
./scripts/reveal.sh 2 docs/specifications/SPEC-AUDIO.md

# Level 3: Full content (with line numbers, paging)
./scripts/reveal.sh 3 SPECIFICATION.md --page-size 50

# With grep filtering
./scripts/reveal.sh 1 src/morphogen/compiler/ --grep "class.*Analyzer"
```

**When to Use**:
- Before reading large files (2000+ lines)
- Surveying domain implementations
- Understanding documentation structure
- Exploring test file organization
- Conserving tokens (get 80% info at 20% token cost)

**Supported File Types**:
- **Python**: AST analysis (imports, classes, functions, docstrings)
- **YAML**: Key structure, nesting depth
- **JSON**: Object/array counts, max depth
- **Markdown**: Heading hierarchy, code blocks
- **Text**: Generic line/word counts

---

### gh.py - GitHub Issue & PR Manager

**Purpose**: Manage GitHub issues and PRs from the command line without requiring `gh` CLI.

**Installation**:
```bash
# Add to PATH or use directly
alias tia-gh='python /home/user/kairo/scripts/gh.py'
```

**Authentication**:
Requires GitHub token from:
1. `GITHUB_TOKEN` environment variable
2. `~/.config/gh/hosts.yml` (gh CLI config)

**Usage**:
```bash
# Issues
tia-gh issue 42                     # View issue #42
tia-gh issue list                   # List open issues
tia-gh issue create "Bug fix"       # Create new issue
tia-gh issue 42 --comment "Done!"   # Comment on issue
tia-gh issue 42 --close             # Close issue

# Pull Requests
tia-gh pr 38                        # View PR #38
tia-gh pr list                      # List open PRs
tia-gh pr 38 files                  # Show changed files
tia-gh pr 38 --merge                # Merge PR
tia-gh pr create "Feature"          # Create PR from current branch

# Shortcuts
tia-gh i 42                         # View issue
tia-gh p 38                         # View PR
tia-gh p 38 m                       # Merge PR
```

**Features**:
- Auto-detects repository from git config
- Colored terminal output
- Works without `gh` CLI installed
- Supports all major GitHub operations

---

## Development Workflow

### Exploring the Codebase

1. **Start with structure** (Level 1):
   ```bash
   ./scripts/reveal.sh 1 src/morphogen/domains/audio.py
   ```

2. **Preview if needed** (Level 2):
   ```bash
   ./scripts/reveal.sh 2 src/morphogen/domains/audio.py
   ```

3. **Read full content** only when necessary (Level 3):
   ```bash
   ./scripts/reveal.sh 3 src/morphogen/domains/audio.py
   ```

### Working with Issues

```bash
# List current issues
tia-gh issue list

# View specific issue
tia-gh issue 123

# Close issue with comment
tia-gh issue 123 --comment "Fixed in PR #125" --close
```

### Creating Pull Requests

```bash
# Create PR from current branch
tia-gh pr create --title "Add reveal tool documentation" \
                 --body "Documents the progressive reveal CLI for codebase exploration"

# Check PR status
tia-gh pr list

# Merge when ready
tia-gh pr 126 --merge
```

---

## Adding New Tools

When adding new developer tools:

1. **Create the script** in `scripts/`
2. **Make it executable**: `chmod +x scripts/your_tool.sh`
3. **Document it here** in this README
4. **Update claude.md** if Claude should know about it
5. **Add tests** if applicable

### Tool Template

```bash
#!/usr/bin/env bash
# Tool Name - Brief Description
#
# Detailed description of what this tool does and why it exists.
#
# Usage:
#   ./scripts/tool_name.sh <args>

set -euo pipefail

# Your implementation here
```

---

## Tips for Claude

When Claude is working with this codebase:

- **Use reveal for exploration**: Check file structure before reading full content
- **Level 1 is often enough**: Imports, classes, and functions give good context
- **Level 2 for sampling**: Get representative content without full file
- **Level 3 sparingly**: Only when you need complete file contents
- **Grep across levels**: Filter output to find specific patterns

**Token Conservation Strategy**:
```bash
# Instead of reading 2000-line file directly:
reveal --level 1 large_file.py              # Structure only (20% tokens)
reveal --level 1 large_file.py --grep "Foo" # Find what you need
# Then read specific sections with regular Read tool
```

---

**Maintained by**: Morphogen Development Team
**Last Updated**: 2025-11-21

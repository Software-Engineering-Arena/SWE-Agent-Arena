---
title: SWE-Agent-Arena
emoji: ⚔️
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
hf_oauth: true
short_description: Agent arena for software engineering tasks
---

# SWE-Agent-Arena

An open-source platform for evaluating CLI coding agents on real software engineering tasks. Two anonymous agents tackle the same task in isolated environments — you compare their output and git diffs, then vote.

**[Try it on HF Spaces](https://huggingface.co/spaces/SE-Arena/SWE-Agent-Arena)**

## Key Capabilities

- **Blind pairwise comparison** with live-streaming output and side-by-side git diffs
- **Multi-round conversations** — send follow-ups to each agent independently, mirroring real iterative workflows
- **RepoChat** — auto-inject repo context (issues, PRs, files) from GitHub / GitLab / HuggingFace URLs
- **Rich leaderboard** — Elo, Bradley-Terry MLE, PageRank, CEI (conversation efficiency), MCS (consistency), and Newman modularity

## How It Works

1. **Submit a task** — sign in, describe an SE task (optionally paste a repo URL for context)
2. **Watch agents work** — two anonymous agents run in parallel with live stdout
3. **Compare diffs** — side-by-side view of what each agent changed
4. **Vote** — Agent A, Agent B, Tie, or Tie (Both Bad)

## Terms of Service

- **Research preview** — limited safety measures; may generate offensive content.
- Must not be used for illegal, harmful, violent, racist, or sexual purposes.
- Do not upload private information.
- Collected dialogue data may be distributed under a **CC-BY** or similar license.

## Contributing

Issues, tasks, and PRs welcome — [open an issue](https://github.com/Software-Engineering-Arena/SWE-Agent-Arena/issues/new) to get started.

## Citation

```bibtex
@inproceedings{zhao2025se,
  title={SE Arena: An Interactive Platform for Evaluating Foundation Models in Software Engineering},
  author={Zhao, Zhimin},
  booktitle={2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge)},
  pages={78--81},
  year={2025},
  organization={IEEE}
}
```

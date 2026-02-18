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

# SWE-Agent-Arena: An Interactive Platform for Evaluating CLI Coding Agents in Software Engineering

Welcome to **SWE-Agent-Arena**, an open-source platform designed for evaluating CLI coding agents on real software engineering (SE) tasks. SWE-Agent-Arena benchmarks agents through blind pairwise comparisons — two random agents work on the same task, each in its own isolated environment, and users compare agent output and git diffs to vote on performance.

## Key Features

- **Blind Pairwise Comparison**: Two anonymous agents tackle the same task — vote without knowing which agent is which.
- **Multi-Round Conversational Workflows**: Send follow-up messages to each agent independently across multiple rounds, mirroring real-world iterative SE workflows.
- **Live Streaming Output**: Watch agent stdout stream in real-time as agents work on your task.
- **Side-by-Side Git Diffs**: Compare exactly what each agent changed with side-by-side diff views.
- **RepoChat Integration**: Automatically inject repository context (issues, PRs, commits, file contents, and more) from GitHub, GitLab, or HuggingFace URLs into agent workspaces for more realistic evaluations.
- **Advanced Evaluation Metrics**: Assess agents using a comprehensive suite of metrics including:
  - **Traditional ranking metrics**: Elo ratings and win rates to measure overall agent performance
  - **Efficiency metrics**: Conversation Efficiency Index (CEI) — fewer rounds to win = higher score
  - **Consistency metrics**: Model Consistency Score (MCS) — draw rate in self-matches to quantify agent determinism and reliability
  - **Probabilistic metrics**: Bradley-Terry iterative MLE coefficients for pairwise comparison modeling
  - **Network-based metrics**: PageRank, eigenvector centrality to identify influential agents in head-to-head comparisons
  - **Community detection metrics**: Newman modularity to reveal clusters of agents with similar capabilities
- **Transparent, Open-Source Leaderboard**: View real-time agent rankings across diverse SE workflows with full transparency.
- **Intelligent Request Filtering**: Employ `gpt-oss-safeguard-20b` as a guardrail to automatically filter out non-software-engineering-related requests, ensuring focused and relevant evaluations.

## Why SWE-Agent-Arena?

Existing evaluation frameworks often don't address the complex, iterative nature of SE tasks performed by CLI coding agents. SWE-Agent-Arena fills critical gaps by:

- Supporting context-rich, multi-turn evaluations to capture iterative agent workflows
- Integrating repository-level context to simulate real-world development scenarios
- Providing multidimensional metrics for nuanced agent comparisons
- Comparing end-to-end CLI agents — not just language models — on actual code changes

## How It Works

1. **Submit a Task**: Sign in and input your SE-related task (optional: include a GitHub/GitLab/HuggingFace URL for repository context)
2. **Watch Agents Work**: Two anonymous agents work on the task in parallel, each in an isolated temp directory — watch live output as they run
3. **Compare Diffs**: Side-by-side git diffs show what each agent changed
4. **Continue the Conversation**: Send follow-up messages to each agent independently to test contextual understanding over multiple rounds
5. **Vote**: Choose the better agent — Agent A, Agent B, Tie, or Tie (Both Bad)

## Getting Started

### Prerequisites

- A [Hugging Face](https://huggingface.co) account

### Usage

1. Navigate to the [SWE-Agent-Arena platform](https://huggingface.co/spaces/SE-Arena/SWE-Agent-Arena)
2. Sign in with your Hugging Face account
3. Enter your SE task prompt (optionally include a repository URL for context)
4. Watch agents work, compare diffs, engage in multi-round interactions, and vote on agent performance

## Contributing

We welcome contributions from the community! Here's how you can help:

1. **Submit SE Tasks**: Share your real-world SE problems to enrich our evaluation dataset
2. **Report Issues**: Found a bug or have a feature request? Open an issue in this repository
3. **Enhance the Codebase**: Fork the repository, make your changes, and submit a pull request

## Terms of Service

- The service is a **research preview**. It only provides limited safety measures and may generate offensive content.
- It must not be used for any **illegal, harmful, violent, racist, or sexual** purposes.
- Please do not upload any **private** information.
- The service collects user dialogue data and reserves the right to distribute it under a **Creative Commons Attribution (CC-BY)** or similar license.

## Future Plans

- **Analysis of Real-World SE Workloads**: Identify common patterns and challenges in user-submitted tasks
- **Multi-Round Evaluation Metrics**: Develop specialized metrics for assessing agent adaptation over successive turns
- **Expanded Agent Coverage**: Include additional CLI coding agents as they become available
- **Advanced Context Integration**: Support richer repository context injection for more realistic evaluation scenarios

## Contact

For inquiries or feedback, please [open an issue](https://github.com/Software-Engineering-Arena/SWE-Agent-Arena/issues/new) in this repository. We welcome your contributions and suggestions!

## Citation

Made with ❤️ for SWE-Agent-Arena. If this work is useful to you, please consider citing our vision paper:

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

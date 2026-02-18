// References for model evaluation metrics:
// - Chatbot Arena: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH
// - Evalica: https://github.com/dustalov/evalica/blob/master/Chatbot-Arena.ipynb

import "dotenv/config";
import { mkdtempSync, rmSync } from "node:fs";
import { spawn, execFile, execFileSync } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { URL } from "node:url";

import express from "express";
import cookieSession from "cookie-session";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import { Gitlab } from "@gitbeaker/rest";
import { uploadFile, listFiles, downloadFile } from "@huggingface/hub";
import whichSync from "which";

const execFileAsync = promisify(execFile);

// ---------------------------------------------------------------------------
// Environment & constants
// ---------------------------------------------------------------------------

const openaiClient = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
});

const CLI_DATA_REPO = "SWE-Arena/cli_data";
const LEADERBOARD_REPO = "SWE-Arena/leaderboard_data";
const VOTE_REPO = "SWE-Arena/vote_data";
const CONVERSATION_REPO = "SWE-Arena/conversation_data";
const LEADERBOARD_FILE = "agent_arena";

const AGENT_TIMEOUT = 300_000; // 5 minutes per agent (ms)
const AGENT_TIMEOUT_LABEL = `${AGENT_TIMEOUT / 60_000}min`;
const LEADERBOARD_UPDATE_TIME_FRAME_DAYS = 365;

let leaderboardCache = null; // in-memory cache, populated at startup

const SHOW_HINT_STRING = true;
const HINT_STRING = "Once signed in, your votes will be recorded securely.";

const SYSTEM_PREFIX =
  "You MUST operate entirely within the current working directory. " +
  "Do NOT read, write, or execute anything outside this directory.";

// ---------------------------------------------------------------------------
// Agent definitions — loaded from HF dataset SWE-Arena/cli_data at startup.
// Each {id}.json declares CLI binary and two "styles" that drive generic
// buildAgentCommand() / runFollowup().
//
// promptStyle:
//   "flag"  → [bin, "-p", <prompt>, ...initArgs]
//   "exec"  → [bin, "exec", ...initArgs, <prompt>]
//
// followupStyle:
//   "continue" → [bin, "-p", <followup>, ...followupArgs]   (e.g. --continue)
//   "resume"   → [bin, "exec", ...followupArgs, "resume", "--last", <followup>]
//   "replay"   → rebuild full conversation, then use promptStyle
// ---------------------------------------------------------------------------

let agents = [];
let agentById = {};
let agentByName = {};

async function loadAgentsFromHf() {
  const token = process.env.HF_TOKEN;
  const credentials = token ? { accessToken: token } : undefined;
  const repo = { type: "dataset", name: CLI_DATA_REPO };
  const loaded = [];

  for await (const file of listFiles({ repo, credentials })) {
    if (!file.path.endsWith(".json")) continue;
    // Skip hidden / nested paths (e.g. .gitattributes)
    if (file.path.includes("/")) continue;

    const resp = await downloadFile({ repo, path: file.path, credentials });
    if (!resp) continue;

    const data = JSON.parse(await resp.text());
    const name = file.path.replace(/\.json$/, "");
    loaded.push({ id: data.bin, name, ...data });
  }

  agents = loaded;
  agentById = Object.fromEntries(agents.map((a) => [a.id, a]));
  agentByName = Object.fromEntries(agents.map((a) => [a.name, a]));
  console.log(`Loaded ${agents.length} agent(s) from ${CLI_DATA_REPO}: ${agents.map((a) => a.name).join(", ")}`);
}

// ---------------------------------------------------------------------------
// CLI availability
// ---------------------------------------------------------------------------

function availableAgents() {
  return agents.filter((a) => {
    try {
      whichSync.sync(a.bin);
      return true;
    } catch {
      return false;
    }
  });
}

// ---------------------------------------------------------------------------
// URL parsing helpers
// ---------------------------------------------------------------------------

function parseUrlPath(url) {
  try {
    const parsed = new URL(url);
    const segments = parsed.pathname.split("/").filter(Boolean);
    return { hostname: parsed.hostname || "", segments };
  } catch {
    return { hostname: null, segments: [] };
  }
}

// ---------------------------------------------------------------------------
// GitHub
// ---------------------------------------------------------------------------

const octokit = process.env.GITHUB_TOKEN
  ? new Octokit({ auth: process.env.GITHUB_TOKEN })
  : new Octokit();

function classifyGithubUrl(segments) {
  if (segments.length < 2) return null;
  let repo = segments[1];
  if (repo.endsWith(".git")) repo = repo.slice(0, -4);
  const base = { owner: segments[0], repo };

  if (segments.length === 2) return { ...base, resource: null };

  const res = segments[2];

  if (res === "issues" && segments.length >= 4)
    return { ...base, resource: "issues", id: segments[3] };
  if (res === "pull" && segments.length >= 4)
    return { ...base, resource: "pull", id: segments[3] };
  if (res === "commit" && segments.length >= 4)
    return { ...base, resource: "commit", sha: segments[3] };
  if (res === "blob" && segments.length >= 4)
    return {
      ...base,
      resource: "blob",
      branch: segments[3],
      path: segments.slice(4).join("/"),
    };
  if (res === "tree" && segments.length >= 4)
    return {
      ...base,
      resource: "tree",
      branch: segments[3],
      path: segments.slice(4).join("/"),
    };
  if (res === "discussions" && segments.length >= 4)
    return { ...base, resource: "discussions", id: segments[3] };
  if (res === "releases" && segments.length >= 5 && segments[3] === "tag")
    return { ...base, resource: "releases", tag: segments[4] };
  if (res === "compare" && segments.length >= 4)
    return { ...base, resource: "compare", spec: segments[3] };
  if (res === "actions" && segments.length >= 5 && segments[3] === "runs")
    return { ...base, resource: "actions", run_id: segments[4] };
  if (res === "wiki")
    return {
      ...base,
      resource: "wiki",
      page: segments.length >= 4 ? segments[3] : null,
    };

  return { ...base, resource: "unknown" };
}

async function fmtGithubRepo(owner, repo) {
  const { data } = await octokit.repos.get({ owner, repo });
  const parts = [`Repository: ${data.full_name}`];
  if (data.description) parts.push(`Description: ${data.description}`);
  try {
    const readme = await octokit.repos.getReadme({ owner, repo });
    const content = Buffer.from(readme.data.content, "base64").toString(
      "utf-8"
    );
    parts.push(`README (first 2000 chars):\n${content.slice(0, 2000)}`);
  } catch {}
  return parts.join("\n\n");
}

async function fmtGithubIssue(owner, repo, issueId) {
  const { data: issue } = await octokit.issues.get({
    owner,
    repo,
    issue_number: Number(issueId),
  });
  const parts = [
    `Issue #${issue.number}: ${issue.title}`,
    `State: ${issue.state}`,
    `Body:\n${issue.body || "(empty)"}`,
  ];
  const { data: comments } = await octokit.issues.listComments({
    owner,
    repo,
    issue_number: Number(issueId),
    per_page: 10,
  });
  if (comments.length) {
    const texts = comments.map(
      (c) => `  Comment by ${c.user.login}:\n  ${c.body}`
    );
    parts.push("Comments (first 10):\n" + texts.join("\n---\n"));
  }
  return parts.join("\n\n");
}

async function fmtGithubPr(owner, repo, prId) {
  const { data: pr } = await octokit.pulls.get({
    owner,
    repo,
    pull_number: Number(prId),
  });
  const parts = [
    `Pull Request #${pr.number}: ${pr.title}`,
    `State: ${pr.state}  Merged: ${pr.merged}`,
    `Body:\n${pr.body || "(empty)"}`,
  ];
  const { data: files } = await octokit.pulls.listFiles({
    owner,
    repo,
    pull_number: Number(prId),
  });
  const diffParts = files.map((f) => {
    const header = `--- ${f.filename} (${f.status}, +${f.additions}/-${f.deletions})`;
    const patch = f.patch || "(binary or too large)";
    return `${header}\n${patch}`;
  });
  if (diffParts.length) {
    let diffText = diffParts.join("\n\n");
    if (diffText.length > 5000)
      diffText = diffText.slice(0, 5000) + "\n... (diff truncated)";
    parts.push(`Diff:\n${diffText}`);
  }
  return parts.join("\n\n");
}

async function fmtGithubCommit(owner, repo, sha) {
  const { data: commit } = await octokit.repos.getCommit({ owner, repo, ref: sha });
  const parts = [
    `Commit: ${commit.sha}`,
    `Message: ${commit.commit.message}`,
    `Author: ${commit.commit.author.name}`,
    `Stats: +${commit.stats.additions}/-${commit.stats.deletions}`,
  ];
  const fileParts = (commit.files || []).map(
    (f) => `  ${f.filename} (${f.status}): ${f.patch || "(binary)"}`
  );
  if (fileParts.length) {
    let patchText = fileParts.join("\n");
    if (patchText.length > 5000)
      patchText = patchText.slice(0, 5000) + "\n... (patch truncated)";
    parts.push(`Files changed:\n${patchText}`);
  }
  return parts.join("\n\n");
}

async function fmtGithubBlob(owner, repo, branch, path) {
  const { data } = await octokit.repos.getContent({
    owner,
    repo,
    path,
    ref: branch,
  });
  if (Array.isArray(data)) {
    const listing = data.map((c) => `  ${c.path} (${c.type})`).join("\n");
    return `Directory listing at ${branch}/${path}:\n${listing}`;
  }
  let content = Buffer.from(data.content, "base64").toString("utf-8");
  if (content.length > 5000)
    content = content.slice(0, 5000) + "\n... (content truncated)";
  return `File: ${path} (branch: ${branch})\n\n${content}`;
}

async function fmtGithubTree(owner, repo, branch, path) {
  const { data } = await octokit.repos.getContent({
    owner,
    repo,
    path: path || "",
    ref: branch,
  });
  const items = Array.isArray(data) ? data : [data];
  const listing = items
    .map((c) => `  ${c.path} (${c.type}, ${c.size} bytes)`)
    .join("\n");
  return `Tree at ${branch}/${path || "(root)"}:\n${listing}`;
}

async function fmtGithubRelease(owner, repo, tag) {
  const { data: release } = await octokit.repos.getReleaseByTag({
    owner,
    repo,
    tag,
  });
  return [
    `Release: ${release.name || release.tag_name}`,
    `Tag: ${release.tag_name}`,
    `Body:\n${release.body || "(empty)"}`,
  ].join("\n\n");
}

async function fmtGithubCompare(owner, repo, spec) {
  let base, head;
  if (spec.includes("...")) [base, head] = spec.split("...", 2);
  else if (spec.includes("..")) [base, head] = spec.split("..", 2);
  else return null;
  const { data } = await octokit.repos.compareCommits({
    owner,
    repo,
    base,
    head,
  });
  const parts = [
    `Comparison: ${base}...${head}`,
    `Status: ${data.status}`,
    `Ahead by: ${data.ahead_by}, Behind by: ${data.behind_by}`,
    `Total commits: ${data.total_commits}`,
  ];
  const commitSummaries = (data.commits || [])
    .slice(0, 20)
    .map((c) => `  ${c.sha.slice(0, 8)}: ${c.commit.message.split("\n")[0]}`);
  if (commitSummaries.length)
    parts.push("Commits:\n" + commitSummaries.join("\n"));
  const fileSummaries = (data.files || [])
    .slice(0, 30)
    .map(
      (f) =>
        `  ${f.filename} (${f.status}, +${f.additions}/-${f.deletions})`
    );
  if (fileSummaries.length)
    parts.push("Files changed:\n" + fileSummaries.join("\n"));
  return parts.join("\n\n");
}

async function fmtGithubActions(owner, repo, runId) {
  const { data: run } = await octokit.actions.getWorkflowRun({
    owner,
    repo,
    run_id: Number(runId),
  });
  const parts = [
    `Workflow Run: ${run.name} #${run.run_number}`,
    `Status: ${run.status}  Conclusion: ${run.conclusion}`,
    `SHA: ${run.head_sha}`,
  ];
  try {
    const { data: jobsData } = await octokit.actions.listJobsForWorkflowRun({
      owner,
      repo,
      run_id: Number(runId),
    });
    for (const job of jobsData.jobs) {
      if (job.conclusion === "failure") {
        parts.push(`Failed job: ${job.name}`);
        for (const step of job.steps || []) {
          if (step.conclusion === "failure")
            parts.push(`  Failed step: ${step.name}`);
        }
      }
    }
  } catch {}
  return parts.join("\n\n");
}

function fmtGithubWiki(owner, repo, page) {
  if (page)
    return `Wiki page: ${page} (from ${owner}/${repo}/wiki)\nNote: Wiki content cannot be fetched via API.`;
  return `Wiki: ${owner}/${repo}/wiki\nNote: Wiki content cannot be fetched via API.`;
}

async function fetchGithubContent(url) {
  if (!process.env.GITHUB_TOKEN) {
    console.log("GITHUB_TOKEN not set.");
    return null;
  }
  const { hostname, segments } = parseUrlPath(url);
  if (!hostname || !hostname.includes("github.com")) return null;
  const info = classifyGithubUrl(segments);
  if (!info) return null;

  try {
    const { owner, repo, resource } = info;
    if (resource === null) return await fmtGithubRepo(owner, repo);
    if (resource === "issues") return await fmtGithubIssue(owner, repo, info.id);
    if (resource === "pull") return await fmtGithubPr(owner, repo, info.id);
    if (resource === "commit") return await fmtGithubCommit(owner, repo, info.sha);
    if (resource === "blob")
      return await fmtGithubBlob(owner, repo, info.branch, info.path);
    if (resource === "tree")
      return await fmtGithubTree(owner, repo, info.branch, info.path);
    if (resource === "releases")
      return await fmtGithubRelease(owner, repo, info.tag);
    if (resource === "compare")
      return await fmtGithubCompare(owner, repo, info.spec);
    if (resource === "actions")
      return await fmtGithubActions(owner, repo, info.run_id);
    if (resource === "wiki") return fmtGithubWiki(owner, repo, info.page);
    return null;
  } catch (err) {
    console.error(`GitHub API error: ${err.message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// GitLab
// ---------------------------------------------------------------------------

const gitlab = process.env.GITLAB_TOKEN
  ? new Gitlab({ token: process.env.GITLAB_TOKEN })
  : null;

function classifyGitlabUrl(segments) {
  let dashIdx = segments.indexOf("-");
  if (dashIdx === -1) {
    if (segments.length >= 2)
      return { projectPath: segments.join("/"), resource: null };
    return null;
  }

  const projectPath = segments.slice(0, dashIdx).join("/");
  const resSegments = segments.slice(dashIdx + 1);

  if (!projectPath || !resSegments.length)
    return { projectPath, resource: null };

  const res = resSegments[0];

  if (res === "issues" && resSegments.length >= 2)
    return { projectPath, resource: "issues", id: resSegments[1] };
  if (res === "merge_requests" && resSegments.length >= 2)
    return { projectPath, resource: "merge_requests", id: resSegments[1] };
  if ((res === "commit" || res === "commits") && resSegments.length >= 2)
    return { projectPath, resource: "commit", sha: resSegments[1] };
  if (res === "blob" && resSegments.length >= 2)
    return {
      projectPath,
      resource: "blob",
      branch: resSegments[1],
      path: resSegments.slice(2).join("/"),
    };
  if (res === "tree" && resSegments.length >= 2)
    return {
      projectPath,
      resource: "tree",
      branch: resSegments[1],
      path: resSegments.slice(2).join("/"),
    };
  if (res === "releases" && resSegments.length >= 2)
    return { projectPath, resource: "releases", tag: resSegments[1] };
  if (res === "compare" && resSegments.length >= 2)
    return { projectPath, resource: "compare", spec: resSegments[1] };
  if (res === "pipelines" && resSegments.length >= 2)
    return { projectPath, resource: "pipelines", id: resSegments[1] };
  if (res === "wikis")
    return {
      projectPath,
      resource: "wikis",
      page: resSegments.length >= 2 ? resSegments[1] : null,
    };

  return { projectPath, resource: "unknown" };
}

async function fetchGitlabContent(url) {
  if (!gitlab) {
    console.log("GITLAB_TOKEN not set.");
    return null;
  }
  const { hostname, segments } = parseUrlPath(url);
  if (!hostname || !hostname.includes("gitlab.com")) return null;
  const info = classifyGitlabUrl(segments);
  if (!info) return null;

  try {
    const project = await gitlab.Projects.show(info.projectPath);
    const { resource } = info;

    if (resource === null) {
      const parts = [`Repository: ${project.path_with_namespace}`];
      if (project.description)
        parts.push(`Description: ${project.description}`);
      try {
        const readme = await gitlab.RepositoryFiles.show(
          project.id,
          "README.md",
          project.default_branch
        );
        const content = Buffer.from(readme.content, "base64").toString("utf-8");
        parts.push(`README (first 2000 chars):\n${content.slice(0, 2000)}`);
      } catch {}
      return parts.join("\n\n");
    }
    if (resource === "issues") {
      const issue = await gitlab.Issues.show(project.id, Number(info.id));
      const parts = [
        `Issue #${issue.iid}: ${issue.title}`,
        `State: ${issue.state}`,
        `Body:\n${issue.description || "(empty)"}`,
      ];
      const notes = await gitlab.IssueNotes.all(project.id, Number(info.id), {
        perPage: 10,
      });
      const noteTexts = notes.map(
        (n) => `  Comment by ${n.author.username}: ${n.body}`
      );
      if (noteTexts.length)
        parts.push("Comments (first 10):\n" + noteTexts.join("\n---\n"));
      return parts.join("\n\n");
    }
    if (resource === "merge_requests") {
      const mr = await gitlab.MergeRequests.show(project.id, Number(info.id));
      const parts = [
        `Merge Request !${mr.iid}: ${mr.title}`,
        `State: ${mr.state}`,
        `Body:\n${mr.description || "(empty)"}`,
      ];
      try {
        const changes = await gitlab.MergeRequests.allDiffs(
          project.id,
          Number(info.id)
        );
        const diffParts = changes
          .slice(0, 30)
          .map(
            (c) =>
              `  ${c.new_path || "?"}: ${(c.diff || "").slice(0, 500)}`
          );
        if (diffParts.length) {
          let diffText = diffParts.join("\n");
          if (diffText.length > 5000)
            diffText = diffText.slice(0, 5000) + "\n... (diff truncated)";
          parts.push(`Changes:\n${diffText}`);
        }
      } catch {}
      return parts.join("\n\n");
    }
    if (resource === "commit") {
      const commit = await gitlab.Commits.show(project.id, info.sha);
      const parts = [
        `Commit: ${commit.id}`,
        `Title: ${commit.title}`,
        `Message: ${commit.message}`,
        `Author: ${commit.author_name}`,
      ];
      try {
        const diffs = await gitlab.Commits.showDiff(project.id, info.sha);
        const diffParts = diffs
          .slice(0, 30)
          .map(
            (d) =>
              `  ${d.new_path || "?"}: ${(d.diff || "").slice(0, 500)}`
          );
        if (diffParts.length) {
          let diffText = diffParts.join("\n");
          if (diffText.length > 5000)
            diffText = diffText.slice(0, 5000) + "\n... (diff truncated)";
          parts.push(`Diff:\n${diffText}`);
        }
      } catch {}
      return parts.join("\n\n");
    }
    if (resource === "blob") {
      const file = await gitlab.RepositoryFiles.show(
        project.id,
        info.path,
        info.branch
      );
      let content = Buffer.from(file.content, "base64").toString("utf-8");
      if (content.length > 5000)
        content = content.slice(0, 5000) + "\n... (content truncated)";
      return `File: ${info.path} (branch: ${info.branch})\n\n${content}`;
    }
    if (resource === "tree") {
      const items = await gitlab.Repositories.allRepositoryTrees(project.id, {
        path: info.path || "",
        ref: info.branch,
        perPage: 100,
      });
      const listing = items
        .map((item) => `  ${item.path} (${item.type})`)
        .join("\n");
      return `Tree at ${info.branch}/${info.path || "(root)"}:\n${listing}`;
    }
    if (resource === "releases") {
      const release = await gitlab.ProjectReleases.show(
        project.id,
        info.tag
      );
      return [
        `Release: ${release.name || release.tag_name}`,
        `Tag: ${release.tag_name}`,
        `Description:\n${release.description || "(empty)"}`,
      ].join("\n\n");
    }
    if (resource === "compare") {
      let base, head;
      if (info.spec.includes("...")) [base, head] = info.spec.split("...", 2);
      else if (info.spec.includes(".."))
        [base, head] = info.spec.split("..", 2);
      else return null;
      const result = await gitlab.Repositories.compare(project.id, base, head);
      const parts = [`Comparison: ${base}...${head}`];
      const commits = (result.commits || [])
        .slice(0, 20)
        .map((c) => `  ${c.short_id || "?"}: ${c.title || ""}`);
      if (commits.length) parts.push("Commits:\n" + commits.join("\n"));
      const diffs = (result.diffs || [])
        .slice(0, 30)
        .map(
          (d) =>
            `  ${d.new_path || "?"}: ${(d.diff || "").slice(0, 500)}`
        );
      if (diffs.length) {
        let diffText = diffs.join("\n");
        if (diffText.length > 5000)
          diffText = diffText.slice(0, 5000) + "\n... (diff truncated)";
        parts.push(`Diffs:\n${diffText}`);
      }
      return parts.join("\n\n");
    }
    if (resource === "pipelines") {
      const pipeline = await gitlab.Pipelines.show(
        project.id,
        Number(info.id)
      );
      const parts = [
        `Pipeline #${pipeline.id}`,
        `Status: ${pipeline.status}`,
        `Ref: ${pipeline.ref}`,
        `SHA: ${pipeline.sha}`,
      ];
      try {
        const jobs = await gitlab.PipelineJobs.all(project.id, pipeline.id, {
          perPage: 20,
        });
        const failed = jobs.filter((j) => j.status === "failed");
        if (failed.length) {
          parts.push("Failed jobs:");
          for (const j of failed)
            parts.push(`  ${j.name}: ${j.status} (stage: ${j.stage})`);
        }
      } catch {}
      return parts.join("\n\n");
    }
    if (resource === "wikis") {
      if (info.page) {
        try {
          const page = await gitlab.Wikis.show(project.id, info.page);
          return `Wiki page: ${page.title}\n\n${page.content}`;
        } catch {
          return `Wiki page: ${info.page}\nNote: Could not fetch wiki page content.`;
        }
      }
      try {
        const pages = await gitlab.Wikis.all(project.id, { perPage: 20 });
        const listing = pages.map((p) => `  ${p.slug}: ${p.title}`).join("\n");
        return `Wiki pages:\n${listing}`;
      } catch {
        return "Wiki: Could not fetch wiki pages.";
      }
    }
    return null;
  } catch (err) {
    console.error(`GitLab API error: ${err.message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// HuggingFace
// ---------------------------------------------------------------------------

function classifyHuggingfaceUrl(segments) {
  if (!segments.length) return null;
  const segs = [...segments];
  let repoType = null;
  if (segs[0] === "datasets" || segs[0] === "spaces") {
    repoType = segs[0] === "datasets" ? "dataset" : "space";
    segs.splice(0, 1);
  }
  if (segs.length < 2) return null;
  const repoId = `${segs[0]}/${segs[1]}`;
  const base = { repoId, repoType };

  if (segs.length === 2) return { ...base, resource: null };
  const res = segs[2];

  if (res === "blob" && segs.length >= 4)
    return {
      ...base,
      resource: "blob",
      revision: segs[3],
      path: segs.slice(4).join("/"),
    };
  if (res === "resolve" && segs.length >= 4)
    return {
      ...base,
      resource: "resolve",
      revision: segs[3],
      path: segs.slice(4).join("/"),
    };
  if (res === "tree" && segs.length >= 4)
    return {
      ...base,
      resource: "tree",
      revision: segs[3],
      path: segs.slice(4).join("/"),
    };
  if (res === "commit" && segs.length >= 4)
    return { ...base, resource: "commit", sha: segs[3] };
  if (res === "discussions" && segs.length >= 4)
    return { ...base, resource: "discussions", num: segs[3] };

  return { ...base, resource: "unknown" };
}

async function fetchHuggingfaceContent(url) {
  const token = process.env.HF_TOKEN;
  if (!token) {
    console.log("HF_TOKEN not set.");
    return null;
  }
  const { hostname, segments } = parseUrlPath(url);
  if (!hostname || !hostname.includes("huggingface.co")) return null;
  const info = classifyHuggingfaceUrl(segments);
  if (!info) return null;

  try {
    const credentials = { accessToken: token };
    const repo = { type: info.repoType || "model", name: info.repoId };

    if (info.resource === null) {
      const parts = [`Repository: ${info.repoId}`];
      try {
        const resp = await downloadFile({ repo, path: "README.md", credentials });
        if (resp) {
          const content = await resp.text();
          parts.push(
            `README (first 2000 chars):\n${content.slice(0, 2000)}`
          );
        }
      } catch {}
      return parts.join("\n\n");
    }
    if (info.resource === "blob" || info.resource === "resolve") {
      try {
        const resp = await downloadFile({
          repo,
          path: info.path,
          revision: info.revision,
          credentials,
        });
        if (resp) {
          let content = await resp.text();
          if (content.length > 5000)
            content = content.slice(0, 5000) + "\n... (content truncated)";
          return `File: ${info.path} (revision: ${info.revision})\n\n${content}`;
        }
      } catch {
        return `File: ${info.path} (revision: ${info.revision})\n(binary or unreadable file)`;
      }
    }
    if (info.resource === "tree") {
      const items = [];
      for await (const entry of listFiles({
        repo,
        path: info.path || undefined,
        revision: info.revision,
        credentials,
      })) {
        items.push(`  ${entry.path} (${entry.type})`);
        if (items.length >= 100) {
          items.push("  ... (truncated)");
          break;
        }
      }
      return `Tree at ${info.revision}/${info.path || "(root)"}:\n${items.join("\n")}`;
    }
    return null;
  } catch (err) {
    console.error(`Hugging Face API error: ${err.message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// URL router
// ---------------------------------------------------------------------------

async function fetchUrlContent(url) {
  if (!url || !url.trim()) return "";
  url = url.trim();
  try {
    const { hostname } = parseUrlPath(url);
    if (hostname && hostname.includes("github.com"))
      return await fetchGithubContent(url);
    if (hostname && hostname.includes("gitlab.com"))
      return await fetchGitlabContent(url);
    if (hostname && hostname.includes("huggingface.co"))
      return await fetchHuggingfaceContent(url);
  } catch (err) {
    console.error(`Error fetching URL content: ${err.message}`);
  }
  return "";
}

// ---------------------------------------------------------------------------
// Agent execution via CLI
// ---------------------------------------------------------------------------

function buildAgentCommand(agent, prompt) {
  switch (agent.promptStyle) {
    case "flag":
      return [agent.bin, ["-p", prompt, ...agent.initArgs]];
    case "exec":
      return [agent.bin, ["exec", ...agent.initArgs, prompt]];
    default:
      throw new Error(`Unknown promptStyle "${agent.promptStyle}" for ${agent.id}`);
  }
}

// Extract human-readable text from agent output (some CLIs return JSON/JSONL)
function parseAgentOutput(raw) {
  if (!raw || typeof raw !== "string") return raw || "";
  const trimmed = raw.trim();

  // Try JSONL first (one JSON object per line — e.g. Grok CLI chat format)
  const lines = trimmed.split("\n").filter((l) => l.trim());
  const hasJsonLines = lines.length > 0 && lines.every((l) => {
    const t = l.trim();
    return t.startsWith("{") || t.startsWith("[");
  });

  if (hasJsonLines && lines.length > 1) {
    // Parse each line, extract assistant messages
    const assistantMsgs = [];
    for (const line of lines) {
      try {
        const obj = JSON.parse(line.trim());
        if (obj.role === "assistant" && obj.content) {
          assistantMsgs.push(obj.content);
        }
      } catch { /* skip unparseable lines */ }
    }
    if (assistantMsgs.length) return assistantMsgs.join("\n\n");
    // No assistant messages — try extracting any content field
    const allContent = [];
    for (const line of lines) {
      try {
        const obj = JSON.parse(line.trim());
        if (obj.content) allContent.push(obj.content);
      } catch { /* skip */ }
    }
    if (allContent.length) return allContent.join("\n\n");
  }

  // Try single JSON object
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const obj = JSON.parse(trimmed);
      const text =
        obj.result || obj.response || obj.content || obj.message ||
        obj.text || obj.output || obj.answer ||
        obj.choices?.[0]?.message?.content ||
        obj.choices?.[0]?.text;
      if (typeof text === "string") return text;
      if (Array.isArray(obj)) {
        const msgs = obj.map((m) => m.content || m.text || "").filter(Boolean);
        if (msgs.length) return msgs.join("\n\n");
      }
    } catch { /* not valid JSON, fall through */ }
  }

  return raw;
}

// Streaming agent runner — returns a live state object + promise
function spawnAgent(agent, prompt, agentDir) {
  const [bin, args] = buildAgentCommand(agent, prompt);
  const state = { stdout: "", stderr: "", done: false, ok: false };

  const proc = spawn(bin, args, { cwd: agentDir, env: { ...process.env } });
  proc.stdout.setEncoding("utf-8");
  proc.stderr.setEncoding("utf-8");
  proc.stdout.on("data", (chunk) => { state.stdout += chunk; });
  proc.stderr.on("data", (chunk) => { state.stderr += chunk; });

  const timer = setTimeout(() => {
    proc.kill();
    state.stderr += `\n[Timeout after ${AGENT_TIMEOUT_LABEL}]`;
  }, AGENT_TIMEOUT);

  state.promise = new Promise((resolve) => {
    proc.on("close", (code) => {
      clearTimeout(timer);
      state.done = true;
      state.ok = code === 0;
      resolve(state);
    });
    proc.on("error", (err) => {
      clearTimeout(timer);
      state.done = true;
      state.ok = false;
      state.stderr += err.message;
      resolve(state);
    });
  });

  return state;
}

// Blocking agent runner — used for followups (shorter, less need for streaming)
async function runAgent(agent, prompt, agentDir) {
  const [bin, args] = buildAgentCommand(agent, prompt);
  try {
    const { stdout, stderr } = await execFileAsync(bin, args, {
      cwd: agentDir,
      timeout: AGENT_TIMEOUT,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    });
    return { ok: true, stdout, stderr };
  } catch (err) {
    const partialOut = err.stdout || "";
    const partialErr = err.stderr || "";
    const prefix = err.killed ? `[Timeout after ${AGENT_TIMEOUT_LABEL}]\n` : "";
    return {
      ok: false,
      stdout: partialOut,
      stderr: prefix + (partialErr || err.message),
    };
  }
}

function rebuildPrompt(rounds, followup) {
  const parts = [];
  for (const r of rounds) {
    parts.push(`User: ${r.prompt}`);
    parts.push(`Assistant: ${r.stdout}`);
  }
  parts.push(`User: ${followup}`);
  return parts.join("\n\n");
}

async function runFollowup(agent, followup, agentDir, rounds) {
  let bin = agent.bin, args;

  switch (agent.followupStyle) {
    case "continue":
      args = ["-p", followup, ...agent.followupArgs];
      break;
    case "resume":
      args = ["exec", ...agent.followupArgs, "resume", "--last", followup];
      break;
    case "replay": {
      const full = rebuildPrompt(rounds, followup);
      args = ["-p", full, ...agent.followupArgs];
      break;
    }
    default:
      throw new Error(`Unknown followupStyle "${agent.followupStyle}" for ${agent.id}`);
  }

  try {
    const { stdout, stderr } = await execFileAsync(bin, args, {
      cwd: agentDir,
      timeout: AGENT_TIMEOUT,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    });
    return { ok: true, stdout, stderr };
  } catch (err) {
    const partialOut = err.stdout || "";
    const partialErr = err.stderr || "";
    const prefix = err.killed ? `[Timeout after ${AGENT_TIMEOUT_LABEL}]\n` : "";
    return {
      ok: false,
      stdout: partialOut,
      stderr: prefix + (partialErr || err.message),
    };
  }
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

function buildPrompt(userPrompt, repoContext = "") {
  const parts = [SYSTEM_PREFIX];
  if (repoContext) parts.push(`Repository context:\n${repoContext}`);
  parts.push(userPrompt);
  return parts.join("\n\n");
}

function stripContext(prompt) {
  const marker = "\n\n";
  // Find the last section which is the user query
  // The prompt format is: SYSTEM_PREFIX + \n\n + [repo context + \n\n] + user query
  // We strip SYSTEM_PREFIX and optional repo context
  let rest = prompt;
  if (rest.startsWith(SYSTEM_PREFIX)) {
    rest = rest.slice(SYSTEM_PREFIX.length);
    if (rest.startsWith("\n\n")) rest = rest.slice(2);
  }
  if (rest.startsWith("Repository context:\n")) {
    const idx = rest.indexOf("\n\n", "Repository context:\n".length);
    if (idx >= 0) rest = rest.slice(idx + 2);
  }
  return rest;
}

// ---------------------------------------------------------------------------
// Git operations (clone, checkout, diff)
// ---------------------------------------------------------------------------

function cloneRepo(url, agentDir) {
  const { hostname, segments } = parseUrlPath(url);
  if (!hostname) return false;

  let parsedInfo = null;
  let cloneUrl = null;

  if (hostname.includes("github.com")) {
    parsedInfo = classifyGithubUrl(segments);
    if (!parsedInfo) return false;
    cloneUrl = `https://github.com/${parsedInfo.owner}/${parsedInfo.repo}.git`;
  } else if (hostname.includes("gitlab.com")) {
    parsedInfo = classifyGitlabUrl(segments);
    if (!parsedInfo) return false;
    cloneUrl = `https://gitlab.com/${parsedInfo.projectPath}.git`;
  } else if (hostname.includes("huggingface.co")) {
    parsedInfo = classifyHuggingfaceUrl(segments);
    if (!parsedInfo) return false;
    const prefix = parsedInfo.repoType ? `${parsedInfo.repoType}s/` : "";
    cloneUrl = `https://huggingface.co/${prefix}${parsedInfo.repoId}`;
  } else {
    return false;
  }

  try {
    execFileSync("git", ["clone", "--depth=1", cloneUrl, "."], {
      cwd: agentDir,
      timeout: 120_000,
      stdio: "pipe",
    });
    checkoutRef(parsedInfo, agentDir);
    return true;
  } catch {
    return false;
  }
}

function checkoutRef(parsedInfo, agentDir) {
  const resource = parsedInfo.resource;
  const run = (args) => {
    try {
      execFileSync("git", args, { cwd: agentDir, timeout: 60_000, stdio: "pipe" });
    } catch {}
  };
  try {
    if (resource === "pull" && parsedInfo.id) {
      run(["fetch", "origin", `pull/${parsedInfo.id}/head:pr`]);
      run(["checkout", "pr"]);
    } else if (resource === "merge_requests" && parsedInfo.id) {
      run(["fetch", "origin", `merge-requests/${parsedInfo.id}/head:mr`]);
      run(["checkout", "mr"]);
    } else if (resource === "commit" && parsedInfo.sha) {
      run(["fetch", "--depth=1", "origin", parsedInfo.sha]);
      run(["checkout", parsedInfo.sha]);
    } else if (
      (resource === "blob" || resource === "tree") &&
      parsedInfo.branch
    ) {
      run(["checkout", parsedInfo.branch]);
    } else if (
      (resource === "blob" || resource === "resolve" || resource === "tree") &&
      parsedInfo.revision
    ) {
      run(["checkout", parsedInfo.revision]);
    }
  } catch {} // best effort
}

function captureDiff(agentDir) {
  try {
    execFileSync("git", ["add", "-A"], {
      cwd: agentDir,
      stdio: "pipe",
    });
    // Exclude CLI-specific config/state files so only the agent's
    // actual work appears in the diff.
    const result = execFileSync(
      "git",
      [
        "diff", "--cached", "--",
        ".",
        // Claude Code
        ":(exclude).claude",
        ":(exclude)CLAUDE.md",
        // Gemini CLI
        ":(exclude).gemini",
        // OpenAI Codex
        ":(exclude).codex",
        ":(exclude)codex.json",
        // Grok CLI
        ":(exclude).grok",
        // Common IDE / tool artifacts
        ":(exclude).vscode",
        ":(exclude)settings.json",
      ],
      {
        cwd: agentDir,
        encoding: "utf-8",
        maxBuffer: 10 * 1024 * 1024,
      }
    );
    return result.slice(0, 100_000);
  } catch (err) {
    console.error(`captureDiff failed: ${err.message}`);
    return "";
  }
}

// ---------------------------------------------------------------------------
// HF data I/O
// ---------------------------------------------------------------------------

async function saveContentToHf(data, repoName, fileName, token) {
  const json = JSON.stringify(data, null, 2);
  const content = new Blob([json]);
  if (!token) token = process.env.HF_TOKEN;
  if (!token) throw new Error("No HF token available for upload.");

  await uploadFile({
    repo: { type: "dataset", name: repoName },
    file: { content, path: `${fileName}.json` },
    credentials: { accessToken: token },
  });
}

function isFileWithinTimeFrame(filePath, days) {
  try {
    const timestampStr = filePath.split("/").pop().replace(".json", "");
    // Format: YYYYMMDD_HHMMSS
    const m = timestampStr.match(
      /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/
    );
    if (!m) return false;
    const fileDate = new Date(
      `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}`
    );
    const diffDays = (Date.now() - fileDate.getTime()) / (1000 * 60 * 60 * 24);
    return diffDays <= days;
  } catch {
    return false;
  }
}

async function loadContentFromHf(repoName, filePrefix) {
  const data = [];
  const token = process.env.HF_TOKEN;
  const credentials = token ? { accessToken: token } : undefined;
  const repo = { type: "dataset", name: repoName };

  try {
    for await (const file of listFiles({ repo, credentials })) {
      if (!file.path.startsWith(`${filePrefix}/`)) continue;
      if (!file.path.endsWith(".json")) continue;
      if (
        !isFileWithinTimeFrame(file.path, LEADERBOARD_UPDATE_TIME_FRAME_DAYS)
      )
        continue;

      const resp = await downloadFile({ repo, path: file.path, credentials });
      if (resp) {
        const entry = JSON.parse(await resp.text());
        entry.timestamp = file.path.split("/").pop().replace(".json", "");
        data.push(entry);
      }
    }
    return data;
  } catch (err) {
    console.error(`Error loading data from HF: ${err.message}`);
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Leaderboard computation (custom JS — no evalica)
// ---------------------------------------------------------------------------

function round2(n) {
  return Math.round(n * 100) / 100;
}

const WINNER_MAP = {
  left: "X",
  right: "Y",
  tie: "draw",
  both_bad: "draw",
};

function computeElo(votes) {
  const K = 32;
  const INITIAL = 1000;
  const scores = {};

  for (const v of votes) {
    scores[v.left] ??= INITIAL;
    scores[v.right] ??= INITIAL;

    const w = WINNER_MAP[v.winner];
    if (w === "draw") continue; // tieWeight = 0

    const rA = scores[v.left];
    const rB = scores[v.right];
    const eA = 1 / (1 + 10 ** ((rB - rA) / 400));
    const eB = 1 - eA;

    const sA = w === "X" ? 1 : 0;
    const sB = w === "Y" ? 1 : 0;

    scores[v.left] += K * (sA - eA);
    scores[v.right] += K * (sB - eB);
  }
  return scores;
}

function computeAvgWinRate(votes) {
  const wins = {};
  const losses = {};

  for (const v of votes) {
    wins[v.left] ??= 0;
    wins[v.right] ??= 0;
    losses[v.left] ??= 0;
    losses[v.right] ??= 0;

    const w = WINNER_MAP[v.winner];
    if (w === "draw") continue;

    if (w === "X") {
      wins[v.left]++;
      losses[v.right]++;
    } else {
      wins[v.right]++;
      losses[v.left]++;
    }
  }

  const result = {};
  for (const name of Object.keys(wins)) {
    const total = wins[name] + losses[name];
    result[name] = total > 0 ? wins[name] / total : 0;
  }
  return result;
}

function computeBradleyTerry(votes, iterations = 100) {
  // Collect agents and win counts
  const agentSet = new Set();
  for (const v of votes) {
    agentSet.add(v.left);
    agentSet.add(v.right);
  }
  const agentList = [...agentSet];
  const n = agentList.length;
  const idx = Object.fromEntries(agentList.map((a, i) => [a, i]));

  // Win matrix
  const W = Array.from({ length: n }, () => new Float64Array(n));
  for (const v of votes) {
    const w = WINNER_MAP[v.winner];
    if (w === "draw") continue;
    const i = idx[v.left];
    const j = idx[v.right];
    if (w === "X") W[i][j]++;
    else W[j][i]++;
  }

  // Iterative MLE
  const p = new Float64Array(n).fill(1 / n);

  for (let iter = 0; iter < iterations; iter++) {
    const pNew = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let num = 0;
      let den = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        num += W[i][j];
        const totalGames = W[i][j] + W[j][i];
        if (totalGames > 0) den += totalGames / (p[i] + p[j]);
      }
      pNew[i] = den > 0 ? num / den : 0;
    }
    // Normalize
    const sum = pNew.reduce((a, b) => a + b, 0);
    if (sum > 0) for (let i = 0; i < n; i++) pNew[i] /= sum;
    for (let i = 0; i < n; i++) p[i] = pNew[i];
  }

  const result = {};
  for (let i = 0; i < n; i++) result[agentList[i]] = p[i];
  return result;
}

function computePageRank(votes, damping = 0.85, iterations = 100) {
  const agentSet = new Set();
  for (const v of votes) {
    agentSet.add(v.left);
    agentSet.add(v.right);
  }
  const agentList = [...agentSet];
  const n = agentList.length;
  const idx = Object.fromEntries(agentList.map((a, i) => [a, i]));

  // Adjacency: edge from loser to winner
  const outLinks = Array.from({ length: n }, () => new Float64Array(n));
  const outDegree = new Float64Array(n);

  for (const v of votes) {
    const w = WINNER_MAP[v.winner];
    if (w === "draw") continue;
    const winner = w === "X" ? idx[v.left] : idx[v.right];
    const loser = w === "X" ? idx[v.right] : idx[v.left];
    outLinks[loser][winner]++;
    outDegree[loser]++;
  }

  let pr = new Float64Array(n).fill(1 / n);

  for (let iter = 0; iter < iterations; iter++) {
    const prNew = new Float64Array(n).fill((1 - damping) / n);
    for (let j = 0; j < n; j++) {
      if (outDegree[j] === 0) {
        // Dangling node: distribute evenly
        for (let i = 0; i < n; i++) prNew[i] += damping * pr[j] / n;
      } else {
        for (let i = 0; i < n; i++) {
          if (outLinks[j][i] > 0) {
            prNew[i] += damping * pr[j] * (outLinks[j][i] / outDegree[j]);
          }
        }
      }
    }
    pr = prNew;
  }

  const result = {};
  for (let i = 0; i < n; i++) result[agentList[i]] = pr[i];
  return result;
}

function computeEigen(votes, iterations = 100) {
  const agentSet = new Set();
  for (const v of votes) {
    agentSet.add(v.left);
    agentSet.add(v.right);
  }
  const agentList = [...agentSet];
  const n = agentList.length;
  const idx = Object.fromEntries(agentList.map((a, i) => [a, i]));

  // Adjacency matrix: wins
  const A = Array.from({ length: n }, () => new Float64Array(n));
  for (const v of votes) {
    const w = WINNER_MAP[v.winner];
    if (w === "draw") continue;
    const i = idx[v.left];
    const j = idx[v.right];
    if (w === "X") A[i][j]++;
    else A[j][i]++;
  }

  // Power iteration for dominant eigenvector
  let vec = new Float64Array(n).fill(1 / Math.sqrt(n));

  for (let iter = 0; iter < iterations; iter++) {
    const newVec = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        newVec[i] += A[i][j] * vec[j];
      }
    }
    // Normalize
    const norm = Math.sqrt(newVec.reduce((s, v) => s + v * v, 0));
    if (norm > 0) for (let i = 0; i < n; i++) newVec[i] /= norm;
    vec = newVec;
  }

  const result = {};
  for (let i = 0; i < n; i++) result[agentList[i]] = vec[i];
  return result;
}

function computeNewman(votes) {
  // Simplified Newman modularity on win-graph
  const agentSet = new Set();
  for (const v of votes) {
    agentSet.add(v.left);
    agentSet.add(v.right);
  }
  const agentList = [...agentSet];
  const n = agentList.length;
  const idx = Object.fromEntries(agentList.map((a, i) => [a, i]));

  const A = Array.from({ length: n }, () => new Float64Array(n));
  let totalEdges = 0;
  const degree = new Float64Array(n);

  for (const v of votes) {
    const w = WINNER_MAP[v.winner];
    if (w === "draw") continue;
    const i = idx[v.left];
    const j = idx[v.right];
    if (w === "X") {
      A[i][j]++;
      A[j][i]++;
    } else {
      A[j][i]++;
      A[i][j]++;
    }
    degree[i]++;
    degree[j]++;
    totalEdges++;
  }

  if (totalEdges === 0) {
    const result = {};
    for (const a of agentList) result[a] = 0;
    return result;
  }

  // Each node in its own community -> modularity contribution
  const result = {};
  for (let i = 0; i < n; i++) {
    const qi =
      (A[i][i] || 0) / (2 * totalEdges) -
      (degree[i] / (2 * totalEdges)) ** 2;
    result[agentList[i]] = qi;
  }
  return result;
}

function computeCeiMcs(votes, conversations) {
  const convMap = new Map();
  for (const c of conversations) {
    convMap.set(`${c.timestamp}|${c.left}|${c.right}`, c);
  }

  const stats = {};

  for (const vote of votes) {
    const conv = convMap.get(
      `${vote.timestamp}|${vote.left}|${vote.right}`
    );

    for (const m of [vote.left, vote.right]) {
      stats[m] ??= { ceiSum: 0, ceiMax: 0, selfMatches: 0, selfDraws: 0 };
    }

    if (vote.left === vote.right) {
      stats[vote.left].selfMatches++;
      if (vote.winner === "tie" || vote.winner === "both_bad") {
        stats[vote.left].selfDraws++;
      }
      continue;
    }

    let leftScore, rightScore;
    switch (vote.winner) {
      case "left":
        leftScore = 1;
        rightScore = -1;
        break;
      case "right":
        leftScore = -1;
        rightScore = 1;
        break;
      case "tie":
        leftScore = 0.3;
        rightScore = 0.3;
        break;
      case "both_bad":
        leftScore = -0.3;
        rightScore = -0.3;
        break;
      default:
        continue;
    }

    // CEI: use conversation rounds if available, default to 1
    const leftRounds = conv?.left_rounds?.length || 1;
    const rightRounds = conv?.right_rounds?.length || 1;

    stats[vote.left].ceiMax += 1 / leftRounds;
    stats[vote.right].ceiMax += 1 / rightRounds;
    stats[vote.left].ceiSum += leftScore / leftRounds;
    stats[vote.right].ceiSum += rightScore / rightRounds;
  }

  const cei = {};
  const mcs = {};
  for (const [agent, s] of Object.entries(stats)) {
    cei[agent] = s.ceiMax > 0 ? round2(s.ceiSum / s.ceiMax) : null;
    mcs[agent] = s.selfMatches > 0 ? round2(s.selfDraws / s.selfMatches) : null;
  }
  return { cei, mcs };
}

async function getLeaderboardData({ voteEntry = null, convEntry = null, useCache = true } = {}) {
  // Return in-memory cache if available and no new vote to incorporate
  if (useCache && leaderboardCache && !voteEntry) return leaderboardCache;

  const token = process.env.HF_TOKEN;
  const credentials = token ? { accessToken: token } : undefined;

  if (useCache && !leaderboardCache) {
    try {
      const resp = await downloadFile({
        repo: { type: "dataset", name: LEADERBOARD_REPO },
        path: `${LEADERBOARD_FILE}.json`,
        credentials,
      });
      if (resp) {
        const parsed = JSON.parse(await resp.text());
        if (Array.isArray(parsed) && parsed.length > 0) {
          leaderboardCache = parsed;
          return leaderboardCache;
        }
        console.log("Leaderboard cache is empty, falling back to vote_data...");
      }
    } catch {
      console.log("No cached leaderboard found, computing from votes...");
    }
  }

  let votes = [];
  try {
    votes = await loadContentFromHf(VOTE_REPO, LEADERBOARD_FILE);
    console.log(`Loaded ${votes.length} vote(s) from ${VOTE_REPO}`);
  } catch (err) {
    console.error(`Failed to load votes: ${err.message}`);
  }
  if (voteEntry) votes.push(voteEntry);
  if (votes.length === 0) return [];

  let conversations = [];
  try {
    conversations = await loadContentFromHf(CONVERSATION_REPO, LEADERBOARD_FILE);
    console.log(`Loaded ${conversations.length} conversation(s) from ${CONVERSATION_REPO}`);
  } catch (err) {
    console.error(`Failed to load conversations (non-fatal): ${err.message}`);
  }
  if (convEntry) conversations.push(convEntry);

  const eloScores = computeElo(votes);
  const winRates = computeAvgWinRate(votes);
  const btScores = computeBradleyTerry(votes);
  const pagerankScr = computePageRank(votes);
  const eigenScores = computeEigen(votes);
  const newmanScores = computeNewman(votes);
  const { cei, mcs } = computeCeiMcs(votes, conversations);

  const agentNames = Object.keys(eloScores);
  const rows = agentNames.map((name) => ({
    Agent: name,
    Provider: (agentByName[name] || agentById[name])?.provider || "",
    "Elo Score": round2(eloScores[name] ?? 0),
    "Win Rate": round2(winRates[name] ?? 0),
    "Conversation Efficiency Index": cei[name] ?? null,
    "Consistency Score": mcs[name] ?? null,
    "Bradley-Terry Coefficient": round2(btScores[name] ?? 0),
    "Eigenvector Centrality Value": round2(eigenScores[name] ?? 0),
    "Newman Modularity Score": round2(newmanScores[name] ?? 0),
    "PageRank Score": round2(pagerankScr[name] ?? 0),
  }));

  rows.sort((a, b) => b["Elo Score"] - a["Elo Score"]);
  rows.forEach((row, i) => {
    row.Rank = i + 1;
  });

  leaderboardCache = rows;

  if (voteEntry && token) {
    saveContentToHf(rows, LEADERBOARD_REPO, LEADERBOARD_FILE, token).catch(
      (err) => console.error(`Failed to save leaderboard cache: ${err.message}`)
    );
  }

  return rows;
}

// ---------------------------------------------------------------------------
// Guardrail
// ---------------------------------------------------------------------------

async function guardrailCheckSeRelevance(userInput) {
  try {
    const response = await openaiClient.chat.completions.create({
      model: "openai/gpt-oss-safeguard-20b",
      messages: [
        {
          role: "system",
          content:
            "You are a classifier that decides if a user's question is relevant to software engineering. " +
            "If the question is about software engineering concepts, tools, processes, or code, respond with 'Yes'. " +
            "Otherwise, respond with 'No'.",
        },
        { role: "user", content: userInput },
      ],
    });
    const classification = response.choices[0].message.content
      .trim()
      .toLowerCase();
    return classification.startsWith("yes");
  } catch (err) {
    console.error(`Guardrail check failed: ${err.message}`);
    return true; // fail open
  }
}

// ---------------------------------------------------------------------------
// Express app
// ---------------------------------------------------------------------------

const app = express();
app.set("trust proxy", true);
app.use(express.json({ limit: "10mb" }));
app.use(express.static("public"));
app.use(
  cookieSession({
    name: "session",
    keys: [process.env.SESSION_SECRET || randomUUID()],
    maxAge: 24 * 60 * 60 * 1000,
  })
);

// In-memory battle state: battleId -> battle object
const battles = new Map();

// ---------------------------------------------------------------------------
// Auth routes (HF OAuth)
// ---------------------------------------------------------------------------

function getRedirectUri(req) {
  // On HF Spaces the SPACE_HOST env var gives the canonical public hostname.
  // Using it avoids http/https mismatches caused by reverse-proxy headers.
  if (process.env.SPACE_HOST) {
    return `https://${process.env.SPACE_HOST}/auth/callback`;
  }
  return `${req.protocol}://${req.get("host")}/auth/callback`;
}

app.get("/auth/login", (req, res) => {
  const clientId = process.env.OAUTH_CLIENT_ID;
  if (!clientId) return res.status(500).json({ error: "OAuth not configured" });

  const redirectUri = getRedirectUri(req);
  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: redirectUri,
    response_type: "code",
    scope: process.env.OAUTH_SCOPES || "openid profile",
    state: randomUUID(),
  });
  res.redirect(`https://huggingface.co/oauth/authorize?${params}`);
});

app.get("/auth/callback", async (req, res) => {
  const { code } = req.query;
  if (!code) {
    console.error("OAuth callback: no code parameter received");
    return res.redirect("/");
  }
  try {
    const redirectUri = getRedirectUri(req);
    const tokenResp = await fetch("https://huggingface.co/oauth/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "authorization_code",
        code,
        redirect_uri: redirectUri,
        client_id: process.env.OAUTH_CLIENT_ID,
        client_secret: process.env.OAUTH_CLIENT_SECRET,
      }),
    });
    const data = await tokenResp.json();
    if (!tokenResp.ok || !data.access_token) {
      console.error(`OAuth token exchange failed (${tokenResp.status}):`, data);
      return res.redirect("/");
    }
    req.session.hfToken = data.access_token;
    res.redirect("/");
  } catch (err) {
    console.error(`OAuth callback error: ${err.message}`);
    res.redirect("/");
  }
});

app.get("/auth/status", (req, res) => {
  const token = process.env.HF_TOKEN;
  res.json({
    authenticated: !!token,
    hint: SHOW_HINT_STRING ? HINT_STRING : "",
  });
});

// ---------------------------------------------------------------------------
// API routes
// ---------------------------------------------------------------------------

app.get("/api/config", (_req, res) => {
  res.json({
    agentTimeoutMin: AGENT_TIMEOUT / 60_000,
    oauthClientId: process.env.OAUTH_CLIENT_ID || "",
  });
});

app.get("/api/leaderboard", async (req, res) => {
  try {
    const data = await getLeaderboardData({ useCache: true });
    res.json(data);
  } catch (err) {
    console.error(`Leaderboard error: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/battle/start", async (req, res) => {
  const { prompt, repoUrl } = req.body;
  if (!prompt || !prompt.trim()) {
    return res.status(400).json({ error: "Prompt is required." });
  }

  // Guardrail (skip if URL provided)
  if (!repoUrl) {
    const isRelevant = await guardrailCheckSeRelevance(prompt);
    if (!isRelevant) {
      return res.status(400).json({
        error:
          "Oops! Try asking something about software engineering. Thanks!",
      });
    }
  }

  const available = availableAgents();
  if (available.length < 2) {
    return res
      .status(500)
      .json({ error: "Not enough agents available for a battle." });
  }

  // Pick 2 random agents independently (may be the same — needed for MCS self-match metric)
  const pick = () => available[Math.floor(Math.random() * available.length)];
  const agentA = pick();
  const agentB = pick();

  // Create temp dirs
  const leftDir = mkdtempSync(join(tmpdir(), "agent_left_"));
  const rightDir = mkdtempSync(join(tmpdir(), "agent_right_"));

  try {
    // Git init or clone
    for (const d of [leftDir, rightDir]) {
      if (repoUrl && repoUrl.trim()) {
        cloneRepo(repoUrl, d);
      } else {
        execFileSync("git", ["init"], { cwd: d, stdio: "pipe" });
      }
    }

    // Fetch context & build prompt
    const repoContext = await fetchUrlContent(repoUrl || "");
    const fullPrompt = buildPrompt(prompt, repoContext);

    // Spawn both agents (non-blocking — returns immediately with live state)
    const leftState = spawnAgent(agentA, fullPrompt, leftDir);
    const rightState = spawnAgent(agentB, fullPrompt, rightDir);

    // Build battle state
    const battleId = randomUUID();
    battles.set(battleId, {
      id: battleId,
      left: agentA.name,
      right: agentB.name,
      leftAgent: agentA,
      rightAgent: agentB,
      url: repoUrl || "",
      leftDir,
      rightDir,
      fullPrompt,
      leftState,
      rightState,
      leftDiff: null,
      rightDiff: null,
      leftRounds: [],
      rightRounds: [],
    });

    // Background: when each agent finishes, capture its diff and build its
    // initial round immediately so follow-ups can be sent independently.
    leftState.promise.then(() => {
      const b = battles.get(battleId);
      if (!b) return;
      b.leftDiff = captureDiff(leftDir);
      b.leftRounds = [{
        prompt: fullPrompt,
        stdout: leftState.stdout || leftState.stderr || "",
        stderr: leftState.stderr || "",
        diff: b.leftDiff || "",
      }];
    }).catch((err) => {
      console.error(`Left agent post-process error: ${err.message}`);
      const b = battles.get(battleId);
      if (!b) return;
      b.leftRounds = [{
        prompt: fullPrompt,
        stdout: leftState.stdout || leftState.stderr || "",
        stderr: leftState.stderr || "",
        diff: "",
      }];
    });
    rightState.promise.then(() => {
      const b = battles.get(battleId);
      if (!b) return;
      b.rightDiff = captureDiff(rightDir);
      b.rightRounds = [{
        prompt: fullPrompt,
        stdout: rightState.stdout || rightState.stderr || "",
        stderr: rightState.stderr || "",
        diff: b.rightDiff || "",
      }];
    }).catch((err) => {
      console.error(`Right agent post-process error: ${err.message}`);
      const b = battles.get(battleId);
      if (!b) return;
      b.rightRounds = [{
        prompt: fullPrompt,
        stdout: rightState.stdout || rightState.stderr || "",
        stderr: rightState.stderr || "",
        diff: "",
      }];
    });

    // Return immediately — frontend will poll /api/battle/status
    res.json({ battleId });
  } catch (err) {
    rmSync(leftDir, { recursive: true, force: true });
    rmSync(rightDir, { recursive: true, force: true });
    console.error(`Battle start error: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// Poll for live agent output
app.get("/api/battle/status/:id", (req, res) => {
  const battle = battles.get(req.params.id);
  if (!battle) {
    return res.status(404).json({ error: "Battle not found (session expired)." });
  }

  const { leftState, rightState } = battle;

  const formatOutput = (state) => {
    const out = state.done ? parseAgentOutput(state.stdout) : state.stdout;
    if (state.done && !state.ok) {
      const prefix = out ? out + "\n\n" : "";
      return `${prefix}**Agent error:** ${state.stderr}`;
    }
    // Agent exited 0 but stderr has warnings/errors — append them
    if (state.done && state.stderr) {
      return `${out}\n\n**Agent warnings:** ${state.stderr}`;
    }
    return out;
  };

  res.json({
    leftStatus: leftState.done ? "done" : "running",
    rightStatus: rightState.done ? "done" : "running",
    leftOutput: formatOutput(leftState),
    rightOutput: formatOutput(rightState),
    leftDiff: battle.leftDiff,
    rightDiff: battle.rightDiff,
  });
});

app.post("/api/battle/followup", async (req, res) => {
  const { battleId, side, prompt } = req.body;
  const battle = battles.get(battleId);
  if (!battle)
    return res.status(404).json({ error: "Battle not found (session expired)." });
  if (!prompt || !prompt.trim())
    return res.status(400).json({ error: "Prompt is required." });
  if (side !== "left" && side !== "right")
    return res.status(400).json({ error: 'Side must be "left" or "right".' });

  const state = side === "left" ? battle.leftState : battle.rightState;
  if (!state.done)
    return res.status(400).json({ error: "Agent is still running. Please wait for it to finish." });

  const agent = side === "left" ? battle.leftAgent : battle.rightAgent;
  const agentDir = side === "left" ? battle.leftDir : battle.rightDir;
  const rounds = side === "left" ? battle.leftRounds : battle.rightRounds;

  try {
    const result = await runFollowup(agent, prompt, agentDir, rounds);
    const diff = captureDiff(agentDir);

    rounds.push({
      prompt,
      stdout: result.stdout || result.stderr || "",
      stderr: result.stderr || "",
      diff,
    });

    res.json({
      output: result.ok
        ? parseAgentOutput(result.stdout)
        : `**Agent error:** ${result.stderr}`,
      diff,
      ok: result.ok,
    });
  } catch (err) {
    console.error(`Followup error: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/battle/vote", async (req, res) => {
  const { battleId, winner } = req.body;
  const battle = battles.get(battleId);
  if (!battle)
    return res.status(404).json({ error: "Battle not found (session expired)." });

  const validWinners = ["left", "right", "tie", "both_bad"];
  if (!validWinners.includes(winner))
    return res.status(400).json({ error: "Invalid winner value." });

  const token = process.env.HF_TOKEN;
  const timestamp = new Date()
    .toISOString()
    .replace(/[-:T]/g, (c) => (c === "T" ? "_" : ""))
    .replace(/\.\d+Z$/, "");
  const fileName = `${LEADERBOARD_FILE}/${timestamp}`;

  const voteEntry = {
    left: battle.left,
    right: battle.right,
    winner,
    timestamp,
  };

  // Strip context from first round prompts before saving
  const leftRoundsClean = battle.leftRounds.map((r, i) => ({
    ...r,
    prompt: i === 0 ? stripContext(r.prompt) : r.prompt,
  }));
  const rightRoundsClean = battle.rightRounds.map((r, i) => ({
    ...r,
    prompt: i === 0 ? stripContext(r.prompt) : r.prompt,
  }));

  const convData = {
    left: battle.left,
    right: battle.right,
    url: battle.url,
    left_rounds: leftRoundsClean,
    right_rounds: rightRoundsClean,
    winner,
    timestamp,
  };

  // Save to HF (fire and forget)
  try {
    await Promise.all([
      saveContentToHf(voteEntry, VOTE_REPO, fileName, token),
      saveContentToHf(convData, CONVERSATION_REPO, fileName, token),
    ]);
  } catch (err) {
    console.error(`HF upload error: ${err.message}`);
  }

  // Clean up
  rmSync(battle.leftDir, { recursive: true, force: true });
  rmSync(battle.rightDir, { recursive: true, force: true });
  battles.delete(battleId);

  // Recompute leaderboard
  try {
    const leaderboard = await getLeaderboardData({
      voteEntry,
      convEntry: convData,
      useCache: false,
    });
    res.json({ leaderboard, agentA: battle.left, agentB: battle.right });
  } catch (err) {
    console.error(`Leaderboard recompute error: ${err.message}`);
    res.json({ leaderboard: [], agentA: battle.left, agentB: battle.right });
  }
});

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

process.on("uncaughtException", (err) => {
  console.error("Uncaught exception:", err);
});
process.on("unhandledRejection", (reason) => {
  console.error("Unhandled rejection:", reason);
});

const PORT = process.env.PORT || 7860;

(async () => {
  // Load agent CLI metadata from HF before accepting requests
  try {
    await loadAgentsFromHf();
  } catch (err) {
    console.error(`Failed to load agents from HF: ${err.message}`);
    process.exit(1);
  }

  const available = availableAgents();
  console.log(
    `Available agents: ${available.map((a) => a.name).join(", ") || "(none)"}`
  );

  // Preload leaderboard
  try {
    const data = await getLeaderboardData({ useCache: true });
    console.log(`Leaderboard preloaded: ${data.length} entries.`);
  } catch (err) {
    console.error(`Failed to preload leaderboard: ${err.message}`);
  }

  const server = app.listen(PORT, () => {
    console.log(`SWE-Agent-Arena running on http://localhost:${PORT}`);
  });
  server.on("error", (err) => {
    console.error("Server error:", err);
  });
})();

import "dotenv/config";
import { uploadFile } from "@huggingface/hub";

const repo = { type: "dataset", name: "SWE-Arena/cli_data" };
const credentials = { accessToken: process.env.HF_TOKEN };

const kimi = {
  website: "https://www.kimi.com/code",
  provider: "Moonshot AI",
  bin: "kimi",
  promptStyle: "flag",
  initArgs: ["--output-format", "stream-json"],
  followupStyle: "continue",
  followupArgs: ["--continue", "--output-format", "stream-json"],
  responseStartMarker: "",
  responseEndMarker: "\n\n**Agent warnings:**",
  state: "active",
};

await uploadFile({
  repo,
  credentials,
  file: {
    path: "Kimi Code.json",
    content: new Blob([JSON.stringify(kimi, null, 2)], { type: "application/json" }),
  },
  commitTitle: "fix: kimi binary name is 'kimi' not 'kimi-code'",
});

console.log("Done: Kimi Code.json updated");

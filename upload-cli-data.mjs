import "dotenv/config";
import { uploadFile } from "@huggingface/hub";

const repo = { type: "dataset", name: "SWE-Arena/cli_data" };
const credentials = { accessToken: process.env.HF_TOKEN };

const files = {
  "Kimi Code.json": {
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
  },
  "Cursor.json": {
    website: "https://cursor.com",
    provider: "Cursor",
    bin: "agent",
    promptStyle: "flag",
    initArgs: ["--trust", "--output-format", "text"],
    followupStyle: "replay",
    followupArgs: ["--trust", "--output-format", "text"],
    responseStartMarker: "",
    responseEndMarker: "",
    state: "active",
  },
};

for (const [filename, data] of Object.entries(files)) {
  console.log(`Uploading ${filename}...`);
  await uploadFile({
    repo,
    credentials,
    file: {
      path: filename,
      content: new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    },
    commitTitle: `fix: update ${filename}`,
  });
  console.log(`  Done: ${filename}`);
}
console.log("All uploads complete.");

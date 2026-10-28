import type { BashInput, BashOutput } from "@anthropic-ai/claude-agent-sdk/sdk-tools.js";
import type { ToolCallContent, ToolCallLocation } from "@agentclientprotocol/sdk/unstable";
import type { BetaBashCodeExecutionResultBlock } from "@anthropic-ai/sdk/resources/beta.mjs";
import { exactHunksContent, type FileChange, type PatchHunk } from "../../diff.js";
import type { ClientCapabilities } from "../client-capabilities.js";
import { structuredResult, textContent, toAcpContentUpdate } from "../content.js";
import type { ToolReporter, ToolResultContext, ToolResultFacts, ToolUseFacts } from "../facts.js";

/**
 * The files that a Bash command changed in the git working tree, which Claude
 * Code reports when `bashEditDiffEnabled` (or `CLAUDE_CODE_BASH_EDIT_DIFF`) is
 * on. The SDK types leave this field out, because it is internal to Claude
 * Code. For example, `sed -i s/beta/BETA/ f.txt` gives
 * `{ files: [{ filePath: "/repo/f.txt", hunks: [...] }], moreFiles: 0, changedFiles: ["/repo/f.txt"] }`.
 */
interface BashEditDiff {
  /** At most 5 files. A binary file or a file too large to show has no hunks. */
  files: { filePath: string; hunks: PatchHunk[]; created?: boolean; deleted?: boolean }[];
  /** The changed files that `files` leaves out. */
  moreFiles: number;
  /** Every changed file that Claude Code knows, at most 200 absolute paths. */
  changedFiles?: string[];
  /** Claude Code could not tell every change of the command. */
  unavailable?: boolean;
  /** Claude Code does not diff this command, such as a `git checkout`. */
  skipped?: boolean;
}

/**
 * Reads a Bash or PowerShell input's description the way the CLI validates it.
 * Since CLI 2.1.295 the CLI accepts `command_description` for `description`.
 * The streamed tool use keeps the original key. The canonical key wins when
 * both spellings are present.
 */
export function shellDescription(input: unknown): string | undefined {
  if (!input || typeof input !== "object") return undefined;
  const raw = input as Record<string, unknown>;
  if (typeof raw.description === "string") return raw.description;
  return typeof raw.command_description === "string" ? raw.command_description : undefined;
}

/** Bash and PowerShell: a command whose output goes to the terminal channel. */
export class BashReporter implements ToolReporter {
  toolUse(input: unknown): ToolUseFacts {
    const bash = input as BashInput | undefined;
    const description = shellDescription(input);
    return {
      title: bash?.command ? bash.command : "Terminal",
      kind: "execute",
      command: true,
      ...(description ? { display: [textContent(description)] } : {}),
    };
  }

  /** A client with a terminal reads a failed command from the terminal channel. */
  errorResult(context: ToolResultContext): ToolResultFacts | undefined {
    return context.capabilities.terminalOutput ? this.toolResult(context) : undefined;
  }

  toolResult({ result, structured, capabilities }: ToolResultContext): ToolResultFacts {
    const content = result.content;
    const isError = result.is_error === true;
    // The output comes from one of these forms:
    // 1. The structured BashOutput (message-level tool_use_result): its
    //    stdout/stderr exclude the model-directed suffixes the raw text
    //    carries (stale-read hints, gh rate-limit hints, the persisted-output
    //    wrapper for too-large outputs). The interruption and truncation facts
    //    are re-established from the structured flags below. Skipped for image
    //    output (the raw content array carries the image blocks) and
    //    backgrounded commands (the raw text carries the background notice).
    // 2. BetaBashCodeExecutionResultBlock.
    // 3. Plain string content from a regular tool_result.
    // 4. Array content: text blocks for stdout, or image blocks when the
    //    command produces an image.
    let output = "";
    // Claude Code accepts a non-zero code as a success only with a
    // `returnCodeInterpretation` (grep's "No matches found"), so a success
    // without one exited with 0.
    let exitCode: number | undefined = isError ? undefined : 0;
    let interrupted = false;

    const structuredBash = structuredResult<BashOutput & { bashEditDiff?: unknown }>(structured);
    if (
      structuredBash &&
      typeof structuredBash.stdout === "string" &&
      typeof structuredBash.stderr === "string" &&
      !structuredBash.isImage &&
      structuredBash.backgroundTaskId === undefined
    ) {
      output = [structuredBash.stdout, structuredBash.stderr].filter(Boolean).join("\n");
      // The CLI appends its abort marker only to the model-facing text. The
      // result does not carry the code of the aborted command.
      if (structuredBash.interrupted) {
        output = [output, "[Command was aborted before completion]"].filter(Boolean).join("\n");
        exitCode = undefined;
        interrupted = true;
      }
      if (structuredBash.returnCodeInterpretation !== undefined) exitCode = undefined;
      // Structured stdout is clipped when the full output was persisted to
      // disk. Without this note the clip is silent.
      if (typeof structuredBash.persistedOutputPath === "string") {
        const size =
          typeof structuredBash.persistedOutputSize === "number"
            ? ` (${structuredBash.persistedOutputSize} bytes total)`
            : "";
        output = [
          output,
          `[Output truncated${size}: full output saved to ${structuredBash.persistedOutputPath}]`,
        ]
          .filter(Boolean)
          .join("\n");
      }
    } else if (
      content &&
      typeof content === "object" &&
      "type" in content &&
      content.type === "bash_code_execution_result"
    ) {
      const bashResult = content as BetaBashCodeExecutionResultBlock;
      output = [bashResult.stdout, bashResult.stderr].filter(Boolean).join("\n");
      exitCode = bashResult.return_code;
    } else if (typeof content === "string") {
      output = content;
    } else if (Array.isArray(content) && content.length > 0) {
      const textOnly = content.every(
        (c: any) => c && typeof c === "object" && typeof c.text === "string",
      );
      if (!textOnly) {
        // Binary payloads cannot travel in the terminal channel, so image
        // output is a result to show.
        return toAcpContentUpdate(content, isError);
      }
      output = content.map((c: any) => c.text).join("\n");
    }
    if (isError && exitCode === undefined) exitCode = failureExitCode(output);
    // A backgrounded command is still running: the result only announces it.
    if (structuredBash?.backgroundTaskId !== undefined) exitCode = undefined;
    return {
      command: {
        output,
        ...(exitCode !== undefined ? { exitCode } : {}),
        ...(interrupted ? { interrupted } : {}),
      },
      ...editDiffFacts(structuredBash?.bashEditDiff, capabilities),
    };
  }
}

/** The diffs and the locations of the files that a Bash command changed. */
function editDiffFacts(
  value: unknown,
  capabilities: ClientCapabilities,
): Pick<ToolResultFacts, "content" | "locations"> {
  const diff = parseBashEditDiff(value);
  if (!diff || diff.skipped || (diff.files.length === 0 && diff.moreFiles === 0)) return {};
  const form = capabilities.v2 ? "v2" : capabilities.diffPatch ? "air" : undefined;
  const content: ToolCallContent[] = [];
  const locations: ToolCallLocation[] = [];
  for (const file of diff.files) {
    const change: FileChange = file.created ? "create" : file.deleted ? "delete" : "update";
    // A file without hunks, such as a binary file, gets only its location.
    // A deleted file has no new line to point at.
    if (file.hunks.length === 0 || change === "delete") {
      locations.push({ path: file.filePath });
    } else {
      locations.push(
        ...file.hunks.map(({ newStart }) => ({ path: file.filePath, line: newStart })),
      );
    }
    if (file.hunks.length > 0) {
      content.push(...exactHunksContent(file.filePath, change, file.hunks, form));
    }
  }
  const shown = new Set(diff.files.map((file) => file.filePath));
  for (const path of diff.changedFiles ?? []) {
    if (!shown.has(path)) locations.push({ path });
  }
  const note = changeNote(diff);
  if (note) content.push(textContent(note));
  return {
    ...(content.length > 0 ? { content } : {}),
    ...(locations.length > 0 ? { locations } : {}),
  };
}

/**
 * The note on the changes that have no diff, such as `2 more files changed`.
 * Returns undefined when every change has a diff.
 */
function changeNote({ files, moreFiles, unavailable }: BashEditDiff): string | undefined {
  if (moreFiles === 0) return unavailable ? "Part of the diff is unavailable" : undefined;
  const more = files.length > 0 ? " more" : "";
  const note = `${moreFiles}${more} ${moreFiles === 1 ? "file" : "files"} changed`;
  return unavailable ? `${note} (part of the diff is unavailable)` : note;
}

/** The `bashEditDiff` of a Bash output, or undefined when it has another shape. */
function parseBashEditDiff(value: unknown): BashEditDiff | undefined {
  if (!value || typeof value !== "object") return undefined;
  const diff = value as Partial<BashEditDiff>;
  if (!Array.isArray(diff.files) || typeof diff.moreFiles !== "number") return undefined;
  const wellFormed = diff.files.every(
    (file) =>
      typeof file?.filePath === "string" &&
      Array.isArray(file.hunks) &&
      file.hunks.every((hunk) => Array.isArray(hunk?.lines)),
  );
  if (!wellFormed) return undefined;
  const changedFiles = Array.isArray(diff.changedFiles)
    ? diff.changedFiles.filter((path): path is string => typeof path === "string")
    : undefined;
  return { ...(diff as BashEditDiff), changedFiles };
}

/**
 * The exit code that the text of a failed command names. Claude Code starts the
 * text of a command that exited with a failing code with `Exit code N`. Other
 * failures, such as a denial or a command that could not start, name none.
 */
function failureExitCode(text: string): number | undefined {
  const match = /^Exit code (\d+)(?:\n|$)/.exec(text);
  return match ? Number(match[1]) : undefined;
}

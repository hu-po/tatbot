import type { InkLangIssue, InkLangIssueCode } from "./types.ts";

export class InkLangError extends Error {
  readonly code: InkLangIssueCode;
  readonly field?: string;

  constructor(code: InkLangIssueCode, message: string, field?: string) {
    super(message);
    this.name = "InkLangError";
    this.code = code;
    this.field = field;
  }

  issue(): InkLangIssue {
    return { code: this.code, message: this.message, ...(this.field ? { field: this.field } : {}) };
  }
}

export function asIssue(error: unknown): InkLangIssue {
  if (error instanceof InkLangError) return error.issue();
  return {
    code: "INKLANG_PARSE_SYNTAX",
    message: error instanceof Error ? error.message : String(error),
  };
}

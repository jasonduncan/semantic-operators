# Command line and MCP

`semop` answers typed questions from the shell, and `semop mcp` serves the same thing to
agents over the Model Context Protocol. Both use the same JSON request and response.

```sh
uv tool install "semantic-operators[typesafe,mcp]"     # installs the semop command
```

## `semop ask`

```sh
semop ask --request request.json                    # TypeSafe, the default
semop ask --provider laya --timeout 120 < request.json
echo '{"state": "...", "questions": [...]}' | semop ask --pretty
```

| Option | Meaning |
|---|---|
| `--provider typesafe\|laya` | Which provider answers. Default: `typesafe`. |
| `--model NAME` | Defaults: `jev-latest` (TypeSafe), `convaiinnovations/laya` (Laya). |
| `--timeout SECONDS` | Time limit per call. For Laya, the first call includes loading the model. |
| `--request FILE` | The request; default `-` reads stdin. |
| `--pretty` | Indent the JSON output. |

The TypeSafe provider reads `TYPESAFE_API_KEY` from the environment; without it, `semop`
says so and exits with code 2. Laya downloads its model (about 800 MB) on first use.

Exit codes: **0** answered, including "don't know" answers; **1** the provider failed or
timed out; **2** the request or command line was invalid.

## The request

```json
{
  "schema_version": "1",
  "request_id": "ticket-42",
  "state": "The payment failed and now I cannot sign in.",
  "questions": [
    {"name": "is_complaint", "type": "boolean",
     "instructions": "Is the sender reporting a problem?", "min_confidence": 0.8,
     "true": "They report a problem or are unhappy.", "false": "They don't."},
    {"name": "department", "type": "choice", "instructions": "Which department?",
     "options": [{"name": "billing", "description": "Charges and refunds"}, "technical"],
     "min_confidence": 0.8},
    {"name": "urgency", "type": "score", "instructions": "How urgent is this?",
     "levels": ["low", "medium", "high"]}
  ]
}
```

| Field | |
|---|---|
| `state` | Required. Text, a JSON object, or an array: what the questions are about. |
| `questions` | Required, non-empty. Names must be unique. |
| `type` | `boolean`, `choice`, or `score`. |
| `instructions` | Required. The question. |
| `options` | Choice only, at least 2: a name, or `{"name", "description"}`. |
| `levels` | Score only, at least 2, lowest first. |
| `true`, `false` | Boolean only, optional: what each outcome means. |
| `min_confidence` | Optional, 0 to 1. Below it the answer is "don't know". |
| `request_id` | Optional, echoed back. |
| `schema_version` | Optional; if given, must be `"1"`. |

Questions and options are arrays so their order is explicit. Unknown fields are
rejected, so a typo fails loudly, and errors say where: `questions[1].options: ...`.

## The response

```json
{
  "request_id": "ticket-42",
  "answers": {
    "is_complaint": {"type": "boolean", "value": true, "decided": true,
                     "confidence": 0.97, "probabilities": {"true": 0.97, "false": 0.03}},
    "department": {"type": "choice", "value": null, "decided": false,
                   "confidence": 0.54, "probabilities": {"billing": 0.46, "technical": 0.54}},
    "urgency": {"type": "score", "value": 1.85, "decided": true,
                "confidence": 0.85, "probabilities": {"low": 0.0, "medium": 0.15, "high": 0.85}}
  },
  "call": {"provider": "TypeSafe", "model": "jev-1.13.0", "input_tokens": 386, "output_tokens": 65}
}
```

- `value` is `null` and `decided` is `false` when the model wasn't sure: below
  `min_confidence`, or an exact tie. That's an answer, not an error.
- A Score's `value` is the expected level index (1.85 of 0..2), not a level name.
- `probabilities` follow the question's option or level order. Numbers are rounded to
  4 significant digits.
- `call` is what the provider reported: the model that actually answered (which can
  differ from an alias like `jev-latest`) and tokens used. Unreported values are `null`.

A failure is `{"error": {"code", "message"}}`, with code `invalid_request`,
`provider_error`, or `timeout`.

## `semop mcp`

A stdio MCP server with one tool, `ask`, taking the request above and returning the
response above. Register it with Claude Code:

```sh
claude mcp add semop -e TYPESAFE_API_KEY="$TYPESAFE_API_KEY" -- semop mcp
claude mcp add semop-local -- semop mcp --provider laya --timeout 120
```

- **The provider and model are fixed when the server starts.** Nothing in a tool call
  can change them, so an agent can't switch a local-only server to a paid hosted API.
  Run two servers if you want both.
- **The tool description says** where answers come from, and whether the data leaves
  the machine and may be billed.
- A bad request, a provider failure, or a timeout comes back as a tool error with the
  same `{"error": ...}` body. A "don't know" is a normal result.
- Laya loads on the first call (a few seconds) and is reused after that.

## Guidance for agents

This is also in the tool description:

- Use it for bounded judgments (classify, route, flag, score), not open-ended
  reasoning or permissions. It only answers; it doesn't act.
- Ask several questions about the same state in one call: it costs about the same as one.
- `decided: false` means unresolved. It isn't `false` or "no", and don't swap in the
  most likely option yourself.
- Don't reword a question or lower `min_confidence` just to get the answer you want.
- Confidence is the model's own view, not a guarantee.

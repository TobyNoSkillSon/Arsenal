<!-- SPDX-License-Identifier: MIT -->
# Model READMEs

Every model folder carries `README.md`: the one document that says what the model is for and how to call it through Arsenal. Agents read it with `arsenal_model(model:ID, docs:true)`; the Get more models window shows it under Docs. Write it for an agent choosing and calling a model, then for a person.

Facts come from the upstream model card, repository or paper and from this folder (`model.toml`, `ops/*.json`, the kind's `Resources/kinds/<kind>/ops/*.json`, `settings.toml`, `log/`). A fact not confirmed in one of them is marked *(unverified)*. Field names, bounds, defaults and formats are copied from the operation schemas, never paraphrased.

## Sections, in order

```markdown
# <Name from model.toml>

<One paragraph: what the model is, who made it, its architecture and size in plain words, and the one thing it does best.>

## What it is for

<What the makers optimised it for (speed, quality, languages, a domain), in one or two sentences.>

- <Use case>
- <Use case>

## Inputs and outputs

### `<operation>` (default)

<One line: what this operation does.>

| Field | Type | Required | Notes |
|---|---|---|---|
| `<field>` | <type> | yes/no | <bounds, default, accepted formats, units> |

Returns <files / text / JSON fields, with formats, sizes, sample rates, channels>.

<For models with special inputs, say exactly what the data must look like: sample rate and channels the model expects
and what Arsenal converts; image size it works at internally; how a series, a horizon, a schema or a question set is
written.>

## Run it

MCP: `arsenal_run(model:"<id>", input:{...})` (add `operation:"<op>"` for a non-default operation).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/<id>/ops/<op>" -H 'Content-Type: application/json' -d '{...}'`

<Settings that change requests (seed, steps, voice) and where they are set; how long a typical run takes when measured.>

## Limits

- <What it cannot do, languages it does not cover, input sizes past which quality drops, known failure modes.>

## Licence and credit

<Licence (SPDX) and what it allows; commercial use; conditions. Makers and the citation they ask for.>

## Links

- Model card: <url>
- Code: <url>
- Paper: <url>

## How Arsenal runs it

<Source pin, precisions offered, runner, quality gate and measured figures, research logs: the folder's engineering record.>
```

File inputs are `{"path":"/absolute/path"}` to any file on this Mac. Keep one example call per route; the operation table carries the rest. Leave out a section that has nothing true to say rather than filling it. A model with several operations gets one `###` per operation.

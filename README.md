*This project has been created as part of the 42 curriculum by ayafshar.*

# Call Me Maybe

## Description

Call Me Maybe is a Python function-calling tool. It reads natural-language
prompts and available function definitions, then writes structured JSON function
calls.

The program does not answer the user prompt. For example, for `What is the sum
of 40 and 2?`, it must output the function name and arguments, not `42`.

The generated result contains exactly:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

## Instructions

Install dependencies:

```sh
make install
```

Run with default paths:

```sh
uv run python -m src
```

Run with explicit paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Useful Makefile targets:

```sh
make run
make debug
make clean
make lint
make lint-strict
```

## File Format

Default input files:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
```

Default output file:

```text
data/output/function_calling_results.json
```

The output file is a JSON array. Each object has exactly `prompt`, `name`, and
`parameters`. No extra keys, prose, comments, or trailing commas are written.

## Algorithm

The implementation uses real vocabulary-driven constrained decoding instead of
asking the model to freely write JSON, and instead of pre-guessing answers.

The output JSON has a known shape: `{"prompt": ..., "name": ..., "parameters":
{...}}`. Structural text and the verbatim prompt have exactly one valid value,
so they are inserted directly. Everywhere the content is *not* already known -
the function name, and every parameter value - the model picks the next token
itself, one token at a time, from a masked vocabulary:

1. Load `functions_definition.json` with Pydantic and load the tokenizer's
   vocabulary via the SDK's `get_path_to_vocab_file()` method (falling back to
   the older `get_path_to_vocabulary_json()` name if a given SDK build only
   exposes that one). `load_vocabulary` accepts either an `id -> token text`
   or `token text -> id` mapping, since the two methods return the map in
   opposite directions.
2. Feed the model a natural-language context (available functions + the user
   prompt), then begin emitting the fixed JSON prefix directly.
3. **Function name**: walk all function names' tokenizations in lock-step,
   asking `get_logits_from_input_ids` at each step and keeping only names still
   consistent with what's been generated so far - the model picks among the
   real, finite set of legal names.
4. **Each parameter value**, per its declared type - never guessed from the
   prompt text beforehand:
   - `enum` / `boolean`: same finite-choice walk as the function name. This is
     the only case where the candidate set is closed in advance, because the
     schema itself declares it closed (e.g. a `firmware` field with a fixed
     list of allowed values).
   - `number` / `integer`: at every step, mask the vocabulary down to digit
     tokens (plus a leading `-`, and - for `number` only - one `.`, and the
     closing separator once a digit exists) - the model decides digit-by-digit
     how many digits to write and when to stop. `integer` parameters never
     see a `.` token, so they can't produce a decimal value.
   - `string`: mask the vocabulary down to tokens containing no `"`, `\`, or
     newline, plus the closing quote as an ever-available "stop" option - the
     model writes free text of its own choosing, closes when it decides to.
5. Decode the chosen token ids back to text with the SDK's optional
   `decode()`, falling back to reconstructing text from the vocabulary map if
   `decode()` isn't available.

Every token that would break JSON structure or the schema is masked to
`-inf` before selection, so the output is 100% valid JSON by construction.
There is no prompt-scraping step anywhere in the argument path: the model
always makes the actual token choice for every open-ended value, so the
approach generalizes to prompts and function sets it has never seen, instead
of degrading to word-matching on the specific examples shipped in
`data/input`.

## Design Decisions

The code is split by responsibility:

- `src/__main__.py`: command-line parsing.
- `src/schemas.py`: Pydantic data schemas for input and output data.
- `src/file_io.py`: reads and writes the JSON input/output files.
- `src/vocabulary.py`: loads the `id -> token text` vocabulary map.
- `src/token_io.py`: low-level SDK calls (`encode`/`decode`/`get_logits`).
- `src/literal_decoding.py`: finite-choice walk shared by function-name
  selection and `enum`/`boolean` values.
- `src/number_decoding.py`: free-form digit-by-digit number generation.
- `src/string_decoding.py`: free-form token-by-token string generation.
- `src/value_decoding.py`: re-exports the three value-generation helpers
  above for `function_call_builder`.
- `src/function_call_builder.py`: assembles one full JSON function call.
- `src/model_setup.py`: `Small_LLM_Model` setup and instruction-prompt context.
- `src/pipeline.py`: orchestration and per-prompt error recovery.

Function selection and argument extraction are both done by the LLM choosing
among vocabulary-masked tokens - never by keyword or regex heuristics on the
prompt text. An earlier revision pre-extracted argument candidates from the
prompt with keyword/regex heuristics (e.g. spotting `"vowel"` and offering a
literal `[aeiouAEIOU]` candidate); that logic was overfit to the specific
sample prompts in `data/input` and violated the "don't hardcode against the
provided examples" rule, so it was removed in favor of pure constrained
generation.

## Performance

Each generation step is one `get_logits_from_input_ids` call over a masked
subset of the vocabulary, so cost scales with output length, not with the
number of functions or parameters. A per-value token cap (`MAX_VALUE_TOKENS`)
guarantees termination even if the model never selects a stop token, keeping
the whole batch bounded and within the five-minute target.

## Challenges

Small models often fail at raw JSON generation, so the challenge is not
prompting but preventing invalid output - handled by masking every non-schema
token to `-inf` before each choice.

A subtler challenge is telling the model *when* to stop a free-form number or
string, since JSON has no dedicated token for that. The fix is to always keep
the correct closing token (separator or quote) available as a valid choice
once at least one character of content exists, so the model decides length on
its own instead of a fixed rule.

Review inputs may also change: the decoder reads function names, parameter
types, and enum values dynamically from `functions_definition.json` and never
assumes a fixed function set.

Free-form string generation masks out any token containing a raw `\`, since
an unescaped backslash inside a JSON string is invalid syntax on its own (JSON
only allows `\"`, `\\`, `\n`, and a handful of other fixed escapes). This
keeps every string 100% valid JSON, but as a consequence the model cannot
currently produce string content that itself requires a backslash (for
example a regex like `\d+`). Enumerating a literal candidate for that one
case was considered, but it only works by recognizing the specific sample
prompt and does not generalize - so the honest trade-off is to accept that
gap rather than special-case around it.

## Testing Strategy

Run the required checks:

```sh
make lint
```

This executes:

```sh
flake8 .
mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

Manual testing should include:

- missing or malformed JSON input files
- empty prompts
- special characters in strings
- large numbers
- boolean, string, and number parameters
- functions with multiple parameters
- ambiguous prompts

## Resources

- 42 project subject: `call me maybe`
- Python documentation: `json`, `argparse`, `pathlib`
- Pydantic documentation
- uv documentation
- Qwen model family documentation

The application code interacts with the model only through the public
`llm_sdk.Small_LLM_Model` methods required by the subject. Model runtime
dependencies such as `torch`, `transformers`, and `huggingface-hub` are present
because the provided `llm_sdk` package needs them internally.

AI was used to help scaffold the repository, review the subject requirements,
improve code readability, and generate documentation. The implementation should
still be read, tested, and understood before peer evaluation.

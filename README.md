# Agento-PoC. 

Simple proof of concept for very primitive agent(?) built from tools

## No (TUI) input, only model's decision.

Sessions aimed at interacting with the user don't exist.
The loop goes only to call tools. Once model decides that it's enough to call functions, it exits.

Examples of tested models:

* `Qwen3.5-35B-A3B-UD-Q4_K_M.gguf` and `Qwen3.5-27B-IQ4_NL.gguf` are stubborn. They will not give up until they can.
* `Nemotron-Cascade-2-30B-A3B.i1-Q4_K_S.gguf` will give up once running `cargo check` fails enough.
* `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` is not that good. Can't start calling tools. Maybe llama.cpp is not documented or tested quants are low quality

From these three models Qwen3.5-27B feels smartest but slowest.
For example it helped implemented Suffix Context management. 

`Qwen3.5-35B-A3B-UD-Q4_K_M.gguf` 35B is fine for smaller tasks thanks to its speed.
Both qwens are run with context size 98304

## Prompt file
To start using the tool call it using path to prompt file: `python main.py prompt.md`

Files can contain extra commands at the start of the line (@ must be the first character):

* "@read intro.md" the line is replaced with the content of the file "intro.md". File "intro.md" is also expanded(i.e. it can include other files)
* "@project_dir /home/user/src/project-directory/" sets the project directory which will be used for tools (if tool expects path, internally project directory is prepended to the path)
* "@#" the commentary (line ignored)
* "@done" stops parsing commands from this point and all lines starting with `@` are appended as usual lines
* "@eof" stops parsing the whole file and returns from the file(purpose to make quick and dirty intermid goal, then return to something more important)

Example can be seen in ./tests/manual_test_fork.md, ./tests/manual_test_editor.md
(To launch in I symlinked `main.py` into `~/bin/agento`)

## Context mode 

* Context-mode. Currently file operation supports three kinds of operation: 

    * Raw, if simple as it can -- when model ask to read the file, it is being read. If model asks it to be read again, it is being read again, then in messages there are 2 versions of the file.
    * Prefix mode. First message from user is "file context". It contains content of files being operated on so far. If the same file read twice, the context will contains only one, latest version of the content. It often confuses qwen-35B. Not recommended.
    * Suffix model. Rewrites old versions of the file. For example if tool read `foo.txt`, then decided to edit it, first read will be removed and in edit answer whole file will be output.

## Tools

The tools(for now) intentionally lack generic `bash`. All available shell commands (like `git add`, `cargo test` are passed as separate tools.).
Don't confuse this is with a sandbox: technically model can create a `#[test]` that goes to `$HOME` and destroys everything.
Practically it's anti-stupidity, not anti anti-attack.

* Note, one of the tool is rust-api-helper, that exists in my other git repository. It basically prints API of everything inside `.rs` files.
Like `cargo docs`, but for private methods too. (No treesitter support)

* One of the tool is `fork`. It forks(like `man 2 fork`) llm context, ask it to do an operation nicely and returns the result to the caller.
It is not called on its own by models and shows no good result.

### Folding.

PoC. Folding tool allows model to hide lines from its context.

## Implementation note

* Class LLM keeps calling tools as long as model needs them to be called and print result to the terminal
* Class LLM contains a list of tools. To define a tool, OpenAI syntax explicitly is not used: LLM generates it based on provided tool name, description and parameters information. Parameters information is stored in annotated argument info. E.g.

```python
    def __call__(self, args: Annotated[list, "arguments that go after `cargo add`"]):
        return run_cargo("add", args)
```

Here `list` means argument is a list(array). Documentation from the second argument of `Annotated` will be sent to LLM.

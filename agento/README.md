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
* "@lang rust" forces project to use rust tools
* "@import_tools foobar.py" from project directory reads foobar.py, calls `import_tools(external_tools: list[Tool])` so module can append tools
* "@#" the commentary (line ignored)
* "@done" stops parsing commands from this point and all lines starting with `@` are appended as usual lines
* "@eof" stops parsing the whole file and returns from the file(purpose to make quick and dirty intermid goal, then return to something more important)

Example can be seen in ./tests/manual_test_fork.md, ./tests/manual_test_editor.md
(To launch in I symlinked `main.py` into `~/bin/agento`)

## Notification file
You can send a notification to LLM: create file `/run/user/$(id -u)/agento.notification`. If this file is not empty, its content will be sent as external notification to LLM on next tool call. File will be deleted.

## Tools

Goal is to be yet another YOLO. Most initial tools were removed.

From special one left only is `fork` to run around, but it's disabled.

## Implementation note

* Class LLM keeps calling tools as long as model needs them to be called and print result to the terminal
* Class LLM contains a list of tools. To define a tool, OpenAI syntax explicitly is not used: LLM generates it based on provided tool name, description and parameters information. Parameters information is stored in annotated argument info. E.g.

```python
    def __call__(self, args: Annotated[list, "arguments that go after `cargo add`"]):
        return run_cargo("add", args)
```

Here `list` means argument is a list(array). Documentation from the second argument of `Annotated` will be sent to LLM.

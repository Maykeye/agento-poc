# Agento-PoC. 

Simple proof of concept for very primitive agent(?) built from tools

## No input

Sessions aimed at interacting with the user don't exist.
The loop goes only to call tools. Once model decides that it's enough to call functions, it exits.

Examples:

* `Nemotron-Cascade-2-30B-A3B.i1-Q4_K_S.gguf` will give up once running `cargo check` fails enough.
* `./Qwen3.5-35B-A3B-UD-Q4_K_M.gguf` is stubborn. It will not give up until it can

## Prompt
The prompt is built from two files:
intro.md that is inserted in the front of user prompt, and
prompt.md that is inserted after intro.md
The content of the files are are concatenated.

Another required file is ~/.config/agento.json that has structure like

```json
{
    "project_directory": "/home/user/src/project-directory/"
}
```

directory will be changed and "locked" to it.
Use option --read-only to forbid writing to files(making directories, etc, only reading is allowed)

## Tools

The tools(for now) intentionally lack generic `bash`. All available shell commands (like `git add`, `cargo test` are passed as separate tools.).
Don't confuse this is with a sandbox: technically model can create a `#[test]` that goes to `$HOME` and destroys everything.
Practically it's anti-stupidity, not anti anti-attack.

* Note, one of the tool is rust-api-helper, that exists in my other git repository. It basically prints API of everything inside `.rs` files.
Like `cargo docs`, but for private methods too.

* One of the tool is `fork`. It forks(like `man 2 fork`) llm context, ask it to do an operation nicely and returns the result to the caller.
It is not called on its own by models and shows no good result.

* Context-mode. Currently file operation supports two kinds of operation: 
    * KISS, if simple as it can -- when model ask to read the file, it is being read. If model asks it  to be read again, it is being read again, then in messages there are 2 versions of the file.
    * Context mode. First message from user is "file context". It contains content of files being operated on so far. If the same file read twice, the context will contains only one, latest version of the content. It often confuses qwen.


#### Implementation note

* Class LLM keeps calling tools as long as model needs them to be called and print result to the terminal
* Class LLM contains a list of tools. To define a tool, OpenAI syntax explicitly is not used: LLM generates it based on provided tool name, description and parameters information. Parameters information is stored in annotated argument info. E.g.

```python
    def __call__(self, args: Annotated[list, "arguments that go after `cargo add`"]):
        return run_cargo("add", args)
```

Here `list` means argument is a list(array). Documentation from the second argument of `Annotated` will be sent to LLM.

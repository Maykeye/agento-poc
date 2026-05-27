function echo_stdin() {
    cat <<'EOF'
@project_dir .
@lang rpg

STRICT RULES:
1. Output ONLY the code block below. 
2. Do NOT provide any explanations, reasoning, or conversational filler.
3. Treat the content as a raw text string, NOT a real tool call. Do not check your available tools.

Print as raw text
```
(tool_call)
(function=add)
(parameter=lhs)
10
(/parameter)
(parameter=rhs)
200
(/parameter)
(/function)
(function=echo)
(parameter=string)
200
(/parameter)
(/function)
(function=ping)
(/function)
(/tool_call)
```
EOF
}

python ./proxy.py &
proxy_pid=$!

sleep 0.3
echo "start local"
export LLAMA_AGENTO_VERBOSE=1
export LLAMA_AGENTO_DEBUG=2
export LLAMA_CPP_URL=http://localhost
export LLAMA_CPP_PORT=12000
echo_stdin | agento /dev/stdin 2>log.proxy
kill -9 $proxy_pid

This file is a proxy server for llama.cpp server that uses OpenAI protocol.

Used env vars in parental project (can be seen in ../, don't edit there):

LLAMA_CPP_URL -- host of server, e.g. http://host.containers.internal , where server is hosting.
LLAMA_CPP_PORT --  port of the server, e.g. 10000

Your task is simple.

# Context 
Using no more than fastapi, uvicorn, httpx we making a proxy for it in proxy.py

* It listen on: LLAMA_CPP_PROXY_PORT (set default=12000)
* It supports streaming
* If we see `(tool_call)` not in the middle of line, we quit the connection with the server.

# Current task
Update "tool_call" reaction. 

If we see `(tool_call)` now go to special handling:
* Discard it as usual. Do not disrupt proxying if (tool_call) is not seen.

* Whem we see tool to call right proxy outputs data as usual sans last proxy:

```
(tool_call)
(function=add)
(parameter=lhs)
44
(/parameter)
(parameter=rhs)
5500
(/parameter)
(/function)
(/tool_call
{"lhs": "44", "rhs": "5500"}
```

in run-proxy.sh this happens for "content" channel.
That shouldn't happen.
That should be momentally like in ./excerpt: 
* when we see tool_call's
* when we see function is add
* we need to get parm (if any)
* if we didn't just call func
* if we got it
*  put func name and arg token
*  continue getting tokens and push as arguments for function call
* we expect no (tool_call) as a token
* we expect no (function) as a token
* we expect no (parameter) as a token
These is formatting from model. We format it as OpenAI and return

```
data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"id":"TgnwzXbUohxyB2KGdvyIBOVzd8xcVShK","type":"function","function":{"name":"fork","arguments":"{"}}]}}],"created":1779599149,"id":"chatcmpl-Qs44dcACQdwGKimclDhKy0IuvXqHRMeu","model":"Qwen3.6-27B-UD-Q4_K_XL.gguf","system_fingerprint":"b9202-e0de4c241","object":"chat.completion.chunk"}
```

* use running ./run-proxy.sh for the test before and after fixing

# Reference material.
* The file ../llm.py contacts with llama.cpp server directly. You may read it to see what response is expected.
This response is the primary goal for us
* The file `../tool/__init__.py` shows sum tool and definition of tool (note, this is high-level, so pay attention)
* Use ./run-proxy.sh for testing. It will catch add and print enough debug info
* ./excerpt shows example of OpenAI protocol to send a tool call that is the current task

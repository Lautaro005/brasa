"""Suite de conformance (ADR 0008): el daemon local contra los SDKs oficiales de OpenAI y
Anthropic. Requiere `brasa serve` corriendo (por defecto en http://127.0.0.1:8080).

Uso: .venv/bin/python tools/conformance/run.py [--base http://127.0.0.1:8080] [-k filtro]
Sale con código 1 si algún caso falla.
"""

import argparse
import json
import sys
import time
import traceback

import anthropic
import httpx
import openai

BASE = "http://127.0.0.1:8080"
MODEL = "qwen3-4b-q4"
WEATHER_FN = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}
ASK_WEATHER = "What's the weather in Paris right now? Use the get_weather tool."
TOOL_RESULT = '{"city": "Paris", "temperature_c": 18, "condition": "cloudy"}'
HUGE = "palabra " * 20000  # muy por encima del contexto del daemon de prueba

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def oa():
    return openai.OpenAI(base_url=f"{BASE}/v1", api_key="brasa-local", max_retries=0, timeout=600)


def an():
    return anthropic.Anthropic(base_url=BASE, api_key="brasa-local", max_retries=0, timeout=600)


# ---------------------------------------------------------------- OpenAI Chat Completions


@case
def chat_texto():
    r = oa().chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Say hello in one word."}], max_tokens=20, temperature=0
    )
    assert r.object == "chat.completion" and r.choices[0].message.role == "assistant"
    assert r.choices[0].message.content.strip(), "contenido vacío"
    assert r.choices[0].finish_reason == "stop"
    assert r.usage.prompt_tokens > 0 and r.usage.completion_tokens > 0


@case
def chat_stream_con_usage():
    s = oa().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5, separated by spaces."}],
        max_tokens=30,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    text, finish, usage = "", None, None
    for ch in s:
        if ch.usage:
            usage = ch.usage
        for c in ch.choices:
            text += c.delta.content or ""
            finish = c.finish_reason or finish
    assert "1" in text and "5" in text, text
    assert finish == "stop" and usage and usage.completion_tokens > 0


@case
def chat_herramientas_ida_y_vuelta():
    c = oa()
    msgs = [{"role": "user", "content": ASK_WEATHER}]
    tools = [{"type": "function", "function": WEATHER_FN}]
    r = c.chat.completions.create(model=MODEL, messages=msgs, tools=tools, temperature=0, max_tokens=200)
    m = r.choices[0].message
    assert r.choices[0].finish_reason == "tool_calls", r
    call = m.tool_calls[0]
    assert call.type == "function" and call.function.name == "get_weather"
    args = json.loads(call.function.arguments)
    assert "paris" in args["city"].lower()
    msgs += [
        {"role": "assistant", "content": m.content, "tool_calls": [call.model_dump()]},
        {"role": "tool", "tool_call_id": call.id, "content": TOOL_RESULT},
    ]
    r2 = c.chat.completions.create(model=MODEL, messages=msgs, tools=tools, temperature=0, max_tokens=200)
    assert r2.choices[0].finish_reason == "stop"
    assert "18" in r2.choices[0].message.content, r2.choices[0].message.content


@case
def chat_herramientas_stream():
    s = oa().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": ASK_WEATHER}],
        tools=[{"type": "function", "function": WEATHER_FN}],
        temperature=0,
        max_tokens=200,
        stream=True,
    )
    calls, finish = {}, None
    for ch in s:
        for c in ch.choices:
            finish = c.finish_reason or finish
            for tc in c.delta.tool_calls or []:
                d = calls.setdefault(tc.index, {"name": "", "args": ""})
                if tc.function.name:
                    d["name"] = tc.function.name
                d["args"] += tc.function.arguments or ""
    assert finish == "tool_calls" and calls[0]["name"] == "get_weather"
    assert "paris" in json.loads(calls[0]["args"])["city"].lower()


@case
def chat_error_de_contexto():
    try:
        oa().chat.completions.create(model=MODEL, messages=[{"role": "user", "content": HUGE}], max_tokens=5)
    except openai.BadRequestError as e:
        assert e.code == "context_length_exceeded", e
        return
    raise AssertionError("se esperaba BadRequestError")


# ---------------------------------------------------------------- OpenAI Responses


@case
def responses_texto():
    r = oa().responses.create(model=MODEL, input="Say hello in one word.", max_output_tokens=20, temperature=0)
    assert r.object == "response" and r.status == "completed"
    assert r.output_text.strip(), r
    assert r.usage.input_tokens > 0


@case
def responses_stream():
    s = oa().responses.create(
        model=MODEL,
        instructions="You are terse.",
        input=[{"role": "user", "content": "Count from 1 to 5."}],
        max_output_tokens=40,
        temperature=0,
        stream=True,
    )
    types, deltas, final = [], "", None
    for ev in s:
        types.append(ev.type)
        if ev.type == "response.output_text.delta":
            deltas += ev.delta
        if ev.type == "response.completed":
            final = ev.response
    assert types[0] == "response.created" and types[-1] == "response.completed", types
    assert "response.output_item.added" in types and "response.output_item.done" in types
    assert final.output_text == deltas and "5" in deltas


@case
def responses_herramientas_ida_y_vuelta():
    c = oa()
    tools = [{"type": "function", **WEATHER_FN}]
    r = c.responses.create(model=MODEL, input=ASK_WEATHER, tools=tools, temperature=0, max_output_tokens=200)
    calls = [o for o in r.output if o.type == "function_call"]
    assert calls and calls[0].name == "get_weather", r.output
    call = calls[0]
    assert "paris" in json.loads(call.arguments)["city"].lower()
    follow = [
        {"role": "user", "content": ASK_WEATHER},
        {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": call.arguments},
        {"type": "function_call_output", "call_id": call.call_id, "output": TOOL_RESULT},
    ]
    r2 = c.responses.create(model=MODEL, input=follow, tools=tools, temperature=0, max_output_tokens=200)
    assert "18" in r2.output_text, r2.output_text


@case
def responses_herramientas_stream():
    s = oa().responses.create(
        model=MODEL,
        input=ASK_WEATHER,
        tools=[{"type": "function", **WEATHER_FN}],
        temperature=0,
        max_output_tokens=200,
        stream=True,
    )
    args, final = "", None
    for ev in s:
        if ev.type == "response.function_call_arguments.delta":
            args += ev.delta
        if ev.type == "response.completed":
            final = ev.response
    fc = [o for o in final.output if o.type == "function_call"]
    assert fc and fc[0].name == "get_weather" and fc[0].arguments == args


@case
def responses_herramienta_custom():
    # Herramienta de entrada libre, como `apply_patch` de Codex: llega como `custom_tool_call`.
    c = oa()
    tools = [{"type": "custom", "name": "shout",
              "description": "Repeats the given text in uppercase. The input is the raw text, not JSON."}]
    ask = "Use the shout tool with the text: hola brasa"
    s = c.responses.create(model=MODEL, input=ask, tools=tools, temperature=0, max_output_tokens=200, stream=True)
    delta, final = "", None
    for ev in s:
        if ev.type == "response.custom_tool_call_input.delta":
            delta += ev.delta
        if ev.type == "response.completed":
            final = ev.response
    calls = [o for o in final.output if o.type == "custom_tool_call"]
    assert calls and calls[0].name == "shout", final.output
    call = calls[0]
    assert call.input == delta and "hola brasa" in call.input.lower(), (call.input, delta)
    follow = [
        {"role": "user", "content": ask},
        {"type": "custom_tool_call", "call_id": call.call_id, "name": call.name, "input": call.input},
        {"type": "custom_tool_call_output", "call_id": call.call_id, "output": "HOLA BRASA"},
    ]
    r2 = c.responses.create(model=MODEL, input=follow, tools=tools, temperature=0, max_output_tokens=200)
    assert "HOLA BRASA" in r2.output_text.upper(), r2.output_text


@case
def responses_error_de_contexto():
    try:
        oa().responses.create(model=MODEL, input=HUGE, max_output_tokens=5)
    except openai.BadRequestError as e:
        assert e.code == "context_length_exceeded", e
        return
    raise AssertionError("se esperaba BadRequestError")


# ---------------------------------------------------------------- Anthropic Messages
# El SDK vigente no acepta parámetros de muestreo; se pasan por extra_body para tests deterministas.


@case
def anthropic_texto():
    m = an().messages.create(
        model=MODEL, max_tokens=20, extra_body={"temperature": 0}, messages=[{"role": "user", "content": "Say hello in one word."}]
    )
    assert m.type == "message" and m.role == "assistant" and m.stop_reason == "end_turn"
    assert m.content[0].type == "text" and m.content[0].text.strip()
    assert m.usage.output_tokens > 0


@case
def anthropic_stream():
    with an().messages.stream(
        model=MODEL,
        max_tokens=40,
        extra_body={"temperature": 0},
        system="You are terse.",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    ) as s:
        text = "".join(s.text_stream)
        final = s.get_final_message()
    assert "5" in text and final.content[0].text == text and final.stop_reason == "end_turn"


@case
def anthropic_herramientas_ida_y_vuelta():
    c = an()
    tools = [{"name": WEATHER_FN["name"], "description": WEATHER_FN["description"], "input_schema": WEATHER_FN["parameters"]}]
    msgs = [{"role": "user", "content": ASK_WEATHER}]
    m = c.messages.create(model=MODEL, max_tokens=200, extra_body={"temperature": 0}, tools=tools, messages=msgs)
    assert m.stop_reason == "tool_use", m
    tu = [b for b in m.content if b.type == "tool_use"][0]
    assert tu.name == "get_weather" and "paris" in tu.input["city"].lower()
    msgs += [
        {"role": "assistant", "content": [b.model_dump() for b in m.content]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id, "content": TOOL_RESULT}]},
    ]
    m2 = c.messages.create(model=MODEL, max_tokens=200, extra_body={"temperature": 0}, tools=tools, messages=msgs)
    assert m2.stop_reason == "end_turn" and "18" in m2.content[0].text, m2


@case
def anthropic_herramientas_stream():
    tools = [{"name": WEATHER_FN["name"], "description": WEATHER_FN["description"], "input_schema": WEATHER_FN["parameters"]}]
    with an().messages.stream(
        model=MODEL, max_tokens=200, extra_body={"temperature": 0}, tools=tools, messages=[{"role": "user", "content": ASK_WEATHER}]
    ) as s:
        final = s.get_final_message()
    tu = [b for b in final.content if b.type == "tool_use"]
    assert final.stop_reason == "tool_use" and tu and "paris" in tu[0].input["city"].lower(), final


@case
def anthropic_count_tokens():
    r = an().messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": "Hello there"}])
    assert 5 < r.input_tokens < 40, r


@case
def anthropic_error_de_contexto():
    try:
        an().messages.create(model=MODEL, max_tokens=5, messages=[{"role": "user", "content": HUGE}])
    except anthropic.BadRequestError as e:
        assert e.body["error"]["type"] == "invalid_request_error", e.body
        # Claude Code solo compacta ante un error que reconoce por este texto.
        assert e.body["error"]["message"].startswith("prompt is too long"), e.body
        return
    raise AssertionError("se esperaba BadRequestError")


# ---------------------------------------------------------------- cancelación y prefix cache


@case
def cancelacion_libera_el_modelo():
    # Pedido largo en streaming: se lee un poco y se corta. El siguiente pedido tiene que
    # empezar enseguida (el modelo canceló), no después de generar 2000 tokens.
    with httpx.stream(
        "POST",
        f"{BASE}/v1/chat/completions",
        json={"model": MODEL, "stream": True, "max_tokens": 2000, "temperature": 0,
              "messages": [{"role": "user", "content": "Write a very long story about a dragon."}]},
        timeout=600,
    ) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i > 5:
                break
    t = time.time()
    chat_texto()
    dt = time.time() - t
    assert dt < 15, f"el pedido siguiente tardó {dt:.1f} s"


@case
def prefix_cache_entre_turnos():
    c = oa()
    sys_prompt = "You are a helpful assistant. " + "Context line. " * 400
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Say A."}]
    r1 = c.chat.completions.create(model=MODEL, messages=msgs, max_tokens=5, temperature=0)
    msgs += [{"role": "assistant", "content": r1.choices[0].message.content}, {"role": "user", "content": "Say B."}]
    r2 = c.chat.completions.create(model=MODEL, messages=msgs, max_tokens=5, temperature=0)
    cached = r2.usage.prompt_tokens_details.cached_tokens
    assert cached > 0.8 * r1.usage.prompt_tokens, (cached, r1.usage.prompt_tokens)


def main() -> None:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("-k", default="")
    a = ap.parse_args()
    BASE = a.base
    failed = 0
    for fn in CASES:
        if a.k not in fn.__name__:
            continue
        t = time.time()
        try:
            fn()
            print(f"PASS  {fn.__name__:40s} {time.time() - t:6.1f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__:40s} {time.time() - t:6.1f} s  {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=2)
    print(f"\n{len([c for c in CASES if a.k in c.__name__]) - failed} pasan, {failed} fallan "
          f"(openai {openai.__version__}, anthropic {anthropic.__version__})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

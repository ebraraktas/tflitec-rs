# llm-runner

This example shows how to use signature runners in an LLM model.

It is written for `gemma3-1b-it` but it can be modified easily for other models.

Download model resources:

```shell
# download task file and extract model from it, you can download it from web page directly
huggingface-cli download litert-community/Gemma3-1B-IT gemma3-1b-it-int4.task --local-dir .
unzip gemma3-1b-it-int4.task TF_LITE_PREFILL_DECODE && mv TF_LITE_PREFILL_DECODE gemma3-1b-it-int4.tflitee
# download tokenizer.json
huggingface-cli download google/gemma-3-1b-it tokenizer.json --local-dir .
```

and run (note that this currently requires `xnnpack` features):

```shell
cargo run -r -- gemma3-1b-it-int4.tflite tokenizer.json
```
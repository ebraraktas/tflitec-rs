use std::collections::HashMap;
use std::io::Write;
use tflitec::interpreter::Interpreter;
use tflitec::signature_runner::SignatureRunner;

struct Message {
    role: String,
    content: String,
}

/// LLM pipeline built for tflite models.
///
/// Converted from https://colab.research.google.com/github/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#scrollTo=AM6rDABTXt2F
struct LiteRtLlmPipeline<'a> {
    interpreter: tflitec::interpreter::Interpreter<'a>,
    tokenizer: tokenizers::Tokenizer,
}

impl<'a> LiteRtLlmPipeline<'a> {
    /// Gets the mask for the input to the model.
    ///
    /// All the elements in the mask are set to -inf except that all the
    /// elements below the k-th diagonal are set to 0.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the mask input to the model.
    /// * `k`: all elements below the k-th diagonal are set to 0.
    fn get_mask(shape: &[usize], k: usize) -> Vec<f32> {
        let mut mask = vec![f32::NEG_INFINITY; shape.iter().product()];
        let num_rows = shape[shape.len() - 2];
        let num_cols = shape[shape.len() - 1];

        // set first inner matrix then copy
        for i in 0..num_rows {
            for j in 0..num_cols {
                if j < i + k {
                    mask[i * num_cols + j] = 0.0;
                }
            }
        }
        // copy all
        let inner_matrix_size = num_rows * num_cols;
        let (first_matrix, rest) = mask.split_at_mut(inner_matrix_size);
        for i in (0..rest.len()).step_by(inner_matrix_size) {
            rest[i..i + inner_matrix_size].copy_from_slice(first_matrix);
        }
        mask
    }
    fn new(
        interpreter: tflitec::interpreter::Interpreter<'a>,
        tokenizer: tokenizers::Tokenizer,
    ) -> Self {
        Self {
            interpreter,
            tokenizer,
        }
    }

    /// Returns the prefill runner with the best suitable input size
    fn get_prefill_runner<'b>(
        interpreter: &'b Interpreter<'b>,
        num_input_tokens: usize,
    ) -> SignatureRunner<'b> {
        let mut best = None;
        let mut delta = usize::MAX;
        let mut max_prefill_len = 0;
        for signature_index in 0..interpreter.signature_count() {
            let signature_key = interpreter.get_signature_key(signature_index).unwrap();
            if !signature_key.contains("prefill") {
                continue;
            }
            let signature = interpreter.get_signature_runner(signature_key).unwrap();
            signature.allocate_tensors().unwrap();
            // input_pos shape: (max_seq_len, )
            let input_pos = signature.get_input_tensor("input_pos").unwrap();
            let seq_size = input_pos.shape().dimensions()[0];
            max_prefill_len = seq_size.max(max_prefill_len);
            if num_input_tokens < seq_size && seq_size - num_input_tokens < delta {
                best = Some(signature);
                delta = seq_size - num_input_tokens;
            }
        }
        best.unwrap_or_else(|| {
            panic!(
                "The largest prefill length supported is {max_prefill_len}, but we have {num_input_tokens} number of input tokens",
            )
        })
    }

    fn init_kv_cache(&self, prefill_runner: &SignatureRunner) -> HashMap<String, Vec<f32>> {
        (0..prefill_runner.input_count())
            .map(|i| prefill_runner.get_input_name(i))
            .filter(|name| name.contains("kv_cache"))
            .map(|name| {
                let tensor = prefill_runner.get_input_tensor(name).unwrap();
                let numel = tensor.shape().dimensions().iter().product();
                let data = vec![0.0_f32; numel];
                (name.to_string(), data)
            })
            .collect()
    }

    fn run_prefill(
        &self,
        prefill_runner: &SignatureRunner,
        prefill_token_ids: &[u32],
    ) -> HashMap<String, Vec<f32>> {
        // input_token_shape has shape (batch, max_seq_len)
        let max_seq_len = {
            let tokens = prefill_runner.get_input_tensor("tokens").unwrap();
            let shape = tokens.shape().dimensions();
            if shape.len() != 2 { shape[0] } else { shape[1] }
        };
        let prefill_token_length = prefill_token_ids.len();
        if prefill_token_length == 0 {
            return self.init_kv_cache(prefill_runner);
        }

        // Prepare the input
        let input_token_ids = prefill_token_ids
            .iter()
            .map(|x| *x as i32)
            .chain((prefill_token_length..max_seq_len).map(|_| 0))
            .collect::<Vec<i32>>();
        // Prepare the input position to be [max_seq_len].
        let input_pos = (0..(prefill_token_length as i32))
            .chain((prefill_token_length..max_seq_len).map(|_| 0))
            .collect::<Vec<i32>>();

        // Set kv cache.
        let prefill_inputs = self.init_kv_cache(prefill_runner);
        for (tensor_name, tensor_data) in prefill_inputs.into_iter() {
            prefill_runner
                .get_input_tensor(&tensor_name)
                .unwrap()
                .set_data(&tensor_data)
                .unwrap();
        }
        // Prepare the tokens and input position inputs.
        prefill_runner
            .get_input_tensor("tokens")
            .unwrap()
            .set_data(&input_token_ids)
            .unwrap();
        prefill_runner
            .get_input_tensor("input_pos")
            .unwrap()
            .set_data(&input_pos)
            .unwrap();

        if (0..prefill_runner.input_count())
            .any(|i| prefill_runner.get_input_name(i).contains("mask"))
        {
            // For prefill, mask has shape [batch=1, 1, seq_len, kv_cache_size].
            // We want mask[0, 0, i, j] = 0 for j<=i and -inf otherwise.
            let mask = Self::get_mask(
                prefill_runner
                    .get_input_tensor("mask")
                    .unwrap()
                    .shape()
                    .dimensions(),
                1,
            );
            prefill_runner
                .get_input_tensor("mask")
                .unwrap()
                .set_data(&mask)
                .unwrap();
        }

        prefill_runner.invoke().unwrap();
        (0..prefill_runner.output_count())
            .map(|i| prefill_runner.get_output_name(i))
            .map(|name| {
                (
                    name.to_string(),
                    prefill_runner
                        .get_output_tensor(name)
                        .unwrap()
                        .data::<f32>()
                        .to_vec(),
                )
            })
            .collect()
    }

    /// Runs decode and outputs the token ids from greedy sampler.
    ///
    /// # Arguments
    ///
    /// * `start_pos`: The position of the first token of the decode input.
    /// * `start_token_id`: The token id of the first token of the decode input.
    /// * `kv_cache`: The kv cache from the prefill.
    /// * `max_decode_steps`: The max decode steps.
    fn run_decode(
        &self,
        start_pos: usize,
        start_token_id: u32,
        mut kv_cache: HashMap<String, Vec<f32>>,
        max_decode_steps: usize,
    ) -> Vec<u32> {
        let runner = self.interpreter.get_signature_runner("decode").unwrap();
        let eos_id = self
            .tokenizer
            .token_to_id("<eos>")
            .expect("Tokenizer is missing <eos> token.");
        let mut next_pos = start_pos;
        let mut next_token = start_token_id;
        let mut token_ids = Vec::new();
        runner.allocate_tensors().unwrap();
        for _ in 0..max_decode_steps {
            if let Ok(mask_tensor) = runner.get_input_tensor("mask") {
                // For decode, mask has shape [batch=1, 1, 1, kv_cache_size].
                // We want mask[0, 0, 0, j] = 0 for j<=next_pos and -inf otherwise.
                mask_tensor
                    .set_data(&Self::get_mask(
                        mask_tensor.shape().dimensions(),
                        next_pos + 1,
                    ))
                    .unwrap();
            }
            for (name, tensor_data) in kv_cache.iter() {
                runner
                    .get_input_tensor(name)
                    .unwrap()
                    .set_data(tensor_data)
                    .unwrap();
            }
            runner
                .get_input_tensor("tokens")
                .unwrap()
                .set_data(&[next_token as i32])
                .unwrap();
            runner
                .get_input_tensor("input_pos")
                .unwrap()
                .set_data(&[next_pos as i32])
                .unwrap();
            runner.invoke().unwrap();
            let logits = runner
                .get_output_tensor("logits")
                .unwrap()
                .data::<f32>()
                .to_vec();
            next_token = Self::greedy_sampler(&logits) as u32;
            if next_token == eos_id {
                break;
            }
            let decoded = self.tokenizer.decode(&[next_token], true).unwrap();
            // Break out the loop if we hit the special token.
            if decoded.is_empty() {
                break;
            }
            print!("{decoded}");
            std::io::stdout().flush().unwrap();
            token_ids.push(next_token);

            // set new kv cache values
            for name in (0..runner.output_count()).map(|i| runner.get_output_name(i)) {
                if name.contains("kv_cache") {
                    let tensor = runner.get_output_tensor(name).unwrap();
                    kv_cache.insert(name.to_string(), tensor.data::<f32>().to_vec());
                }
            }
            next_pos += 1;
        }

        // print new line at the end
        println!();
        token_ids
    }

    fn greedy_sampler(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|&(_, x), &(_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0
    }

    fn apply_chat_template(messages: &[Message], add_generation_prompt: bool) -> String {
        // TODO: this is written for gemma-3, need to change for other models
        let mut text = String::from("<bos>");
        for (i, message) in messages.iter().enumerate() {
            if i % 2 == 0 && message.role != "user" {
                panic!("Conversation roles must alternate user/assistant/user/assistant/...")
            }
            if message.role != "user" && message.role != "model" {
                panic!("Conversation roles must be user or model");
            }
            text.push_str("<start_of_turn>");
            text.push_str(&message.role);
            text.push('\n');
            text.push_str(message.content.trim_end());
            text.push_str("<end_of_turn>\n");
        }
        if add_generation_prompt {
            text.push_str("<start_of_turn>model\n");
        }
        text
    }

    fn generate_from_messages(
        &self,
        messages: &[Message],
        max_decode_steps: Option<usize>,
    ) -> String {
        let formatted_prompt = Self::apply_chat_template(messages, true);
        let encoding = self.tokenizer.encode_fast(formatted_prompt, false).unwrap();
        // Prefill up to the seond to the last token of the prompt, because the last
        // token of the prompt will be used to bootstrap decode.
        let prefill_token_length = encoding.len() - 1;

        let start = std::time::Instant::now();
        // Initialize the prefill runner with the suitable input size.
        let prefill_runner = Self::get_prefill_runner(&self.interpreter, encoding.len());
        // kv cache input has shape [batch=1, num_kv_heads, cache_size, head_dim].
        let max_kv_cache_seq_len = {
            let kv_cache_k_0 = prefill_runner.get_input_tensor("kv_cache_k_0").unwrap();
            let shape = kv_cache_k_0.shape().dimensions();
            shape[2]
        };
        // Run prefill.
        let kv_cache =
            self.run_prefill(&prefill_runner, &encoding.get_ids()[..prefill_token_length]);
        println!(
            "[INFO] Running prefill took: {:?} ({:.2} tokens/second)",
            start.elapsed(),
            prefill_token_length as f64 / start.elapsed().as_secs_f64()
        );

        // Run decode.
        let mut actual_max_decode_steps = max_kv_cache_seq_len - prefill_token_length - 1;
        if max_decode_steps.is_some() {
            actual_max_decode_steps =
                std::cmp::min(actual_max_decode_steps, max_decode_steps.unwrap());
        }
        let start = std::time::Instant::now();
        let response = self.run_decode(
            prefill_token_length,
            encoding.get_ids()[prefill_token_length],
            kv_cache,
            actual_max_decode_steps,
        );
        println!(
            "[INFO] Running decode took: {:.1?} ({:.2} tokens/second)",
            start.elapsed(),
            (response.len() as f64 / start.elapsed().as_secs_f64())
        );
        self.tokenizer.decode(&response, true).unwrap()
    }

    #[allow(unused)]
    fn generate(&mut self, prompt: &str, max_decode_steps: Option<usize>) -> String {
        let messages = [Message {
            role: String::from("user"),
            content: prompt.to_string(),
        }];
        self.generate_from_messages(&messages, max_decode_steps)
    }
}

fn load_tokenizer(path: &str) -> tokenizers::Tokenizer {
    let start = std::time::Instant::now();
    let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
    println!("[INFO] Loaded tokenizer from {path} in {:.1?}", start.elapsed());
    tokenizer
}

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() < 3 {
        panic!("Usage: llm-runner <path-to-model-tflite> <path-tokenizer-json>")
    }
    let tokenizer = load_tokenizer(&args[2]);
    // load model
    let start = std::time::Instant::now();
    let model = tflitec::model::Model::new(&args[1]).unwrap();
    let interpreter = tflitec::interpreter::Interpreter::new(
        &model,
        Some(tflitec::interpreter::Options {
            thread_count: 1,
            is_xnnpack_enabled: true,
        }),
    )
    .unwrap();
    println!("[INFO] Loaded tflite model from {} in {:.1?}", &args[1], start.elapsed());

    let llm = LiteRtLlmPipeline::new(interpreter, tokenizer);
    let mut input = String::new();
    let mut messages = vec![];
    loop {
        print!("Input (hit Enter or Ctrl-C to exit):");
        std::io::stdout().flush().unwrap();
        // read input from terminal
        input.clear();
        std::io::stdin().read_line(&mut input).unwrap();
        if input.trim().is_empty() {
            break;
        }
        messages.push(Message {
            role: "user".to_string(),
            content: input.trim_end().to_string(),
        });
        let response = llm.generate_from_messages(&messages, None);
        messages.push(Message {
            role: "model".to_string(),
            content: response.trim_end().to_string(),
        });
    }
}

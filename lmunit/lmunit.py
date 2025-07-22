import math
from typing import Any, Dict, List

from loguru import logger
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LMUnit:
    def __init__(self, model_path: str, tp_size: int):
        self.model_path = model_path
        self.tp_size = tp_size
        self.digit_tokens = ["0", "1", "2", "3", "4", "5", "6"]
        self.digit_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.llm = LLM(model=self.model_path, tensor_parallel_size=self.tp_size)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def score_from_top_logprobs(
        self,
        top_logprobs_dicts: Dict[str, float],
        digit_tokens: List[str],
        digit_values: List[float],
    ) -> float:
        """Calculates a weighted score from logprobs of digit tokens.

        Takes a dictionary of token logprobs and calculates a weighted average score,
        where each digit token contributes its value weighted by its probability.

        Args:
            top_logprobs_dicts: Dictionary mapping tokens to their log probabilities
            digit_tokens: List of digit tokens to consider (e.g. ["0", "1", "2"...])
            digit_values: Corresponding values for each digit token (e.g. [0.0, 1.0, 2.0...])

        Returns:
            Weighted average score in range [0.0, max(digit_values)]. Returns 0.0 if no valid tokens found.

        """
        total_prob = 0.0
        weighted_sum = 0.0

        for digit_tok, digit_val in zip(digit_tokens, digit_values):
            if digit_tok in top_logprobs_dicts:
                p = math.exp(top_logprobs_dicts[digit_tok])
                weighted_sum += p * digit_val
                total_prob += p
        return weighted_sum / total_prob if total_prob > 0 else 0.0

    def postprocess(self, data: Dict[str, Any]):
        """Postprocesses model output to extract score and rationale."""
        try:
            output_text = data["content"]
            score_pos = self._find_score_position(data["tokens"])
            if score_pos is None:
                logger.warning("No score position found in output")
                return {
                    "output_text": "Error: No score position found in output",
                    "score": 0,
                }

            score_token = (
                str(int(float(data["tokens"][score_pos])))
                if isinstance(data["tokens"][score_pos], float)
                else str(data["tokens"][score_pos])
            )
            if score_token.strip() not in self.digit_tokens:
                logger.warning(f"Invalid score token: {score_token}")
                return {"output_text": "Error: Invalid score token", "score": 0}
            # Create logprobs dict inline
            top_logprobs_dicts = {
                str(int(token)) if isinstance(token, float) else str(token): logprob
                for token, logprob in zip(
                    data["tokens_logprobs"][score_pos], data["top_logprobs"][score_pos]
                )
            }

            score_val = self.score_from_top_logprobs(
                top_logprobs_dicts, self.digit_tokens, self.digit_values
            )

            # logger.info(f"Calculated score value: {score_val}")
            return {"output_text": output_text, "score": score_val}

        except Exception as e:
            logger.error(f"Error processing score logprobs: {e!s}")
            return {"output_text": "Error: Error processing score logprobs", "score": 0}

    def _find_score_position(self, tokens: List[str]):
        """Finds the position of the score token in the output."""
        score_positions = [
            i + 3
            for i in range(len(tokens) - 1)
            if tokens[i] == "score" and tokens[i + 1] == '":'
        ]
        return score_positions[0] if score_positions else None

    def generate(self, prompt: str, sampling_params: SamplingParams):
        outputs = self.llm.generate(prompt, sampling_params)
        data = []
        for output in outputs:
            sample = {
                "content": output.outputs[0].text,
                "tokens": [
                    self.tokenizer.decode(token)
                    for token in output.outputs[0].token_ids
                ],
                "tokens_logprobs": [
                    [
                        x.decoded_token
                        for x in list(output.outputs[0].logprobs[i].values())
                    ]
                    for i in range(len(output.outputs[0].logprobs))
                ],
                "top_logprobs": [
                    [x.logprob for x in list(output.outputs[0].logprobs[i].values())]
                    for i in range(len(output.outputs[0].logprobs))
                ],
            }
            data.append(self.postprocess(sample))
        return data

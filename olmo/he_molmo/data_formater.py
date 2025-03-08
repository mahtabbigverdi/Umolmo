from typing import Dict, Tuple

from olmo import tokenizer
from olmo.config import BaseConfig
from olmo.data.data_formatter import DataFormatter


class HeDataFormatter(DataFormatter):
    """The HE models the query BEFORE the image, so we have a custom preprocess to do that"""

    def _format_example(self, message, example, is_training, for_inference, rng):
        metadata = {}
        for k in ["answer_idx", "answers", "answer", "points", "options"]:
            if k in message:
                metadata[k] = message[k]

        if isinstance(message, str):
            messages = [message]
        elif isinstance(message, list):
            messages = message
        elif "messages" in message:
            # Example directly contains the prompts/message to use
            messages = message["messages"]
        elif isinstance(message, dict):
            # An example that requires a custom prompt
            prompt, response, extra_metadata = self.get_user_prompt(message, is_training, for_inference=for_inference, rng=rng)
            if extra_metadata:
                metadata.update(extra_metadata)
            if not for_inference:
                assert response is not None
                messages = [prompt, response]
            else:
                messages = [prompt]
        else:
            raise ValueError(f"Example type {type(message)} not understood")

        # Add the system prompt
        if self.system_prompt and self.system_prompt != "none":
            style = None
            if isinstance(message, dict):
                style = message.get("style", None)
            prefix = self.get_system_prompt(style, for_inference, messages, rng=rng)
            if len(prefix) > 0 and len(messages[0]) > 0:
                with_system_prompt = prefix + " " + messages[0]
            elif len(prefix) > 0:
                with_system_prompt = prefix
            else:
                with_system_prompt = messages[0]
            messages = [with_system_prompt] + messages[1:]

        if "image" in example:
            # Image is always at the end of the input message
            messages[0] = messages[0] + tokenizer.IMAGE_PROMPT

        # Add the role annotations such as "User:" and "Assistant:"
        messages = self.format_messages(messages)
        return messages, metadata

    def __call__(self, ex: Dict, is_training, for_inference, rng) -> Tuple[Dict, Dict]:
        """Returns a formatted example and example metadata"""
        if "message_list" in ex and len(ex["message_list"]) > 1:
            raise NotImplementedError("Multiple messages for an image not supported")
        elif "message_list" in ex:
            return self._format_example(ex["message_list"][0], ex, is_training, for_inference, rng)
        elif "messages" in ex:
            return self._format_example(ex["messages"], ex, is_training, for_inference, rng)
        else:
            return self._format_example(ex, ex, is_training, for_inference, rng)

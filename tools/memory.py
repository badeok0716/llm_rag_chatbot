from typing import Any, List, Dict
from langchain.memory import VectorStoreRetrieverMemory

from langchain.schema import Document


class ConversationVectorStoreRetrieverMemory(VectorStoreRetrieverMemory):
    input_prefix: str = "사용자"
    output_prefix: str = "선준"
    blacklist: list = []  # keys to ignore

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {
            k: v
            for k, v in inputs.items()
            if k != self.memory_key and k not in self.blacklist
        }
        texts = []
        for k, v in list(filtered_inputs.items()) + list(outputs.items()):
            if k == "input":
                k = self.input_prefix
            elif k == "response":
                k = self.output_prefix
            texts.append(f"{k}: {v}")
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]

class ConversationVectorStoreRetrieverMemoryNoUpdate(VectorStoreRetrieverMemory):
    input_prefix: str = "사용자"
    output_prefix: str = "선준"
    blacklist: list = []  # keys to ignore

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {
            k: v
            for k, v in inputs.items()
            if k != self.memory_key and k not in self.blacklist
        }
        texts = []
        for k, v in list(filtered_inputs.items()) + list(outputs.items()):
            if k == "input":
                k = self.input_prefix
            elif k == "response":
                k = self.output_prefix
            texts.append(f"{k}: {v}")
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]


    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        pass

    async def asave_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer."""
        pass
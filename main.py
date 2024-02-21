#llama.cppのpythonをインストールする際は以下のコードでインストールを行う。
#pip install llama-cpp-python==0.2.23


from llama_cpp import Llama
class LlamaAdapter():
    def __init__(self):
        self.llm = Llama(model_path='./japanese-stablelm-3b-4e1t-instruct-q4_K_M.gguf')

    def infer(self):
        output = self.llm(
            "### 指示: 日本の首相は？  \n ### 応答:",
            max_tokens=64,
            temperature=0.7,
        )["choices"][0]["text"]
        return output
    
if __name__ == '__main__':
    llama = LlamaAdapter()
    print(llama.infer())
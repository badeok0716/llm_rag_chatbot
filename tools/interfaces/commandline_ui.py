class CommandLine:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def run(self):
        print(f"안녕하세요! 이선준 캐릭터와 대화해보세요 :)")
        while True:
            text = input("사용자: ")
            if text:
                print(
                    f"{self.chatbot.name}: {self.chatbot.step(text)}"
                )
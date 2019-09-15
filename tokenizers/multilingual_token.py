from allennlp.data import Token


class MultilingualToken(Token):
    def __new__(cls, token: Token, lang: str):
        # TODO: Understand this code and understand why PyCharm type checker is yelling about the
        #  'cls' argument in the __new__ method.
        self = super(MultilingualToken, cls).__new__(cls, **token._asdict())
        self.lang = lang
        return self

    def show_token(self) -> str:
        return (f"{self.text} "
                f"(idx: {self.idx}) "
                f"(lemma: {self.lemma_}) "
                f"(pos: {self.pos_}) "
                f"(tag: {self.tag_}) "
                f"(dep: {self.dep_}) "
                f"(ent_type: {self.ent_type_}) "
                f"(lang: {self.lang}) ")

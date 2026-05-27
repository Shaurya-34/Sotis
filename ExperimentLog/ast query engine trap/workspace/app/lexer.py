from typing import Any, List

class TokenType:
    NUMBER = "NUMBER"
    VARIABLE = "VARIABLE"
    PLUS = "PLUS"
    MINUS = "MINUS"
    MUL = "MUL"
    DIV = "DIV"
    GT = "GT"
    LT = "LT"
    EQ = "EQ"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"

class Token:
    def __init__(self, type_: str, value: Any) -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"

class Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isspace():
                self.pos += 1
                continue
            if char == '+':
                tokens.append(Token(TokenType.PLUS, char))
                self.pos += 1
            elif char == '-':
                tokens.append(Token(TokenType.MINUS, char))
                self.pos += 1
            elif char == '*':
                tokens.append(Token(TokenType.MUL, char))
                self.pos += 1
            elif char == '/':
                tokens.append(Token(TokenType.DIV, char))
                self.pos += 1
            elif char == '>':
                tokens.append(Token(TokenType.GT, char))
                self.pos += 1
            elif char == '<':
                tokens.append(Token(TokenType.LT, char))
                self.pos += 1
            elif char == '=':
                tokens.append(Token(TokenType.EQ, char))
                self.pos += 1
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, char))
                self.pos += 1
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, char))
                self.pos += 1
            elif char.isdigit():
                start = self.pos
                while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
                    self.pos += 1
                val_str = self.text[start:self.pos]
                val = float(val_str) if '.' in val_str else int(val_str)
                tokens.append(Token(TokenType.NUMBER, val))
            elif char.isalpha() or char == '_':
                start = self.pos
                while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
                    self.pos += 1
                tokens.append(Token(TokenType.VARIABLE, self.text[start:self.pos]))
            else:
                raise ValueError(f"Invalid character: {char}")
        tokens.append(Token(TokenType.EOF, None))
        return tokens

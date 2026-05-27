from typing import Union
from .lexer import TokenType, Token

class AST:
    pass

class BinOp(AST):
    def __init__(self, left: AST, op: str, right: AST) -> None:
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self) -> str:
        return f"BinOp({self.left}, {self.op}, {self.right})"

class Num(AST):
    def __init__(self, value: Union[int, float]) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Num({self.value})"

class Var(AST):
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Var({self.name})"

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    @property
    def current_token(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, token_type: str) -> None:
        if self.current_token.type == token_type:
            self.pos += 1
        else:
            raise ValueError(f"Expected token {token_type}, got {self.current_token.type}")

    def parse(self) -> AST:
        return self.parse_expr()

    def parse_expr(self) -> AST:
        node = self.parse_term()
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS, TokenType.GT, TokenType.LT, TokenType.EQ):
            op = self.current_token.value
            self.consume(self.current_token.type)
            node = BinOp(node, op, self.parse_term())
        return node

    def parse_term(self) -> AST:
        node = self.parse_factor()
        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            op = self.current_token.value
            self.consume(self.current_token.type)
            node = BinOp(node, op, self.parse_factor())
        return node

    def parse_factor(self) -> AST:
        tok = self.current_token
        if tok.type == TokenType.NUMBER:
            self.consume(TokenType.NUMBER)
            return Num(tok.value)
        elif tok.type == TokenType.VARIABLE:
            self.consume(TokenType.VARIABLE)
            return Var(tok.value)
        elif tok.type == TokenType.LPAREN:
            self.consume(TokenType.LPAREN)
            node = self.parse_expr()
            self.consume(TokenType.RPAREN)
            return node
        raise ValueError(f"Unexpected token in factor: {tok}")
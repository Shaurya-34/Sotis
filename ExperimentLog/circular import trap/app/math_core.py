from app.helper import format_result
import math

class AdvancedMath:
    def factorial(self, n):
        return format_result("Factorial", self._calc_fact(n))

    @staticmethod
    def _calc_fact(n):
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        return 1 if n <= 1 else n * AdvancedMath._calc_fact(n - 1)

    def power(self, base, exp):
        return format_result("Power", base ** exp)

    def sqrt(self, n):
        if n < 0:
            raise ValueError("Square root is not defined for negative numbers")
        return format_result("Square Root", math.sqrt(n))

    def log(self, num, base):
        if num <= 0 or base <= 1:
            raise ValueError("Logarithm is not defined for given numbers")
        return format_result("Logarithm", math.log(num, base))

    def divide(self, num, den):
        if den == 0:
            raise ValueError("Cannot divide by zero")
        return format_result("Division", num / den)

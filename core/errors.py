"""错误码体系 E100~E500。"""


class MMForgeError(Exception):
    code = "E000"

    def __init__(self, msg: str = ""):
        super().__init__(f"[{self.code}] {msg}")


class ConfigError(MMForgeError):
    code = "E100"


class DataError(MMForgeError):
    code = "E200"


class ModelError(MMForgeError):
    code = "E300"


class TrainingError(MMForgeError):
    code = "E400"


class EvalError(MMForgeError):
    code = "E500"

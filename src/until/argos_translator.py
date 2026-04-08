from functools import lru_cache
import argostranslate.package
import argostranslate.translate


class ArgosTranslator:
    def __init__(self, from_code: str = "en", to_code: str = "zh"):
        self.from_code = from_code.strip()
        self.to_code = to_code.strip()
        self._available = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            self._install_language_package()
            test_text = "[Info] argostranslate is loaded."
            argostranslate.translate.translate(test_text, self.from_code, self.to_code)
            self._available = True
            print(test_text)
        except (ImportError, StopIteration, Exception):
            print("[Info] argostranslate initialization failed")

    def _install_language_package(self) -> None:
        installed_pkgs = argostranslate.package.get_installed_packages()
        if any(pkg.from_code == self.from_code and pkg.to_code == self.to_code for pkg in installed_pkgs):
            return

        print("[Info] Installing language package...")
        available_pkgs = argostranslate.package.get_available_packages()
        target_pkg = next(filter(
            lambda x: x.from_code == self.from_code and x.to_code == self.to_code,
            available_pkgs
        ))
        argostranslate.package.install_from_path(target_pkg.download())
        print("[Info] Language package installation complete.")

    @lru_cache(maxsize=1000)
    def translate(self, text) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.strip()
        if not text:
            return ""

        if self._available:
            translated = argostranslate.translate.translate(text, self.from_code, self.to_code)
            return f"{translated}\n{text}"
        return text

# if __name__ == "__main__":
#     translator = ArgosTranslator()
#     result = translator.translate("[Info] argostranslate is loaded.")
#     print(result)
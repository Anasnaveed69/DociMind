from typing import Any

_NLP_CACHE = None


class FallbackDoc:
    """Lightweight fallback spaCy Doc mock when spaCy is not installed."""
    def __init__(self, text: str):
        self.text = text
        self.ents = []


class FallbackNLP:
    """Lightweight fallback spaCy pipeline mock when spaCy is not installed."""
    def __call__(self, text: str) -> FallbackDoc:
        return FallbackDoc(text)

    def blank(self, lang: str):
        return self


def load_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """
    Singleton spaCy NLP model loader to prevent re-instantiation overhead.
    Falls back to FallbackNLP if spacy library is not installed.
    """
    global _NLP_CACHE
    if _NLP_CACHE is None:
        try:
            import spacy
            try:
                _NLP_CACHE = spacy.load(model_name)
            except Exception:
                try:
                    import spacy.cli
                    spacy.cli.download(model_name)
                    _NLP_CACHE = spacy.load(model_name)
                except Exception as e:
                    print(f"Warning: Could not download spaCy model {model_name} ({e}). Using blank English pipeline.")
                    _NLP_CACHE = spacy.blank("en")
        except ImportError:
            print("Warning: spaCy package not installed. Operating with Rule-Based Fallback NLP.")
            _NLP_CACHE = FallbackNLP()

    return _NLP_CACHE

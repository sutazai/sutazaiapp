class Document:
    def __init__(self, filename: str) -> None: ...
    def __iter__(self) -> Iterator[Page]: ...
    
class Page:
    def get_text(self) -> str: ... 
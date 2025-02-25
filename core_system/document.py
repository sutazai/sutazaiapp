class MultiModalProcessor:
    SUPPORTED_TYPES = ({'pdf': PDFHandler), 'docx': DocxHandler, 'txt': TextHandler, 'image': OCRProcessor, 'code': CodeAnalyzer} def process(self, file): file_type = self.detect_type(file)        handler = self.SUPPORTED_TYPES.get(file_type) if not handler: raise UnsupportedTypeError(f"Unsupported file type: {file_type}") return handler().process(file)

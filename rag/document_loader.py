"""
文档加载器
支持 PDF 和 TXT 文件
"""
from pathlib import Path
from typing import List
from utils.logger import logger


def load_pdf(file_path: Path) -> str:
    """加载 PDF 文件，返回纯文本"""
    from pypdf import PdfReader
    reader = PdfReader(str(file_path))
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n".join(texts)


def load_txt(file_path: Path) -> str:
    """加载 TXT 文件"""
    return file_path.read_text(encoding="utf-8")


def load_document(file_path: Path) -> str:
    """
    根据扩展名加载文档
    支持 .pdf, .txt
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def load_uploaded_file(uploaded_file) -> tuple[str, str]:
    """
    加载 Streamlit UploadedFile
    返回 (文件名, 文本内容)
    """
    import tempfile
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        content = load_document(tmp_path)
        logger.info(f"加载文档成功: {uploaded_file.name}, 长度={len(content)} 字符")
        return uploaded_file.name, content
    finally:
        tmp_path.unlink(missing_ok=True)

from __future__ import annotations

import pymupdf
import pytest

from askbio import AskBioConfig, DocumentError
from main import add_document


def make_pdf(path) -> None:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Authorized biology fixture.")
    document.save(path)
    document.close()


def test_add_document_copies_pdf_into_configured_corpus(tmp_path) -> None:
    source = tmp_path / "source.pdf"
    make_pdf(source)
    config = AskBioConfig(data_dir=tmp_path / "corpus", index_dir=tmp_path / "index")

    destination = add_document(source, config)

    assert destination == config.data_dir / "source.pdf"
    assert destination.read_bytes() == source.read_bytes()


def test_add_document_requires_replace_for_existing_file(tmp_path) -> None:
    source = tmp_path / "source.pdf"
    make_pdf(source)
    config = AskBioConfig(data_dir=tmp_path / "corpus", index_dir=tmp_path / "index")
    add_document(source, config)

    with pytest.raises(DocumentError, match="--replace"):
        add_document(source, config)

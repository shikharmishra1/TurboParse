import pypdfium2 as pdfium

# ── Full pipeline: predict + regions + TSR (ALL pages) ──────────────
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from converters.markdown_converter import predictions_to_markdown


from predict import predict_with_layout, pdf_features_with_images
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from collections import Counter
from pdf_token_type_labels.TokenType import TokenType

pdf_path = "test/manual.pdf"

# Open document once — reused by ALL three pipeline steps
pdf_doc = pdfium.PdfDocument(pdf_path)

pdf_features, token_image_overlaps, all_regions = pdf_features_with_images(
    pdf_path, dpi=150, pdf_document=pdf_doc,
)

    # ── Step 2: predict token types ───────────────────────────────

trainer = TokenTypeTrainer(
    [pdf_features], ModelConfiguration(), token_image_overlaps=token_image_overlaps,
)


pdf_features, predictions, all_figures, table_tsr_results = predict_with_layout(
    pdf_path, pdf_features=pdf_features, token_image_overlaps=token_image_overlaps,
    all_regions=all_regions, model_path="model\pdf_tokens_type.model",
    pdf_document=pdf_doc,
)


# ── Convert predictions to Markdown ────────────────────────────────


md_text = predictions_to_markdown(
    pdf_features=pdf_features,
    predictions=predictions,
    regions=all_figures,
    table_tsr_results=table_tsr_results,
    pdf_path=pdf_path,
    pdf_document=pdf_doc,   # ← reuse the doc opened above
)

print("=" * 60)
print("MARKDOWN OUTPUT")
print("=" * 60)
print(md_text[:2000] + "..." if len(md_text) > 2000 else md_text)

# Optionally save to file
out_path = Path("output.md")
out_path.write_text(md_text, encoding="utf-8")
print(f"\nSaved to {out_path.resolve()} ({len(md_text)} chars)")

# Close the shared document when done
pdf_doc.close()
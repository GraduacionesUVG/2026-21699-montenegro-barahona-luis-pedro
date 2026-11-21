"""Deterministic rendering of external sources before memory-policy dispatch."""

from __future__ import annotations

import hashlib

from membench.ingress.models import (
    ExternalDelivery,
    ExternalSourceKind,
    NormalizedExternalContent,
)

EXTERNAL_SOURCE_NORMALIZER = "external_source_normalizer_v1"


def normalize_external_delivery(
    delivery: ExternalDelivery,
) -> NormalizedExternalContent:
    """Render one source with stable headers, preserving the exact source body."""

    heading = (
        "External email"
        if delivery.source_kind == ExternalSourceKind.EMAIL
        else "External document"
    )
    headers = [heading]
    if delivery.sender is not None:
        headers.append(f"From: {delivery.sender}")
    headers.extend((f"Title: {delivery.title}", f"Source-ID: {delivery.source_id}"))
    text = "\n".join(headers) + "\n\n" + delivery.content
    return NormalizedExternalContent(
        normalizer=EXTERNAL_SOURCE_NORMALIZER,
        source_kind=delivery.source_kind,
        source_id=delivery.source_id,
        source_content_sha256=delivery.content_sha256,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )

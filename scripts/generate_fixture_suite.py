#!/usr/bin/env python3
"""
Generate the public synthetic PDF fixture suite.

The generated files are safe to publish and are used by parser tests,
UI smoke tests, and optional provider benchmarks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"
EXPECTED_DIR = FIXTURE_DIR / "expected"


@dataclass
class LineItemSpec:
    line_number: int
    description: str
    sku: str
    quantity: int
    unit_price: float
    extended_price: float
    item_type: str
    manufacturer: str | None = None
    term_months: int | None = None
    taxable: bool | None = None


@dataclass
class QuoteSpec:
    stem: str
    title: str
    vendor_name: str
    customer_name: str
    quote_id: str
    quote_date: str
    valid_until: str
    payment_terms: str
    amounts: dict[str, float]
    line_items: list[LineItemSpec]
    notes: list[str] = field(default_factory=list)
    extra_fields: dict[str, str] = field(default_factory=dict)


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FixtureTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "FixtureHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#111827"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "FixtureBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1F2937"),
        ),
        "small": ParagraphStyle(
            "FixtureSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#4B5563"),
        ),
    }


def _section_table(rows: list[list[str]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("ALIGN", (-2, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F9FAFB")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def discounted_hardware_quote() -> QuoteSpec:
    items = [
        LineItemSpec(
            1,
            "Rack Enclosure 42U",
            "RACK-42U",
            2,
            2500.00,
            5000.00,
            "hardware",
            "Axis Rackworks",
        ),
        LineItemSpec(
            2,
            "Compute Server X750",
            "SVR-X750",
            4,
            4200.00,
            16800.00,
            "hardware",
            "Northwind Compute",
        ),
        LineItemSpec(
            3, "Structured Cabling Kit", "CAB-KIT", 1, 1558.02, 1558.02, "hardware"
        ),
        LineItemSpec(
            4, "Install Labor Bundle", "LABOR-SETUP", 8, 187.50, 1500.00, "services"
        ),
    ]
    return QuoteSpec(
        stem="discounted_hardware_quote",
        title="Discounted Hardware Quote",
        vendor_name="Summit Equipment Supply",
        customer_name="Atlas Manufacturing Group",
        quote_id="DHQ-2024-0142",
        quote_date="2024-02-01",
        valid_until="2024-03-01",
        payment_terms="Net 30",
        amounts={
            "subtotal": 24858.02,
            "tax": 1905.17,
            "grand_total": 26763.19,
            "list_total": 40973.00,
            "discounted_total": 26763.19,
        },
        line_items=items,
        notes=[
            "The document shows both list pricing and discounted pricing. "
            "The discounted finance total should be used.",
            "Taxes are calculated on the discounted subtotal.",
        ],
    )


def enterprise_term_quote() -> QuoteSpec:
    items = [
        LineItemSpec(
            1,
            "Enterprise Platform Subscription",
            "ENT-PLATFORM",
            5,
            19878.32,
            99391.60,
            "software",
            term_months=60,
        ),
        LineItemSpec(
            2,
            "Premium Support Package",
            "SUPPORT-PREM",
            5,
            4000.00,
            20000.00,
            "services",
            term_months=60,
        ),
        LineItemSpec(
            3, "Implementation Services", "SERV-IMPL", 1, 24000.00, 24000.00, "services"
        ),
    ]
    return QuoteSpec(
        stem="enterprise_term_quote",
        title="Enterprise Term Quote",
        vendor_name="Bluewave Software",
        customer_name="Global Retail Group",
        quote_id="ETQ-2024-1001",
        quote_date="2024-11-01",
        valid_until="2024-12-31",
        payment_terms="Net 45",
        amounts={
            "subtotal": 143391.60,
            "tax": 0.0,
            "grand_total": 143391.60,
        },
        line_items=items,
        notes=[
            "Coverage period runs from 2024-11-22 to 2029-11-21.",
            "Subscription term is implied by the coverage dates rather "
            "than a separate month count field.",
        ],
        extra_fields={
            "coverage_start": "2024-11-22",
            "coverage_end": "2029-11-21",
            "subscription_term_months": "60",
            "auto_renew": "true",
        },
    )


def service_order_with_credits() -> QuoteSpec:
    items = [
        LineItemSpec(
            1,
            "Managed Infrastructure Services - 36 Month",
            "MNS-36M",
            36,
            12000.00,
            432000.00,
            "services",
            term_months=36,
        ),
        LineItemSpec(
            2,
            "Core Compute Infrastructure",
            "CORE-INFRA",
            1,
            165312.00,
            165312.00,
            "hardware",
            "Northwind Compute",
        ),
        LineItemSpec(
            3,
            "Initial Setup and Migration NRC",
            "NRC-SETUP",
            1,
            45300.00,
            45300.00,
            "services",
        ),
        LineItemSpec(
            4, "Trade-In Credit", "CREDIT-TRADE", 1, 45300.00, -45300.00, "credit"
        ),
    ]
    return QuoteSpec(
        stem="service_order_with_credits",
        title="Service Order With Credits",
        vendor_name="Cedar Managed Services",
        customer_name="Beacon Financial Group",
        quote_id="SOC-2024-0789",
        quote_date="2024-03-15",
        valid_until="2024-04-15",
        payment_terms="Net 60",
        amounts={
            "subtotal": 597312.00,
            "tax": 0.0,
            "grand_total": 597312.00,
            "net_amount_due": 552012.00,
            "nrc_total": 45300.00,
        },
        line_items=items,
        notes=[
            "Total Contract Value should be treated as the grand total.",
            "The net amount due is shown separately after applying the "
            "trade-in credit.",
        ],
        extra_fields={
            "subscription_term_months": "36",
            "net_amount_due": "552012.00",
        },
    )


def mixed_category_quote() -> QuoteSpec:
    items = [
        LineItemSpec(
            1,
            "Switch Core 48-Port",
            "SW-CORE-48",
            4,
            5500.00,
            22000.00,
            "hardware",
            "Northwind Systems",
            taxable=True,
        ),
        LineItemSpec(
            2,
            "Network License 3-Year",
            "LIC-NET-3Y",
            4,
            1200.00,
            4800.00,
            "software",
            "Northwind Systems",
            term_months=36,
            taxable=True,
        ),
        LineItemSpec(
            3,
            "Implementation Services",
            "SERV-INSTALL",
            1,
            4000.00,
            4000.00,
            "services",
            taxable=True,
        ),
    ]
    return QuoteSpec(
        stem="mixed_category_quote",
        title="Mixed Category Quote",
        vendor_name="Northwind Systems",
        customer_name="Acme Retail Group",
        quote_id="MCQ-2024-0001",
        quote_date="2024-01-15",
        valid_until="2024-02-15",
        payment_terms="Net 30",
        amounts={
            "subtotal": 30800.00,
            "tax": 1398.00,
            "grand_total": 32198.00,
        },
        line_items=items,
        notes=[
            "This quote includes hardware, software, and services in a single table.",
            "Tax is included in the finance total, while shipping is zero.",
        ],
    )


def multi_page_quote() -> QuoteSpec:
    items: list[LineItemSpec] = []
    for line_number in range(1, 52):
        item_type = "hardware" if line_number % 3 != 0 else "software"
        unit_price = float(Decimal("5000.00") + Decimal(line_number * 100))
        quantity = 1 + (line_number % 3)
        extended_price = round(unit_price * quantity, 2)
        items.append(
            LineItemSpec(
                line_number=line_number,
                description=f"Automation Component Part #{line_number:03d}",
                sku=f"AUTO-{line_number:04d}",
                quantity=quantity,
                unit_price=unit_price,
                extended_price=extended_price,
                item_type=item_type,
                manufacturer="Various",
            )
        )

    subtotal = round(sum(item.extended_price for item in items), 2)
    tax = round(subtotal * 0.0625, 2)
    shipping = 1500.00
    grand_total = 314432.95
    discounts = round(subtotal + tax + shipping - grand_total, 2)

    return QuoteSpec(
        stem="multi_page_quote",
        title="Multi-Page Quote",
        vendor_name="Riverstone Distribution",
        customer_name="Harbor Manufacturing Co.",
        quote_id="MPQ-2024-0500",
        quote_date="2024-04-01",
        valid_until="2024-05-01",
        payment_terms="Net 30",
        amounts={
            "subtotal": subtotal,
            "shipping": shipping,
            "tax": tax,
            "discounts": discounts,
            "grand_total": grand_total,
        },
        line_items=items,
        notes=[
            "This document intentionally spans multiple pages with "
            "repeated table headers.",
            "The final total reconciles subtotal, shipping, tax, "
            "and discount adjustments.",
        ],
    )


FIXTURES = [
    discounted_hardware_quote(),
    enterprise_term_quote(),
    service_order_with_credits(),
    mixed_category_quote(),
    multi_page_quote(),
]


def _render_quote(spec: QuoteSpec) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    pdf_path = FIXTURE_DIR / f"{spec.stem}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=spec.title,
        author="Vendor Quote Extractor Fixture Generator",
    )

    story = [
        Paragraph(spec.title, styles["title"]),
        Paragraph(spec.vendor_name, styles["body"]),
        Spacer(1, 0.1 * inch),
    ]

    header_rows = [
        ["Quote ID", spec.quote_id, "Quote Date", spec.quote_date],
        ["Customer", spec.customer_name, "Valid Until", spec.valid_until],
        ["Payment Terms", spec.payment_terms, "Currency", "USD"],
    ]
    for key, value in spec.extra_fields.items():
        header_rows.append([key.replace("_", " ").title(), value, "", ""])

    story.append(Paragraph("Document Summary", styles["heading"]))
    story.append(
        _section_table(
            [["Field", "Value", "Field", "Value"], *header_rows],
            [1.6 * inch, 2.0 * inch, 1.4 * inch, 1.8 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Notes", styles["heading"]))
    for note in spec.notes:
        story.append(Paragraph(f"- {note}", styles["small"]))
    story.append(Spacer(1, 0.12 * inch))

    item_rows = [
        ["Line", "Description", "SKU", "Qty", "Unit Price", "Extended", "Type"]
    ]
    for item in spec.line_items:
        item_rows.append(
            [
                str(item.line_number),
                item.description,
                item.sku,
                str(item.quantity),
                _money(item.unit_price),
                _money(item.extended_price),
                item.item_type,
            ]
        )

    story.append(Paragraph("Line Items", styles["heading"]))
    item_table = LongTable(
        item_rows,
        repeatRows=1,
        colWidths=[
            0.45 * inch,
            2.55 * inch,
            1.1 * inch,
            0.45 * inch,
            0.9 * inch,
            0.95 * inch,
            0.75 * inch,
        ],
        hAlign="LEFT",
    )
    item_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),
                ("ALIGN", (4, 1), (5, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F9FAFB")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(item_table)
    story.append(Spacer(1, 0.15 * inch))

    amount_rows = [["Amount", "Value"]]
    for key, value in spec.amounts.items():
        amount_rows.append([key.replace("_", " ").title(), _money(value)])
    story.append(Paragraph("Financial Summary", styles["heading"]))
    story.append(_section_table(amount_rows, [2.25 * inch, 1.45 * inch]))

    doc.build(story)

    expected_payload = {
        "quote_id": spec.quote_id,
        "vendor_name": spec.vendor_name,
        "customer_name": spec.customer_name,
        "quote_date": spec.quote_date,
        "valid_until": spec.valid_until,
        "payment_terms": spec.payment_terms,
        "currency": "USD",
        "amounts": spec.amounts,
        "line_items": [asdict(item) for item in spec.line_items],
        "notes": spec.notes,
        "extra_fields": spec.extra_fields,
    }
    expected_path = EXPECTED_DIR / f"expected_{spec.stem}.json"
    expected_path.write_text(
        json.dumps(expected_payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    for spec in FIXTURES:
        _render_quote(spec)
        print(f"Generated {spec.stem}.pdf")


if __name__ == "__main__":
    main()

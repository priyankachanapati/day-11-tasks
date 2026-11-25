
from pydantic import BaseModel, Field


class InvoiceLineItem(BaseModel):
    item_name: str = Field(description="Name of the purchased item")
    quantity: float = Field(description="Quantity purchased")
    unit_price: float = Field(description="Price per unit")
    line_total: float = Field(description="Total price for this item")


class ExtractedInvoice(BaseModel):
    invoice_number: str = Field(description="Unique invoice number")
    vendor_name: str = Field(description="Name of the vendor")
    invoice_date: str = Field(description="Invoice date in YYYY-MM-DD format")
    due_date: str = Field(description="Payment due date in YYYY-MM-DD format")
    total_amount: float = Field(description="Total invoice amount")
    currency: str = Field(description="Currency code such as USD or INR")
    line_items: list[InvoiceLineItem] = Field(
        description="List of purchased items"
    )
    notes: str = Field(description="Additional invoice notes, empty if unavailable")

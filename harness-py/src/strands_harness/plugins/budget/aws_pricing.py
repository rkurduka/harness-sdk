"""Optional live rate fetching from the AWS Price List API.

Fetches on-demand token pricing for Amazon Bedrock models and converts it
into the plugin's per-token rate-table format. Coverage note: AWS publishes
token pricing for many model families (Amazon Nova, Meta Llama, Mistral,
DeepSeek, and others) but NOT for recent Anthropic Claude models — those
continue to be priced by the static defaults or ``custom_rates``.

Requires AWS credentials with the read-only ``pricing:GetProducts``
permission. Any failure returns an empty dict and logs a warning, so callers
always fall back to the static table.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# The Price List API is only served from a few endpoint regions; the Bedrock
# region whose prices we want is passed as a regionCode *filter* instead.
_PRICING_ENDPOINT_REGION = "us-east-1"

# Usagetype variants that carry discounted or special pricing. Without these
# exclusions, e.g. batch rates can overwrite the on-demand rate for a model.
_EXCLUDED_USAGETYPE_MARKERS = ("-batch", "-custom-model", "cross-region", "-flex", "-priority")


def find_rates(products: list[dict[str, Any]], suffix: str) -> dict[str, float]:
    """Map rate-table keys to per-token USD prices for one direction.

    Args:
        products: Parsed Price List product records.
        suffix: The usagetype suffix for this direction, e.g. "-input-tokens".

    Each product contributes up to two lookup keys: the normalized ``model``
    attribute ("Nova Lite" -> "nova-lite") and the usagetype core
    ("USE1-Llama3-3-70B-input-tokens" -> "llama3-3-70b"). Different providers'
    runtime model IDs match different conventions; unused keys are harmless.
    """
    rates: dict[str, float] = {}

    for product in products:
        attributes = product.get("product", {}).get("attributes", {})
        usage_type = attributes.get("usagetype", "")

        # Exact suffix match keeps out audio/image token counts and other
        # variants; the marker check keeps out discounted pricing tiers.
        if not usage_type.endswith(suffix):
            continue
        if any(marker in usage_type for marker in _EXCLUDED_USAGETYPE_MARKERS):
            continue

        for term in product.get("terms", {}).get("OnDemand", {}).values():
            for dimension in term.get("priceDimensions", {}).values():
                if dimension.get("unit") != "1K tokens":
                    continue
                price_per_1k = float(dimension.get("pricePerUnit", {}).get("USD", 0))
                if price_per_1k <= 0:
                    continue  # zero-priced records would bill the model as free
                rate = price_per_1k / 1000  # per-1K -> per-token

                # Key 1: usagetype core, e.g. "USE1-Llama3-3-70B-input-tokens" -> "llama3-3-70b"
                core = usage_type[: -len(suffix)].partition("-")[2].lower()
                if core:
                    # round rate to 12 decimal points
                    rates[core] = rate
                # Key 2: normalized model attribute, e.g. "Nova Lite" -> "nova-lite"
                model = attributes.get("model", "").lower().strip().replace(" ", "-")
                if model:
                    rates[model] = rate

    return rates


def build_rates_from_products(
    input_products: list[dict[str, Any]], output_products: list[dict[str, Any]]
) -> dict[str, tuple[float, float]]:
    """Combine input- and output-token product records into a rate table.

    Only models with both an input and an output price are included.
    """
    input_rates = find_rates(input_products, "-input-tokens")
    output_rates = find_rates(output_products, "-output-tokens")

    all_models = set(input_rates.keys()) & set(output_rates.keys())
    return {model: (input_rates[model], output_rates[model]) for model in all_models}


def fetch_aws_rates(region_name: str = "us-east-1", boto_session: Any = None) -> dict[str, tuple[float, float]]:
    """Fetch current Bedrock on-demand token rates from the AWS Price List API.

    Args:
        region_name: The Bedrock region whose prices to fetch (e.g. "us-east-1").
        boto_session: Optional boto3 Session to create the pricing client from.

    Returns:
        A per-token rate table in ``custom_rates`` format, or an empty dict if
        the fetch fails for any reason (no credentials, no permission, etc.).
    """
    try:
        import boto3

        session = boto_session or boto3
        pricing_client = session.client("pricing", region_name=_PRICING_ENDPOINT_REGION)
        paginator = pricing_client.get_paginator("get_products")

        def query(inference_type: str) -> list[dict[str, Any]]:
            """One filtered query per direction: ~2 pages instead of the full catalog."""
            filters = [
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region_name},
                {"Type": "TERM_MATCH", "Field": "inferenceType", "Value": inference_type},
            ]
            products: list[dict[str, Any]] = []
            for page in paginator.paginate(ServiceCode="AmazonBedrock", Filters=filters):
                products.extend(json.loads(price_string) for price_string in page["PriceList"])
            return products

        rates = build_rates_from_products(query("Input tokens"), query("Output tokens"))
        logger.info("Fetched AWS pricing for %d Bedrock model entries (region %s)", len(rates), region_name)
        return rates
    except Exception as e:
        logger.warning(f"Failed to fetch AWS pricing data: {e}")
        return {}


def format_rates(rates: dict[str, tuple[float, float]]) -> str:
    """Human-readable per-token rate table in plain decimal notation.

    For display only — the stored rates are used as-is for billing.
    Fixed-point formatting (no scientific notation) with 10 decimal
    places, enough for the smallest published token prices.
    """
    if not rates:
        return "(no rates)"
    width = max(len(key) for key in rates)
    header = f"{'model':<{width}}  {'input $/token':>15}  {'output $/token':>15}"
    lines = [header, "-" * len(header)]
    for key in sorted(rates):
        input_rate, output_rate = rates[key]
        lines.append(f"{key:<{width}}  {input_rate:>15.10f}  {output_rate:>15.10f}")
    return "\n".join(lines)


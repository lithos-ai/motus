"""Multimodal tools — tools that return rich content types.

Demonstrates tools returning different content types: plain text, JSON
data, and formatted tables. The agent combines outputs from multiple
tools into a cohesive response.

Deploy: motus serve start examples.google_adk.multimodal_tools:root_agent
"""

from motus.google_adk.agents.llm_agent import Agent


def get_chart_data(metric: str, period: str = "weekly") -> dict:
    """Get chart data for a business metric.

    Args:
        metric: The metric name (e.g. 'revenue', 'users', 'orders').
        period: Time period ('daily', 'weekly', 'monthly').
    """
    data = {
        "revenue": {"weekly": [12000, 15000, 13500, 18000, 16000, 19000, 21000]},
        "users": {"weekly": [150, 180, 165, 200, 210, 195, 230]},
        "orders": {"weekly": [45, 52, 48, 61, 58, 65, 72]},
    }
    values = data.get(metric, {}).get(period, [0] * 7)
    return {
        "metric": metric,
        "period": period,
        "values": values,
        "total": sum(values),
        "average": round(sum(values) / len(values), 1),
        "trend": "up" if values[-1] > values[0] else "down",
    }


def generate_report_section(title: str, content: str) -> dict:
    """Generate a formatted report section.

    Args:
        title: Section title.
        content: Section content in markdown.
    """
    return {
        "section": title,
        "formatted": f"## {title}\n\n{content}",
        "word_count": len(content.split()),
    }


def compare_periods(metric: str, period_a: str, period_b: str) -> dict:
    """Compare a metric across two time periods.

    Args:
        metric: The metric to compare.
        period_a: First period label.
        period_b: Second period label.
    """
    # Mock comparison data
    comparisons = {
        "revenue": {"this_week": 21000, "last_week": 19000},
        "users": {"this_week": 230, "last_week": 195},
        "orders": {"this_week": 72, "last_week": 65},
    }
    data = comparisons.get(metric, {"this_week": 0, "last_week": 0})
    change = data["this_week"] - data["last_week"]
    pct = round(change / max(data["last_week"], 1) * 100, 1)
    return {
        "metric": metric,
        period_a: data["last_week"],
        period_b: data["this_week"],
        "change": change,
        "change_pct": f"{pct}%",
    }


root_agent = Agent(
    model="gemini-2.5-flash",
    name="analytics_agent",
    description="Generates business analytics reports with charts and comparisons.",
    instruction=(
        "You are a business analytics assistant. Help users understand "
        "their metrics by fetching chart data, generating report sections, "
        "and comparing time periods. Present data clearly with trends "
        "and insights."
    ),
    tools=[get_chart_data, generate_report_section, compare_periods],
)

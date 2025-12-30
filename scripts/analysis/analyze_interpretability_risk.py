#!/usr/bin/env python3
"""
Analyze the AI risk assessment TSV to compare open vs closed source interpretability risks.

Outputs a report with:
1. Overall conclusion on whether open or closed source is more risky
2. Breakdown by risk domain
3. Standout examples of highest risk items for each approach
"""

import csv
from pathlib import Path
from collections import defaultdict
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).parent.parent


class RiskEntry(NamedTuple):
    title: str
    category: str
    subcategory: str
    description: str
    likelihood: int  # 2=open beneficial, 3=neutral, 4=open risky
    magnitude: int   # same scale
    reason: str
    domain: str
    subdomain: str


def parse_tsv(filepath: Path) -> list[RiskEntry]:
    """Parse the TSV file and extract risk entries with interpretability assessments.

    The file has multi-line entries where description text wraps, so we need to
    handle line continuation by joining lines that don't start with a paper title.
    """
    entries = []

    # Read all lines and join multi-line entries
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    # Skip header rows (first 3 lines)
    data_lines = raw_lines[3:]

    # Join multi-line entries - lines that don't have tab-separated fields are continuations
    processed_lines = []
    current_line = ""

    for line in data_lines:
        # Count tabs to identify if this is a new record or continuation
        # A proper record line should have many tabs (representing columns)
        tab_count = line.count('\t')

        # A new record starts with a title and has many tab separators
        # Continuation lines typically have few or no tabs at the start
        if tab_count >= 10 and not line.startswith(' ') and not line.startswith('\t'):
            # This is a new record
            if current_line:
                processed_lines.append(current_line)
            current_line = line.rstrip('\n\r')
        else:
            # This is a continuation line - append to current
            current_line += " " + line.strip()

    # Don't forget the last line
    if current_line:
        processed_lines.append(current_line)

    # Now parse the processed lines
    for line in processed_lines:
        row = line.split('\t')

        # Need at least 19 columns (indices 0-18)
        # Columns are:
        # [0] Title, [1] QuickRef, [2] Ev_ID, [3] Paper_ID, [4] Cat_ID, [5] SubCat_ID,
        # [6] AddEv_ID, [7] Category level, [8] Risk category, [9] Risk subcategory,
        # [10] Description, [11] Interp_Likelihood, [12] Interp_Magnitude, [13] Interp_Reason,
        # [14] Entity, [15] Intent, [16] Timing, [17] Domain, [18] Sub-domain
        if len(row) < 19:
            continue

        # Skip rows without interpretability assessment
        try:
            likelihood_str = row[11].strip() if len(row) > 11 else ""
            magnitude_str = row[12].strip() if len(row) > 12 else ""

            if not likelihood_str or not magnitude_str:
                continue

            likelihood = int(likelihood_str)
            magnitude = int(magnitude_str)
        except (ValueError, IndexError):
            continue

        # Skip if scores are out of expected range
        if not (2 <= likelihood <= 4) or not (2 <= magnitude <= 4):
            continue

        title = row[0].strip() if len(row) > 0 else ""
        category = row[8].strip() if len(row) > 8 else ""
        subcategory = row[9].strip() if len(row) > 9 else ""
        description = row[10].strip() if len(row) > 10 else ""
        reason = row[13].strip() if len(row) > 13 else ""
        domain = row[17].strip() if len(row) > 17 else ""  # Domain is at index 17
        subdomain = row[18].strip() if len(row) > 18 else ""  # Sub-domain is at index 18

        if not reason:  # Skip entries without a reason
            continue

        entries.append(RiskEntry(
            title=title,
            category=category,
            subcategory=subcategory,
            description=description,
            likelihood=likelihood,
            magnitude=magnitude,
            reason=reason,
            domain=domain,
            subdomain=subdomain,
        ))

    return entries


def analyze_entries(entries: list[RiskEntry]) -> dict:
    """Analyze entries and compute statistics."""

    # Overall stats
    total = len(entries)

    # Count by score (2=open beneficial, 3=neutral, 4=open risky)
    likelihood_counts = defaultdict(int)
    magnitude_counts = defaultdict(int)

    # Combined score (likelihood + magnitude) for ranking
    combined_scores = []

    # By domain
    domain_stats = defaultdict(lambda: {
        'count': 0,
        'likelihood_sum': 0,
        'magnitude_sum': 0,
        'entries': []
    })

    for entry in entries:
        likelihood_counts[entry.likelihood] += 1
        magnitude_counts[entry.magnitude] += 1

        combined = entry.likelihood + entry.magnitude
        combined_scores.append((combined, entry))

        domain = entry.domain or "Unknown"
        domain_stats[domain]['count'] += 1
        domain_stats[domain]['likelihood_sum'] += entry.likelihood
        domain_stats[domain]['magnitude_sum'] += entry.magnitude
        domain_stats[domain]['entries'].append(entry)

    # Sort for standout examples
    # Highest combined score = most risky for open source (pro-closed)
    # Lowest combined score = most beneficial for open source (pro-open)
    combined_scores.sort(key=lambda x: x[0])

    pro_open = [e for score, e in combined_scores if score <= 4]  # 2+2=4
    pro_closed = [e for score, e in combined_scores if score >= 7]  # 4+3=7 or higher
    neutral = [e for score, e in combined_scores if 5 <= score <= 6]

    return {
        'total': total,
        'likelihood_counts': dict(likelihood_counts),
        'magnitude_counts': dict(magnitude_counts),
        'pro_open': pro_open,
        'pro_closed': pro_closed,
        'neutral': neutral,
        'domain_stats': dict(domain_stats),
        'combined_scores': combined_scores,
    }


def generate_report(analysis: dict) -> str:
    """Generate a markdown report from the analysis."""

    lines = []
    lines.append("# Open vs Closed Source Interpretability Risk Analysis")
    lines.append("")
    lines.append("**Source**: AI Risk Database with Interpretability Assessment")
    lines.append("**Scale**: 2 = Open-source beneficial, 3 = Neutral, 4 = Open-source increases risk")
    lines.append("")

    # Overall summary
    lines.append("---")
    lines.append("## Executive Summary")
    lines.append("")

    total = analysis['total']
    pro_open = len(analysis['pro_open'])
    pro_closed = len(analysis['pro_closed'])
    neutral = len(analysis['neutral'])

    lines.append(f"**Total Assessed Risks**: {total}")
    lines.append("")
    lines.append(f"| Category | Count | Percentage |")
    lines.append(f"|----------|-------|------------|")
    lines.append(f"| Pro Open-Source (score ≤4) | {pro_open} | {100*pro_open/total:.1f}% |")
    lines.append(f"| Neutral (score 5-6) | {neutral} | {100*neutral/total:.1f}% |")
    lines.append(f"| Pro Closed-Source (score ≥7) | {pro_closed} | {100*pro_closed/total:.1f}% |")
    lines.append("")

    # Calculate weighted averages
    combined = analysis['combined_scores']
    avg_combined = sum(s for s, _ in combined) / len(combined)
    avg_likelihood = sum(e.likelihood for _, e in combined) / len(combined)
    avg_magnitude = sum(e.magnitude for _, e in combined) / len(combined)

    lines.append(f"**Average Scores** (lower = more beneficial for open-source):")
    lines.append(f"- Likelihood: {avg_likelihood:.2f}")
    lines.append(f"- Magnitude: {avg_magnitude:.2f}")
    lines.append(f"- Combined: {avg_combined:.2f} (minimum possible: 4, maximum: 8)")
    lines.append("")

    # Overall conclusion
    lines.append("### Overall Conclusion")
    lines.append("")

    if avg_combined < 5.0:
        conclusion = "**OPEN-SOURCE INTERPRETABILITY IS NET BENEFICIAL**"
        explanation = "The majority of risk assessments indicate that open-source interpretability tools would reduce AI risks by enabling broader detection, mitigation, and oversight of harmful AI behaviors."
    elif avg_combined > 6.0:
        conclusion = "**CLOSED-SOURCE INTERPRETABILITY IS SAFER**"
        explanation = "The majority of risk assessments indicate that restricting interpretability tools would reduce AI risks by limiting malicious actors' ability to understand and exploit AI systems."
    else:
        conclusion = "**MIXED RESULTS - DOMAIN-DEPENDENT**"
        explanation = "The overall balance is close to neutral, suggesting that the optimal approach depends heavily on the specific risk domain. Some risks are better addressed by open tools, others by restricted access."

    lines.append(conclusion)
    lines.append("")
    lines.append(explanation)
    lines.append("")

    # Domain breakdown
    lines.append("---")
    lines.append("## Analysis by Risk Domain")
    lines.append("")

    domain_stats = analysis['domain_stats']

    # Sort domains by average combined score
    domain_avgs = []
    for domain, stats in domain_stats.items():
        if stats['count'] > 0:
            avg = (stats['likelihood_sum'] + stats['magnitude_sum']) / stats['count']
            domain_avgs.append((avg, domain, stats))

    domain_avgs.sort(key=lambda x: x[0])

    lines.append("| Domain | Count | Avg Likelihood | Avg Magnitude | Avg Combined | Verdict |")
    lines.append("|--------|-------|----------------|---------------|--------------|---------|")

    for avg, domain, stats in domain_avgs:
        count = stats['count']
        avg_l = stats['likelihood_sum'] / count
        avg_m = stats['magnitude_sum'] / count

        if avg <= 5.0:
            verdict = "Pro Open"
        elif avg >= 6.0:
            verdict = "Pro Closed"
        else:
            verdict = "Neutral"

        lines.append(f"| {domain} | {count} | {avg_l:.2f} | {avg_m:.2f} | {avg:.2f} | {verdict} |")

    lines.append("")

    # Domain-specific conclusions
    lines.append("### Domain Conclusions")
    lines.append("")

    for avg, domain, stats in domain_avgs:
        count = stats['count']
        avg_l = stats['likelihood_sum'] / count
        avg_m = stats['magnitude_sum'] / count

        lines.append(f"**{domain}** ({count} risks)")

        if avg <= 4.5:
            lines.append(f"- Strongly favors **open-source** interpretability (avg {avg:.2f})")
            lines.append(f"- Open tools enable detection and mitigation of risks in this domain")
        elif avg <= 5.5:
            lines.append(f"- Slight lean toward **open-source** (avg {avg:.2f})")
            lines.append(f"- Benefits of transparency slightly outweigh risks of misuse")
        elif avg <= 6.5:
            lines.append(f"- Slight lean toward **closed-source** (avg {avg:.2f})")
            lines.append(f"- Risks of enabling bad actors slightly outweigh transparency benefits")
        else:
            lines.append(f"- Strongly favors **closed-source** interpretability (avg {avg:.2f})")
            lines.append(f"- Restricting access reduces risks in this domain")

        lines.append("")

    # Standout examples
    lines.append("---")
    lines.append("## Standout Examples")
    lines.append("")

    # Most risky for open source
    lines.append("### Highest Risk Items for Open-Source Approach")
    lines.append("*(These risks are most amplified by widely available interpretability tools)*")
    lines.append("")

    # Get top 5 highest combined scores
    highest_risk = sorted(analysis['combined_scores'], key=lambda x: -x[0])[:5]

    for i, (score, entry) in enumerate(highest_risk, 1):
        lines.append(f"**{i}. {entry.category}** (Score: {score})")
        lines.append(f"- Domain: {entry.domain}")
        lines.append(f"- Likelihood: {entry.likelihood}, Magnitude: {entry.magnitude}")
        lines.append(f"- Reason: {entry.reason[:300]}..." if len(entry.reason) > 300 else f"- Reason: {entry.reason}")
        lines.append("")

    # Most beneficial for open source
    lines.append("### Highest Benefit Items for Open-Source Approach")
    lines.append("*(These risks are most reduced by widely available interpretability tools)*")
    lines.append("")

    # Get top 5 lowest combined scores
    lowest_risk = sorted(analysis['combined_scores'], key=lambda x: x[0])[:5]

    for i, (score, entry) in enumerate(lowest_risk, 1):
        lines.append(f"**{i}. {entry.category}** (Score: {score})")
        lines.append(f"- Domain: {entry.domain}")
        lines.append(f"- Likelihood: {entry.likelihood}, Magnitude: {entry.magnitude}")
        lines.append(f"- Reason: {entry.reason[:300]}..." if len(entry.reason) > 300 else f"- Reason: {entry.reason}")
        lines.append("")

    # Risk distribution by score
    lines.append("---")
    lines.append("## Score Distribution")
    lines.append("")

    lines.append("### Likelihood Scores")
    lines.append("| Score | Meaning | Count |")
    lines.append("|-------|---------|-------|")
    for score in [2, 3, 4]:
        count = analysis['likelihood_counts'].get(score, 0)
        if score == 2:
            meaning = "Open reduces likelihood"
        elif score == 3:
            meaning = "Neutral"
        else:
            meaning = "Open increases likelihood"
        lines.append(f"| {score} | {meaning} | {count} |")

    lines.append("")

    lines.append("### Magnitude Scores")
    lines.append("| Score | Meaning | Count |")
    lines.append("|-------|---------|-------|")
    for score in [2, 3, 4]:
        count = analysis['magnitude_counts'].get(score, 0)
        if score == 2:
            meaning = "Open reduces impact"
        elif score == 3:
            meaning = "Neutral"
        else:
            meaning = "Open increases impact"
        lines.append(f"| {score} | {meaning} | {count} |")

    lines.append("")

    return "\n".join(lines)


def main():
    tsv_path = PROJECT_ROOT / "docs" / "risk assessment" / "ai_risks_with_interpretability_assessment.tsv"

    if not tsv_path.exists():
        print(f"Error: TSV file not found at {tsv_path}")
        return

    print(f"Parsing {tsv_path}...")
    entries = parse_tsv(tsv_path)
    print(f"Found {len(entries)} risk entries with interpretability assessments")

    print("Analyzing entries...")
    analysis = analyze_entries(entries)

    print("Generating report...")
    report = generate_report(analysis)

    # Output report
    output_path = PROJECT_ROOT / "docs" / "results" / "INTERPRETABILITY_RISK_ANALYSIS.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
    print("\n" + "=" * 80)
    print(report)


if __name__ == "__main__":
    main()

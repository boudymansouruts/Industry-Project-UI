#!/usr/bin/env python3
"""
Programmatically expand data/company_sentiment_examples.csv with aged-care–specific
examples across multiple labels. Produces ~200+ additional rows using templates.
"""
import csv
from pathlib import Path

CSV_PATH = Path("data/company_sentiment_examples.csv")


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({"id": int(row["id"]), "label": row["label"], "text": row["text"]})
    return rows


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "text"]) 
        w.writeheader()
        for row in rows:
            w.writerow({"id": row["id"], "label": row["label"], "text": row["text"]})


def main():
    rows = read_rows(CSV_PATH)
    next_id = max(r["id"] for r in rows) + 1 if rows else 1

    # Templated expansions per label (aged-care context)
    frustration_templates = [
        "No one came during the window; we waited with my {relation} all afternoon.",
        "I called twice yesterday about {need}, still no reply.",
        "We were promised a call this morning about {need}, but nothing came.",
        "I keep repeating the same details for my {relation} and nothing happens.",
        "The nurse said they'd email a plan; it never arrived.",
        "We rearranged appointments, and yet no carer attended.",
        "I've been on hold multiple times and then disconnected.",
        "It’s been days with no update on equipment delivery.",
        "You said someone would check on {relation}—no one did.",
        "Why must I chase simple updates every time?",
    ]

    leave_templates = [
        "We’re moving to another provider who can support my {relation} properly.",
        "Please cancel our service; we’ve found alternative care.",
        "We’ve decided to discontinue; send closure confirmation.",
        "We’ll start looking elsewhere if this doesn’t improve immediately.",
        "End the contract—this hasn’t met our needs.",
    ]

    risk_templates = [
        "{Relation} missed medication due to no carer visit—safety concern.",
        "Wheelchair battery isn't holding charge; mobility risk.",
        "Bathroom has no grab rails; increased fall risk.",
        "{Relation} reported chest discomfort; needs nurse review.",
        "Heating failed overnight; {relation} was in the cold—wellbeing risk.",
        "Two different staff left conflicting instructions in the notes.",
        "Stair lift stalled midway; hazard.",
        "No interpreter provided despite language barrier—communication risk.",
        "Transport delay caused missed specialist appointment.",
        "Care plan steps not followed; {relation} condition worsening.",
    ]

    anger_templates = [
        "This is unacceptable—I’ve asked repeatedly and nothing changes.",
        "Stop passing me around and fix this for my {relation}.",
        "You promised support and didn’t deliver—again.",
        "How many times must I call to get basic help?",
        "I’m angry this keeps happening; take responsibility.",
    ]

    urgency_templates = [
        "We need someone tonight; {relation} can’t manage alone.",
        "Immediate callback required—{relation} is unwell.",
        "Urgent equipment delivery needed today.",
        "Please arrange same‑day transport to the clinic.",
        "This cannot wait until tomorrow; act now.",
    ]

    escalation_templates = [
        "Open a formal complaint and escalate to a supervisor.",
        "I want the complaints procedure emailed to me now.",
        "Escalate to management and confirm in writing.",
        "Provide a reference number for this escalation.",
        "Escalate priority and update me hourly.",
    ]

    compliance_templates = [
        "Personal details were sent in plain email—privacy incident.",
        "Photo taken without consent—policy breach.",
        "Paper notes misplaced; sensitive info at risk.",
        "No consent recorded for sharing medical information.",
        "Identifiers read aloud in public—confidentiality breach.",
    ]

    safety_templates = [
        "{Relation} had two near-falls this week—risk assessment needed.",
        "Front steps are icy and ungritted; hazard.",
        "No night check‑ins; {relation} felt unsafe.",
        "Smoke detector battery low—safety risk.",
        "Door lock faulty—premises unsecured.",
    ]

    finance_templates = [
        "We can’t afford the increased fees; need options.",
        "Invoice is incorrect; cannot pay this amount.",
        "Funding delayed; services at risk of interruption.",
        "Please set up a payment plan; we’re struggling.",
        "Unexpected fees appeared; need a refund review.",
    ]

    confusion_templates = [
        "I’m not sure who to call for scheduling.",
        "We received two appointment times; which is correct?",
        "Care plan isn’t clear—please explain next steps.",
        "Do you arrange transport or do we?",
        "We’re getting mixed information from different staff.",
    ]

    # Domain tokens
    rels = ["mother", "father", "nan", "granddad", "husband", "wife", "neighbor", "client"]

    def expand(templates, label):
        nonlocal next_id
        out = []
        # Mix templates with relation substitutions where applicable
        for t in templates:
            if "{relation}" in t or "{Relation}" in t:
                for r in rels[:4]:  # keep controlled expansion
                    text = t.replace("{relation}", r).replace("{Relation}", r.capitalize())
                    out.append({"id": next_id, "label": label, "text": text})
                    next_id += 1
            else:
                out.append({"id": next_id, "label": label, "text": t})
                next_id += 1
        return out

    additions = []
    additions += expand(frustration_templates, "frustration")
    additions += expand(leave_templates, "client_wants_to_leave")
    additions += expand(risk_templates, "risk_issue")
    additions += expand(anger_templates, "anger")
    additions += expand(urgency_templates, "urgency")
    additions += expand(escalation_templates, "escalation")
    additions += expand(compliance_templates, "compliance_privacy")
    additions += expand(safety_templates, "safety_wellbeing")
    additions += expand(finance_templates, "financial_distress")
    additions += expand(confusion_templates, "confusion")

    # Also add short fragments common to aged-care calls (implicitly labeled)
    fragment_map = {
        "frustration": [
            "still waiting on the nurse", "no one came", "no call back again",
            "window missed", "tired of chasing"
        ],
        "risk_issue": [
            "no rails in bathroom", "wheelchair battery flat", "missed meds", "heater broken"
        ],
        "urgency": [
            "need help tonight", "urgent visit please", "call within the hour"
        ],
    }
    for label, frags in fragment_map.items():
        for frag in frags:
            additions.append({"id": next_id, "label": label, "text": frag})
            next_id += 1

    rows.extend(additions)
    write_rows(CSV_PATH, rows)
    print(f"Appended {len(additions)} rows. Total: {len(rows)}")


if __name__ == "__main__":
    main()



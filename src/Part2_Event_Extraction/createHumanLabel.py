"""
Human Labeling Tool for Cohen's Kappa
=====================================
Run this AFTER Part 2, BEFORE Part 3

This tool shows you extracted claims and asks YOU to label them.
Your labels become the "ground truth" for validating the LLM Judge.
"""

import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION - Update these paths to match your setup
# =============================================================================
DATA_DIR = Path(__file__).parent.parent.parent / 'src' / 'Part2_Event_Extraction' / 'data' / 'extractions'
OUTPUT_FILE = Path(__file__).parent / "data" / "human_labels.json"


def load_extractions():
    """Load Lincoln and Other author extractions from Part 2"""
    lincoln_path = DATA_DIR / "extractions_lincoln.json"
    others_path = DATA_DIR / "extractions_others.json"
    
    if not lincoln_path.exists():
        print(f"❌ File not found: {lincoln_path}")
        print("   Run Part 2 first!")
        return None, None
    
    if not others_path.exists():
        print(f"❌ File not found: {others_path}")
        print("   Run Part 2 first!")
        return None, None
    
    with open(lincoln_path, 'r', encoding='utf-8') as f:
        lincoln = json.load(f)
    with open(others_path, 'r', encoding='utf-8') as f:
        others = json.load(f)
    
    return lincoln, others


def create_pairs_for_labeling(lincoln_data, others_data):
    """Create (Lincoln, Other) pairs for each event"""
    
    # Group Lincoln claims by event
    lincoln_by_event = {}
    for item in lincoln_data:
        event = item.get("event", "")
        if event and item.get("claims"):
            if event not in lincoln_by_event:
                lincoln_by_event[event] = []
            lincoln_by_event[event].append(item)
    
    # Group Other claims by event
    others_by_event = {}
    for item in others_data:
        event = item.get("event", "")
        if event and item.get("claims"):
            if event not in others_by_event:
                others_by_event[event] = []
            others_by_event[event].append(item)
    
    # Create pairs
    pairs = []
    for event in lincoln_by_event:
        if event not in others_by_event:
            continue
        
        for lincoln_item in lincoln_by_event[event]:
            for other_item in others_by_event[event]:
                pair_id = f"{event}_{other_item.get('source_id', 'unknown')}"
                pairs.append({
                    "pair_id": pair_id,
                    "event": event,
                    "event_name": lincoln_item.get("event_name", event),
                    "lincoln_source": lincoln_item.get("source_id", "unknown"),
                    "lincoln_claims": lincoln_item.get("claims", []),
                    "lincoln_quotes": lincoln_item.get("quotes", []),
                    "other_source": other_item.get("source_id", "unknown"),
                    "other_author": other_item.get("author", "Unknown"),
                    "other_claims": other_item.get("claims", []),
                    "other_quotes": other_item.get("quotes", [])
                })
    
    return pairs


def display_pair(pair, index, total):
    """Display a claim pair for human review"""
    print("\n" + "=" * 70)
    print(f"PAIR {index + 1} of {total}")
    print("=" * 70)
    print(f"EVENT: {pair['event_name']}")
    print(f"PAIR ID: {pair['pair_id']}")
    print("-" * 70)
    
    print("\n🔵 LINCOLN'S ACCOUNT (First-Person):")
    print(f"   Source: {pair['lincoln_source']}")
    if pair['lincoln_claims']:
        for i, claim in enumerate(pair['lincoln_claims'], 1):
            print(f"   {i}. {claim}")
    else:
        print("   [No claims extracted]")
    
    if pair['lincoln_quotes']:
        print("\n   Direct quotes:")
        for q in pair['lincoln_quotes'][:2]:  # Show max 2 quotes
            print(f"   \"{q[:100]}...\"" if len(q) > 100 else f"   \"{q}\"")
    
    print("\n🟢 OTHER AUTHOR'S ACCOUNT (Third-Person):")
    print(f"   Author: {pair['other_author']}")
    print(f"   Source: {pair['other_source']}")
    if pair['other_claims']:
        for i, claim in enumerate(pair['other_claims'], 1):
            print(f"   {i}. {claim}")
    else:
        print("   [No claims extracted]")
    
    if pair['other_quotes']:
        print("\n   Direct quotes:")
        for q in pair['other_quotes'][:2]:
            print(f"   \"{q[:100]}...\"" if len(q) > 100 else f"   \"{q}\"")
    
    print("-" * 70)


def get_human_label():
    """Get human judgment for a pair"""
    print("\n📋 YOUR JUDGMENT:")
    print("   [C] Consistent - Claims generally align, no major contradictions")
    print("   [X] Contradictory - Claims conflict on facts, dates, or interpretation")
    print("   [S] Skip - Not enough information to judge")
    print("   [Q] Quit - Save progress and exit")
    
    while True:
        choice = input("\n   Your choice [C/X/S/Q]: ").strip().upper()
        
        if choice == 'C':
            return "consistent"
        elif choice == 'X':
            return "contradictory"
        elif choice == 'S':
            return "skip"
        elif choice == 'Q':
            return "quit"
        else:
            print("   Invalid choice. Please enter C, X, S, or Q.")


def save_labels(labels, output_path):
    """Save human labels to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_labeled": len([l for l in labels.values() if l != "skip"]),
            "total_skipped": len([l for l in labels.values() if l == "skip"]),
            "consistent_count": len([l for l in labels.values() if l == "consistent"]),
            "contradictory_count": len([l for l in labels.values() if l == "contradictory"])
        },
        "labels": labels
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Labels saved to: {output_path}")
    return output_data


def load_existing_labels(output_path):
    """Load any existing labels to continue from"""
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("labels", {})
    return {}


def main():
    print("=" * 70)
    print("HUMAN LABELING TOOL FOR COHEN'S KAPPA")
    print("=" * 70)
    print("\nThis tool helps you create ground-truth labels for LLM Judge validation.")
    print("You will review extracted claims and label each pair as:")
    print("  • CONSISTENT - Claims align")
    print("  • CONTRADICTORY - Claims conflict")
    print("\nYour labels will be compared against the LLM Judge's labels")
    print("to calculate Cohen's Kappa (human-AI agreement score).")
    
    # Load data
    print("\n📂 Loading extractions from Part 2...")
    lincoln_data, others_data = load_extractions()
    
    if lincoln_data is None or others_data is None:
        return 1
    
    # Create pairs
    pairs = create_pairs_for_labeling(lincoln_data, others_data)
    print(f"✓ Found {len(pairs)} claim pairs to label")
    
    if not pairs:
        print("❌ No valid pairs found. Check your Part 2 output.")
        return 1
    
    # Load existing labels (to continue previous session)
    existing_labels = load_existing_labels(OUTPUT_FILE)
    if existing_labels:
        print(f"✓ Found {len(existing_labels)} existing labels (will continue from there)")
    
    # Filter out already-labeled pairs
    unlabeled_pairs = [p for p in pairs if p['pair_id'] not in existing_labels]
    print(f"✓ {len(unlabeled_pairs)} pairs remaining to label")
    
    # Recommendation for assessment
    min_labels_needed = 10
    print(f"\n⚠️  Assessment requires at least {min_labels_needed} labeled pairs for valid Kappa.")
    
    if len(existing_labels) >= min_labels_needed:
        print(f"   You already have {len(existing_labels)} labels. You can proceed to Part 3!")
        choice = input("   Continue labeling more? [y/N]: ").strip().lower()
        if choice != 'y':
            return 0
    
    # Start labeling
    labels = existing_labels.copy()
    
    print("\n" + "=" * 70)
    print("STARTING LABELING SESSION")
    print("=" * 70)
    
    for i, pair in enumerate(unlabeled_pairs):
        display_pair(pair, i, len(unlabeled_pairs))
        
        label = get_human_label()
        
        if label == "quit":
            print("\n⏸️  Saving progress and exiting...")
            break
        
        labels[pair['pair_id']] = label
        
        # Auto-save every 5 labels
        if (i + 1) % 5 == 0:
            save_labels(labels, OUTPUT_FILE)
            print(f"   [Auto-saved {len(labels)} labels]")
    
    # Final save
    output_data = save_labels(labels, OUTPUT_FILE)
    
    # Summary
    print("\n" + "=" * 70)
    print("LABELING SUMMARY")
    print("=" * 70)
    print(f"Total labeled: {output_data['metadata']['total_labeled']}")
    print(f"  • Consistent: {output_data['metadata']['consistent_count']}")
    print(f"  • Contradictory: {output_data['metadata']['contradictory_count']}")
    print(f"  • Skipped: {output_data['metadata']['total_skipped']}")
    
    if output_data['metadata']['total_labeled'] >= min_labels_needed:
        print(f"\n✅ You have enough labels for Cohen's Kappa! Run Part 3 now.")
    else:
        remaining = min_labels_needed - output_data['metadata']['total_labeled']
        print(f"\n⚠️  Label {remaining} more pairs for valid Kappa calculation.")
    
    print(f"\n📁 Labels saved to: {OUTPUT_FILE}")
    print("   Part 3 will automatically load these labels.")
    
    return 0


if __name__ == "__main__":
    exit(main())
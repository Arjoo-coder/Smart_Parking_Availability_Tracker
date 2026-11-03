"""Smart parking recommendations.

What is ranked
--------------
The model predicts how many spaces will be free at each candidate parking lot at the
chosen time. The recommendation is the lot with the highest predicted availability.

What is NOT known
-----------------
The dataset has no distances, coordinates or travel times. So, unless you supply
an optional distance file (data/raw/lot_distances.csv, see README), the ranking is
based on parking availability only and NOT on how far each lot is from you. The
functions below always return a note that says which of the two cases applies.
"""

import numpy as np
import pandas as pd

from src import config
from src.predict import lot_supports_events, predict_many, predict_slots

NO_DISTANCE_NOTE = ("Recommendations are based on predicted parking availability only. The dataset has "
                    "no travel-distance information, so the ranking does not consider how far each "
                    "location is from your destination.")
RANK_OPTIONS = {"free_spaces": "Most free spaces", "free_share": "Highest share of free spaces"}


# ----------------------------------------------------------------------------
# Optional distance data
# ----------------------------------------------------------------------------
def load_distances(path=config.DISTANCE_FILE):
    """Read the optional distance file. Returns (DataFrame or None, message).

    Expected columns: destination (a location name from the app), parking_lot_id,
    distance_km. Rows with problems are skipped and reported in the message.
    """
    if not path.exists():
        return None, NO_DISTANCE_NOTE
    try:
        table = pd.read_csv(path)
    except Exception as error:
        return None, f"The distance file could not be read ({error}). " + NO_DISTANCE_NOTE
    missing = [c for c in ("destination", "parking_lot_id", "distance_km") if c not in table.columns]
    if missing:
        return None, (f"The distance file is missing the column(s) {', '.join(missing)}. " + NO_DISTANCE_NOTE)

    table = table.copy()
    table["distance_km"] = pd.to_numeric(table["distance_km"], errors="coerce")
    table["parking_lot_id"] = table["parking_lot_id"].astype(str).str.strip().str.upper()
    table["destination"] = table["destination"].astype(str).str.strip()
    locations = {info["location"] for info in config.PARKING_LOTS.values()}
    good = (table["distance_km"].notna() & (table["distance_km"] >= 0)
            & table["parking_lot_id"].isin(config.PARKING_LOTS) & table["destination"].isin(locations))
    skipped = int((~good).sum())
    table = table[good].drop_duplicates(["destination", "parking_lot_id"])
    if table.empty:
        return None, "The distance file has no usable rows. " + NO_DISTANCE_NOTE
    note = "Distances from your distance file are used together with availability."
    if skipped:
        note += f" ({skipped} unusable row(s) were skipped.)"
    return table.reset_index(drop=True), note


# ----------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------
def recommend(lot_ids, when, weather=None, event_today=False, model=None, destination=None,
              distances=None, distance_weight=0.3, rank_by="free_spaces"):
    """Rank the candidate lots for one arrival time.

    lot_ids         - the lots the user would accept (their preferred locations)
    destination     - optional location NAME the user is going to
    distances       - DataFrame from load_distances() or None
    distance_weight - 0..1, how much distance counts compared with availability
                      (only used when `distances` has data for the destination)
    rank_by         - "free_spaces" (default) or "free_share"
    Returns a dictionary: table, best, headline, notes, uses_distance.
    """
    lot_ids = [str(lot).strip().upper() for lot in (lot_ids or [])]
    if not lot_ids:
        raise ValueError("Choose at least one parking location to compare.")
    unknown = [lot for lot in lot_ids if lot not in config.PARKING_LOTS]
    if unknown:
        raise ValueError(f"Unknown parking location(s): {', '.join(unknown)}.")
    if rank_by not in RANK_OPTIONS:
        raise ValueError(f"rank_by must be one of {list(RANK_OPTIONS)}.")
    if not 0.0 <= float(distance_weight) <= 1.0:
        raise ValueError("distance_weight must be between 0 and 1.")
    lot_ids = list(dict.fromkeys(lot_ids))          # remove duplicates, keep order

    requests = [{"parking_lot_id": lot, "timestamp": when, "weather": weather,
                 "event_today": bool(event_today) and lot_supports_events(lot)} for lot in lot_ids]
    table = predict_many(requests, model=model)
    table["free_share"] = table["predicted_available_spaces"] / table["total_spaces"]
    table["is_full"] = table["status"] == "Full"
    table["at_destination"] = table["location"] == destination if destination else False

    notes = []
    uses_distance = False
    if distances is not None and destination:
        near = distances[distances["destination"] == destination].set_index("parking_lot_id")["distance_km"]
        table["distance_km"] = table["parking_lot_id"].map(near)
        if table["distance_km"].notna().all():
            uses_distance = True
        else:
            table = table.drop(columns="distance_km")
            notes.append("The distance file has no distances from this destination to every selected "
                         "location, so distance was ignored. " + NO_DISTANCE_NOTE)
    elif distances is not None and not destination:
        notes.append("Distances are available, but no destination was chosen, so only availability is used.")
    else:
        notes.append(NO_DISTANCE_NOTE)

    availability = table["predicted_available_spaces"] if rank_by == "free_spaces" else table["free_share"]
    top = availability.max()
    table["availability_score"] = availability / top if top > 0 else 0.0
    if uses_distance:
        far = table["distance_km"].max()
        closeness = 1.0 - (table["distance_km"] / far if far > 0 else 0.0)
        w = float(distance_weight)
        table["score"] = (1 - w) * table["availability_score"] + w * closeness
    else:
        table["score"] = table["availability_score"]

    table = (table.sort_values(["is_full", "score", "parking_lot_id"], ascending=[True, False, True])
                  .reset_index(drop=True))
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    best = table.iloc[0].to_dict() if not table.iloc[0]["is_full"] else None
    table["recommended"] = (table.index == 0) & (not table.iloc[0]["is_full"])
    table["reason"] = [_reason(row, uses_distance, rank_by) for row in table.itertuples()]

    if best is None:
        headline = ("All selected locations are predicted to be Full at this time. "
                    "Consider another time slot or more locations.")
    else:
        headline = (f"Best choice: {best['location']} with about {best['predicted_available_spaces']} free "
                    f"of {best['total_spaces']} spaces ({best['predicted_occupancy_rate'] * 100:.0f}% full).")
    if destination:
        mine = table[table["at_destination"]]
        if mine.empty:
            notes.append(f"Your destination ({destination}) is not among the selected locations.")
        else:
            row = mine.iloc[0]
            advice = {"Available": "parking at your destination looks fine.",
                      "Limited": "parking at your destination may be tight, so consider arriving earlier "
                                 "or check the alternatives.",
                      "Full": "your destination's parking is likely full, so use an alternative."}[row["status"]]
            notes.insert(0, f"Your destination ({destination}): about {row['predicted_available_spaces']} free "
                            f"({row['status']}), {advice}")
    notes.append("These are estimates from a statistical model, not guaranteed real-time availability.")
    return {"table": table, "best": best, "headline": headline, "notes": notes,
            "uses_distance": uses_distance, "rank_by": rank_by}


def _reason(row, uses_distance, rank_by):
    if row.is_full:
        return "Predicted Full - not recommended"
    parts = []
    if row.rank == 1:
        parts.append("Highest predicted availability" if not uses_distance
                     else "Best balance of availability and distance")
    if row.at_destination:
        parts.append("At your destination")
    if uses_distance:
        parts.append(f"{row.distance_km:.1f} km away")
    if row.status == "Limited":
        parts.append("Limited - may fill up")
    return "; ".join(parts) if parts else "Available alternative"


# ----------------------------------------------------------------------------
# Time slots: when is the best time to arrive?
# ----------------------------------------------------------------------------
def best_time_to_arrive(lot_id, start, hours, weather=None, event_today=False, model=None):
    """For ONE lot, look at the next `hours` slots. Returns a dictionary:
    slots (DataFrame), best (the slot with the most free spaces), worst (the fewest),
    first_available (first slot not Full, or None)."""
    slots = predict_slots([lot_id], start, hours, weather, event_today, model=model)
    best = slots.sort_values(["predicted_available_spaces", "slot"], ascending=[False, True]).iloc[0]
    worst = slots.sort_values(["predicted_available_spaces", "slot"], ascending=[True, True]).iloc[0]
    open_slots = slots[slots["status"] != "Full"]
    return {"slots": slots, "best": best.to_dict(), "worst": worst.to_dict(),
            "first_available": open_slots.iloc[0].to_dict() if not open_slots.empty else None}

"""
Generator for airline multi-step cross-tool dependency tasks.

Produces tasks_multistep.json by mining the existing db.json for real scenarios.
Each task type forces the agent to propagate intermediate values (reservation_id,
flight_number, etc.) across sequential tool calls — the user is explicitly told
not to know those intermediate IDs.

Run:
    conda run -n tau2 python -m tau2.domains.airline.tasks.create_multistep_tasks
"""

import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import (
    Action,
    Description,
    EvaluationCriteria,
    RewardType,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.domains.airline.data_model import FlightDB, Reservation, User
from tau2.domains.airline.utils import AIRLINE_DATA_DIR, AIRLINE_DB_PATH

MULTISTEP_TASK_SET_PATH = AIRLINE_DATA_DIR / "tasks_multistep.json"
MULTISTEP_SPLIT_PATH = AIRLINE_DATA_DIR / "split_tasks_multistep.json"

# Reference datetime used by _get_datetime() in tools.py
TODAY = "2024-05-15"
# First date with available flights (TODAY's flights are all in-flight/landed)
BOOKING_DATE = "2024-05-16"

# Free bags per passenger: [regular, silver, gold] indexed by membership
_FREE_BAGS = {
    "basic_economy": {"regular": 0, "silver": 1, "gold": 2},
    "economy":       {"regular": 1, "silver": 2, "gold": 3},
    "business":      {"regular": 2, "silver": 3, "gold": 4},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _most_recent_reservation(user: User, db: FlightDB) -> Optional[Reservation]:
    """Return the last reservation in the user's list that exists in the DB."""
    for rid in reversed(user.reservations):
        if rid in db.reservations:
            res = db.reservations[rid]
            if res.status != "cancelled":
                return res
    return None


def _flight_status(db: FlightDB, flight_number: str, date: str) -> Optional[str]:
    if flight_number not in db.flights:
        return None
    flight = db.flights[flight_number]
    if date not in flight.dates:
        return None
    return flight.dates[date].status


def _is_compensation_eligible(user: User, reservation: Reservation) -> bool:
    return user.membership in ("silver", "gold") or reservation.insurance == "yes"


def _free_bags_total(membership: str, cabin: str, n_passengers: int) -> int:
    return _FREE_BAGS.get(cabin, {}).get(membership, 0) * n_passengers


def _cheapest_direct_flight(db: FlightDB, origin: str, destination: str, date: str):
    """Return (flight_number, economy_price) for the cheapest available economy flight."""
    best = None
    for flight in db.flights.values():
        if flight.origin != origin or flight.destination != destination:
            continue
        if date not in flight.dates:
            continue
        fds = flight.dates[date]
        if fds.status != "available":
            continue
        price = fds.prices.get("economy")
        if price is None:
            continue
        if best is None or price < best[1]:
            best = (flight.flight_number, price)
    return best


def _credit_card_id(user: User) -> Optional[str]:
    for pid, pm in user.payment_methods.items():
        if pm.source == "credit_card":
            return pid
    return None


def _non_certificate_payment_id(user: User) -> Optional[str]:
    """Return first gift_card or credit_card payment ID."""
    for pid, pm in user.payment_methods.items():
        if pm.source in ("credit_card", "gift_card"):
            return pid
    return None


def _make_action(task_id: str, seq: int, name: str, arguments: dict,
                 compare_args: Optional[list[str]] = None) -> Action:
    return Action(
        action_id=f"{task_id}_{seq}",
        requestor="assistant",
        name=name,
        arguments=arguments,
        compare_args=compare_args,
    )


def _make_task(
    task_id: str,
    purpose: str,
    reason_for_call: str,
    known_info: str,
    unknown_info: str,
    task_instructions: str,
    actions: list[Action],
    communicate_info: list[str],
    nl_assertions: list[str],
    reward_basis: list[RewardType],
) -> Task:
    return Task(
        id=task_id,
        description=Description(purpose=purpose),
        user_scenario=UserScenario(
            persona=None,
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call=reason_for_call,
                known_info=known_info,
                unknown_info=unknown_info,
                task_instructions=task_instructions,
            ),
        ),
        ticket=None,
        initial_state=None,
        evaluation_criteria=EvaluationCriteria(
            actions=actions,
            communicate_info=communicate_info,
            nl_assertions=nl_assertions,
            reward_basis=reward_basis,
        ),
    )


# ── Type A: Flight Status Discovery ──────────────────────────────────────────

def _generate_type_a(db: FlightDB, max_tasks: int = 3) -> list[Task]:
    """
    3-step chain: get_user_details → get_reservation_details → get_flight_status
    User knows only their user_id; must discover reservation_id and flight_number.
    """
    tasks = []
    for user in db.users.values():
        if len(tasks) >= max_tasks:
            break
        res = _most_recent_reservation(user, db)
        if res is None or not res.flights:
            continue
        flight_info = res.flights[0]
        status = _flight_status(db, flight_info.flight_number, flight_info.date)
        if status is None:
            continue

        tid = f"ms_a_{len(tasks)}"
        full_name = f"{user.name.first_name} {user.name.last_name}"

        actions = [
            _make_action(tid, 0, "get_user_details",
                         {"user_id": user.user_id}),
            _make_action(tid, 1, "get_reservation_details",
                         {"reservation_id": res.reservation_id}),
            _make_action(tid, 2, "get_flight_status",
                         {"flight_number": flight_info.flight_number,
                          "date": flight_info.date}),
        ]

        tasks.append(_make_task(
            task_id=tid,
            purpose=(
                "3-step dependency chain: user_id → reservation_id → flight_number → status. "
                "Agent must discover reservation and flight via tool calls, not user context."
            ),
            reason_for_call=(
                f"I'd like to know the status of the flight on my most recent reservation."
            ),
            known_info=(
                f"You are {full_name}. Your user id is {user.user_id}."
            ),
            unknown_info=(
                "You do not know your reservation ID or flight number. "
                "Do not provide them — let the agent look them up."
            ),
            task_instructions=(
                "Ask the agent for the flight status on your most recent reservation. "
                "Do not volunteer your reservation ID or flight number. "
                "If the agent asks for your reservation ID, say you don't have it handy "
                "and ask them to look it up from your account."
            ),
            actions=actions,
            communicate_info=[status],
            nl_assertions=[
                f"Agent correctly reports the flight status as '{status}'.",
                "Agent must look up the reservation from the user account, not ask the user for the reservation ID.",
            ],
            reward_basis=[RewardType.ACTION, RewardType.COMMUNICATE],
        ))
    return tasks


# ── Type B: Cancellation Compensation ────────────────────────────────────────

def _generate_type_b(db: FlightDB, max_tasks: int = 3) -> list[Task]:
    """
    4-step chain: get_user_details → get_reservation_details → get_flight_status
                  → send_certificate
    Requires: user eligible for compensation + first flight on reservation is
    cancelled or delayed.
    """
    tasks = []
    for user in db.users.values():
        if len(tasks) >= max_tasks:
            break
        res = _most_recent_reservation(user, db)
        if res is None or not res.flights:
            continue
        if not _is_compensation_eligible(user, res):
            continue

        flight_info = res.flights[0]
        status = _flight_status(db, flight_info.flight_number, flight_info.date)
        if status not in ("cancelled", "delayed"):
            continue

        n_passengers = len(res.passengers)
        amount = (100 if status == "cancelled" else 50) * n_passengers
        tid = f"ms_b_{len(tasks)}"
        full_name = f"{user.name.first_name} {user.name.last_name}"

        actions = [
            _make_action(tid, 0, "get_user_details",
                         {"user_id": user.user_id}),
            _make_action(tid, 1, "get_reservation_details",
                         {"reservation_id": res.reservation_id}),
            _make_action(tid, 2, "get_flight_status",
                         {"flight_number": flight_info.flight_number,
                          "date": flight_info.date}),
            _make_action(tid, 3, "send_certificate",
                         {"user_id": user.user_id, "amount": amount}),
        ]

        tasks.append(_make_task(
            task_id=tid,
            purpose=(
                f"4-step dependency chain: user_id → reservation_id → "
                f"flight_number → send_certificate({amount}). "
                f"Flight is {status}; compensation = ${amount // n_passengers} × "
                f"{n_passengers} passenger(s)."
            ),
            reason_for_call=(
                "My most recent flight was disrupted and I would like compensation."
            ),
            known_info=(
                f"You are {full_name}. Your user id is {user.user_id}."
            ),
            unknown_info=(
                "You do not know your reservation ID or flight number. "
                "Do not provide them — let the agent look them up."
            ),
            task_instructions=(
                "Tell the agent your most recent flight was disrupted and you want compensation. "
                "Do not volunteer your reservation ID or flight number. "
                "If the agent asks, say you don't have it on hand. "
                "Accept the compensation certificate once the agent offers it."
            ),
            actions=actions,
            communicate_info=[],
            nl_assertions=[
                f"Agent verifies the flight status is '{status}' via tool call.",
                f"Agent issues a certificate of ${amount} (${amount // n_passengers} × {n_passengers} passenger(s)).",
                "Agent must not issue compensation without first verifying the flight status.",
            ],
            reward_basis=[RewardType.ACTION],
        ))
    return tasks


# ── Type C: Search-then-Book ──────────────────────────────────────────────────

def _generate_type_c(db: FlightDB, max_tasks: int = 3) -> list[Task]:
    """
    3-step chain: search_direct_flight → get_user_details → book_reservation
    The flight_number in book_reservation must come from the search results.
    """
    tasks = []
    seen_routes: set[tuple] = set()

    for user in db.users.values():
        if len(tasks) >= max_tasks:
            break
        cc_id = _credit_card_id(user)
        if cc_id is None:
            continue
        if not user.saved_passengers:
            continue

        # Find a route with available economy on BOOKING_DATE
        for flight in db.flights.values():
            if (flight.origin, flight.destination) in seen_routes:
                continue
            result = _cheapest_direct_flight(
                db, flight.origin, flight.destination, BOOKING_DATE
            )
            if result is None:
                continue
            flight_number, economy_price = result
            seen_routes.add((flight.origin, flight.destination))

            tid = f"ms_c_{len(tasks)}"
            full_name = f"{user.name.first_name} {user.name.last_name}"
            passenger = user.saved_passengers[0]
            total_price = economy_price  # 1 passenger, no insurance, no extra bags

            actions = [
                _make_action(tid, 0, "search_direct_flight",
                             {"origin": flight.origin,
                              "destination": flight.destination,
                              "date": BOOKING_DATE}),
                _make_action(tid, 1, "get_user_details",
                             {"user_id": user.user_id}),
                _make_action(tid, 2, "book_reservation",
                             {
                                 "user_id": user.user_id,
                                 "origin": flight.origin,
                                 "destination": flight.destination,
                                 "flight_type": "one_way",
                                 "cabin": "economy",
                                 "flights": [{"flight_number": flight_number,
                                              "date": BOOKING_DATE}],
                                 "passengers": [{"first_name": passenger.first_name,
                                                 "last_name": passenger.last_name,
                                                 "dob": passenger.dob}],
                                 "payment_methods": [{"payment_id": cc_id,
                                                      "amount": total_price}],
                                 "total_baggages": 1,
                                 "nonfree_baggages": 0,
                                 "insurance": "no",
                             },
                             compare_args=["flights"]),
            ]

            tasks.append(_make_task(
                task_id=tid,
                purpose=(
                    f"3-step dependency chain: search_direct_flight → get_user_details → "
                    f"book_reservation. Flight number in book_reservation must come from "
                    f"search results, not fabricated. Route: {flight.origin}→{flight.destination}."
                ),
                reason_for_call=(
                    f"I'd like to book the cheapest economy flight from "
                    f"{flight.origin} to {flight.destination} on {BOOKING_DATE}."
                ),
                known_info=(
                    f"You are {full_name}. Your user id is {user.user_id}. "
                    f"You want to fly from {flight.origin} to {flight.destination} "
                    f"on {BOOKING_DATE} in economy class. "
                    f"You have 1 passenger: {passenger.first_name} {passenger.last_name} "
                    f"(DOB: {passenger.dob}). "
                    f"Pay with your credit card on file (payment id: {cc_id})."
                ),
                unknown_info=(
                    "You do not know the flight number. "
                    "The agent must search for available flights and pick the cheapest."
                ),
                task_instructions=(
                    f"Ask the agent to book the cheapest economy flight from "
                    f"{flight.origin} to {flight.destination} on {BOOKING_DATE} for 1 passenger. "
                    "Do not suggest a flight number — let the agent search. "
                    f"Confirm the booking once the agent presents the details."
                ),
                actions=actions,
                communicate_info=[],
                nl_assertions=[
                    "Agent searches for available flights before booking.",
                    "Agent books the cheapest economy option found in the search results.",
                ],
                reward_basis=[RewardType.ACTION],
            ))
            break  # one task per user

    return tasks


# ── Type D: Flight Update ─────────────────────────────────────────────────────

def _generate_type_d(db: FlightDB, max_tasks: int = 3) -> list[Task]:
    """
    3-step chain: get_reservation_details → search_direct_flight
                  → update_reservation_flights
    User knows reservation_id but not origin/destination/date/flight_number.
    """
    tasks = []
    for res in db.reservations.values():
        if len(tasks) >= max_tasks:
            break
        if res.status == "cancelled":
            continue
        if res.cabin == "business":
            continue  # already in business
        if not res.flights:
            continue

        flight_info = res.flights[0]
        # Check business is available on the same flight/date
        status = _flight_status(db, flight_info.flight_number, flight_info.date)
        if status != "available":
            continue
        fds = db.flights[flight_info.flight_number].dates[flight_info.date]
        if fds.available_seats.get("business", 0) < len(res.passengers):
            continue

        user = db.users.get(res.user_id)
        if user is None:
            continue
        cc_id = _credit_card_id(user)
        if cc_id is None:
            continue

        full_name = f"{user.name.first_name} {user.name.last_name}"
        tid = f"ms_d_{len(tasks)}"

        actions = [
            _make_action(tid, 0, "get_reservation_details",
                         {"reservation_id": res.reservation_id}),
            _make_action(tid, 1, "search_direct_flight",
                         {"origin": res.origin,
                          "destination": res.destination,
                          "date": flight_info.date}),
            _make_action(tid, 2, "update_reservation_flights",
                         {
                             "reservation_id": res.reservation_id,
                             "cabin": "business",
                             "flights": [{"flight_number": flight_info.flight_number,
                                          "date": flight_info.date}],
                             "payment_id": cc_id,
                         },
                         compare_args=["reservation_id", "cabin", "flights"]),
        ]

        tasks.append(_make_task(
            task_id=tid,
            purpose=(
                f"3-step dependency chain: get_reservation_details → "
                f"search_direct_flight → update_reservation_flights. "
                f"Agent must read origin/destination/date from reservation before searching. "
                f"Reservation: {res.reservation_id}."
            ),
            reason_for_call=(
                "I'd like to upgrade my reservation to business class."
            ),
            known_info=(
                f"You are {full_name}. Your user id is {user.user_id}. "
                f"Your reservation id is {res.reservation_id}. "
                f"You want to upgrade to business class. "
                f"Pay with your credit card on file (payment id: {cc_id})."
            ),
            unknown_info=(
                "You do not know the origin, destination, date, or flight number — "
                "let the agent read those from your reservation."
            ),
            task_instructions=(
                f"Ask the agent to upgrade reservation {res.reservation_id} to business class. "
                "Do not mention origin, destination, or flight number — the agent must "
                "look those up from your reservation details. "
                "Confirm the upgrade once the agent presents the price difference."
            ),
            actions=actions,
            communicate_info=[],
            nl_assertions=[
                "Agent reads the reservation details before searching for flights.",
                "Agent uses the origin/destination/date from the reservation when searching.",
                "Agent upgrades to business class on the same flight.",
            ],
            reward_basis=[RewardType.ACTION],
        ))
    return tasks


# ── Type E: Add Baggage ───────────────────────────────────────────────────────

def _generate_type_e(db: FlightDB, max_tasks: int = 3) -> list[Task]:
    """
    3-step chain: get_user_details → get_reservation_details
                  → update_reservation_baggages
    Free bag count depends on BOTH membership (from user_details) AND cabin
    (from reservation_details). Agent must read both before calculating cost.
    """
    tasks = []
    for user in db.users.values():
        if len(tasks) >= max_tasks:
            break
        payment_id = _non_certificate_payment_id(user)
        if payment_id is None:
            continue

        res = _most_recent_reservation(user, db)
        if res is None or not res.flights:
            continue

        # Only worth testing if adding 1 bag would cost money
        free_per_pax = _FREE_BAGS.get(res.cabin, {}).get(user.membership, 0)
        current_free_total = free_per_pax * len(res.passengers)
        current_total = res.total_baggages

        # We want to add 1 bag that crosses the free threshold
        # e.g. current total = free total, add 1 → nonfree = 1, cost = $50
        if current_total > current_free_total:
            # already paying for bags — skip (less interesting)
            continue

        new_total = current_total + 1
        new_nonfree = max(0, new_total - current_free_total)
        cost = 50 * new_nonfree

        full_name = f"{user.name.first_name} {user.name.last_name}"
        tid = f"ms_e_{len(tasks)}"

        actions = [
            _make_action(tid, 0, "get_user_details",
                         {"user_id": user.user_id}),
            _make_action(tid, 1, "get_reservation_details",
                         {"reservation_id": res.reservation_id}),
            _make_action(tid, 2, "update_reservation_baggages",
                         {
                             "reservation_id": res.reservation_id,
                             "total_baggages": new_total,
                             "nonfree_baggages": new_nonfree,
                             "payment_id": payment_id,
                         }),
        ]

        communicate = [str(cost)] if cost > 0 else []

        tasks.append(_make_task(
            task_id=tid,
            purpose=(
                f"3-step dependency chain: get_user_details (membership={user.membership}) "
                f"→ get_reservation_details (cabin={res.cabin}, "
                f"passengers={len(res.passengers)}) → update_reservation_baggages. "
                f"Free bag calculation requires both membership and cabin. Cost: ${cost}."
            ),
            reason_for_call=(
                "I'd like to add one more checked bag to my most recent reservation."
            ),
            known_info=(
                f"You are {full_name}. Your user id is {user.user_id}. "
                f"You want to add 1 checked bag to your most recent reservation. "
                f"Pay with payment method {payment_id} if there is a charge."
            ),
            unknown_info=(
                "You do not know your reservation ID. "
                "Do not provide it — let the agent look it up."
            ),
            task_instructions=(
                "Ask the agent to add one checked bag to your most recent reservation. "
                "Do not volunteer your reservation ID — the agent must look it up. "
                f"If asked about payment, use {payment_id}. "
                "Confirm once the agent presents the updated baggage details."
            ),
            actions=actions,
            communicate_info=communicate,
            nl_assertions=[
                f"Agent checks user membership ({user.membership}) and reservation cabin ({res.cabin}) before calculating the cost.",
                f"Agent correctly calculates {free_per_pax} free bag(s) per passenger "
                f"({current_free_total} free total for {len(res.passengers)} passenger(s)).",
                f"Agent reports the cost as ${cost}." if cost > 0 else "Agent correctly notes the bag is free.",
            ],
            reward_basis=(
                [RewardType.ACTION, RewardType.COMMUNICATE]
                if cost > 0
                else [RewardType.ACTION]
            ),
        ))
    return tasks


# ── Orchestration ─────────────────────────────────────────────────────────────

def create_multistep_tasks(save: bool = True, seed: int = 42) -> list[Task]:
    random.seed(seed)
    db = FlightDB.load(AIRLINE_DB_PATH)

    tasks: list[Task] = []
    type_a = _generate_type_a(db)
    type_b = _generate_type_b(db)
    type_c = _generate_type_c(db)
    type_d = _generate_type_d(db)
    type_e = _generate_type_e(db)

    print(f"Type A (Flight Status Discovery):   {len(type_a)} tasks")
    print(f"Type B (Cancellation Compensation): {len(type_b)} tasks")
    print(f"Type C (Search-then-Book):          {len(type_c)} tasks")
    print(f"Type D (Flight Update):             {len(type_d)} tasks")
    print(f"Type E (Add Baggage):               {len(type_e)} tasks")

    tasks = type_a + type_b + type_c + type_d + type_e
    print(f"Total:                              {len(tasks)} tasks")

    if save:
        AIRLINE_DATA_DIR.mkdir(parents=True, exist_ok=True)

        with open(MULTISTEP_TASK_SET_PATH, "w") as f:
            json.dump([t.model_dump() for t in tasks], f, indent=2)
        print(f"Saved → {MULTISTEP_TASK_SET_PATH}")

        # 60/40 train/test split, stratified by type
        all_ids = [t.id for t in tasks]
        type_groups = {}
        for t in tasks:
            prefix = t.id.rsplit("_", 1)[0]  # e.g. "ms_a"
            type_groups.setdefault(prefix, []).append(t.id)

        train_ids, test_ids = [], []
        for group_ids in type_groups.values():
            random.shuffle(group_ids)
            split = max(1, round(len(group_ids) * 0.6))
            train_ids += group_ids[:split]
            test_ids += group_ids[split:]

        # Ensure every task is in base
        base_ids = all_ids
        split_data = {"train": train_ids, "test": test_ids, "base": base_ids}

        with open(MULTISTEP_SPLIT_PATH, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved → {MULTISTEP_SPLIT_PATH}")

    return tasks


def main():
    parser = ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_multistep_tasks(save=not args.no_save, seed=args.seed)


if __name__ == "__main__":
    main()
